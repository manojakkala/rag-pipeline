"""
app/server.py  —  a small web backend for the chat UI.

Serves the chat page and a /ask endpoint that runs the pipeline WITH tracing,
returning the answer PLUS the per-step trace so the UI can show observability.
"""
import os
import sys
import json
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
os.chdir(ROOT)                                  # so .env.local / data resolve
sys.path.insert(0, os.path.join(ROOT, "query"))

from flask import Flask, request, jsonify, send_file, Response
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.middleware.proxy_fix import ProxyFix
from dotenv import load_dotenv

load_dotenv(".env.local")

import router
import rewrite
import retrieve
import generate
import github_api
import guardrails
import contextualize
from sources import source_url

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)   # trust Render's proxy → real client IP

# Rate limit the public endpoint so a traffic spike can't drain the Groq quota.
# In-memory storage is fine here because we run a single gunicorn worker.
limiter = Limiter(get_remote_address, app=app, default_limits=[])


@app.errorhandler(429)
def _rate_limited(e):
    return jsonify({"answer": "You're sending requests too quickly — please wait a moment and try again.",
                    "route": "rate-limited", "grounded": None, "sources": [], "spans": []}), 429


# On tiny hosts (e.g. a 512MB free tier) loading BOTH models can run out of memory.
# LITE_RETRIEVAL=1 skips the cross-encoder reranker: we still do hybrid (vector+BM25) search,
# just without the extra rerank model — so the whole app fits in ~512MB.
LITE = bool(os.environ.get("LITE_RETRIEVAL"))


def _warmup():
    """Load + warm the ML model(s) once at startup so the FIRST real request is fast.
    (No DB needed here — we just wake the models from cold.)"""
    try:
        list(retrieve.model.query_embed(["warm up"]))                       # wake the embedder
        if not LITE:
            list(retrieve._get_reranker().rerank("warm up", ["warm up"]))   # wake the reranker
        print(f"[startup] models warmed up (lite={LITE})", file=sys.stderr)
    except Exception as e:                                                  # never let warmup crash boot
        print("[startup] warmup skipped:", repr(e), file=sys.stderr)


_warmup()


NO_HITS = "I couldn't find anything about that in the documents."


def _classify(question, history, span):
    """Shared front of the pipeline: injection guard, route (history-aware), and — only for the
    retrieval lanes — contextualize the follow-up. Returns (decision, q) where decision is one of
    RAG / API / CHAT / DECLINE / BLOCKED and q is the (possibly contextualized) question.
    Routing happens on the RAW message so small talk is never dragged into the topic by a rewrite."""
    if guardrails.looks_like_injection(question):          # input guardrail — on the RAW question
        return "BLOCKED", question
    t = time.time(); decision = router.route(question, history); span("route", t, decision=decision)
    if decision in ("CHAT", "DECLINE"):                     # need neither history nor retrieval
        return decision, question
    q = question
    if history:                                            # RAG/API follow-up → make it standalone
        t = time.time(); q = contextualize.contextualize(history, question); span("contextualize", t, standalone=q)
        if guardrails.looks_like_injection(q):             # re-check the history-derived question
            return "BLOCKED", q
    return decision, q


def _rag_context(q, span):
    """Shared RAG preparation: semantic-cache lookup, then (on a miss) rewrite + retrieve + build the
    context. Returns (cached_payload | None, context, source_list, candidates, hits). On a cache hit,
    context/candidates/hits are None and source_list comes from the cache."""
    t = time.time(); cached = generate.cache_get(q); span("cache hit" if cached else "cache miss", t)
    if cached:
        return cached, None, cached.get("sources") or [], [], None

    t = time.time(); rq = rewrite.rewrite_query(q); span("rewrite", t, rewritten=rq)

    t = time.time(); candidates = []
    if LITE:                                               # memory-constrained host: hybrid only
        hits = retrieve.retrieve_hybrid_text(rq); stage = "retrieve (hybrid)"
    else:                                                  # full quality: hybrid + cross-encoder rerank
        hits, fused = retrieve.retrieve_rerank(rq, return_candidates=True); stage = "retrieve+rerank"
        candidates = [{"doc": did, "score": round(float(sc), 4)} for _, did, sc in fused[:12]]

    blocks = [f"[{i}] (source: {h[1]})\n{h[3]}" for i, h in enumerate(hits, 1)]
    context = "\n\n".join(blocks)
    source_list = [{"id": h[1], "url": source_url(h[1])} for h in hits]
    # Deeper trace: the pre-rerank candidate pool, each kept chunk's text, and the full context.
    span(stage, t,
         chunks=[{"doc": h[1], "score": round(float(h[2]), 3), "text": " ".join(h[3].split())[:280]} for h in hits],
         candidates=candidates, context=context[:6000])
    return None, context, source_list, candidates, hits


def traced_ask(question: str, history=None) -> dict:
    """Non-streaming pipeline (the retry-quality path) with a per-step trace.
    `history` (recent {role, content} turns) is what makes follow-up questions work."""
    spans = []

    def span(stage, t0, **info):
        spans.append({"stage": stage, "ms": round((time.time() - t0) * 1000), **info})

    decision, q = _classify(question, history, span)

    if decision == "BLOCKED":
        return {"answer": "That looks like a prompt-injection attempt, so I won't act on it.",
                "route": "BLOCKED", "grounded": None, "sources": [], "spans": spans}
    if decision == "CHAT":
        t = time.time(); ans = generate.chat_reply(question); span("chat", t)
        return {"answer": ans, "route": decision, "grounded": None, "sources": [], "spans": spans}
    if decision == "DECLINE":
        return {"answer": "That's outside what I can help with — GitHub docs and live GitHub data only.",
                "route": decision, "grounded": None, "sources": [], "spans": spans}
    if decision == "API":
        t = time.time(); ans = github_api.answer_api(q); span("live_api", t)
        return {"answer": ans, "route": decision, "grounded": None,
                "sources": [{"id": "live GitHub API", "url": None}], "spans": spans}

    # --- RAG ---
    cached, context, source_list, candidates, hits = _rag_context(q, span)
    if cached:
        return {"answer": cached["answer"], "route": decision, "grounded": cached.get("grounded"),
                "sources": source_list, "spans": spans}
    if not hits:                                           # nothing retrieved → honest no-answer
        return {"answer": NO_HITS, "route": decision, "grounded": None, "sources": [], "spans": spans}

    t = time.time(); out, grounded, _ = generate.answer_with_check(context, q)
    span("generate+check", t, model=generate.MODEL.split("/")[-1], grounded=grounded)
    generate.cache_put(q, out, source_list, grounded)      # store for next time (best-effort)
    return {"answer": out, "route": decision, "grounded": grounded, "sources": source_list, "spans": spans}


@app.route("/")
def index():
    return send_file(os.path.join(HERE, "index.html"))


@app.route("/ask", methods=["POST"])
@limiter.limit("15 per minute; 150 per hour")               # per-IP fairness; friendly 429 handled above
@limiter.limit("40 per minute; 400 per hour", key_func=lambda: "global")  # GLOBAL cap → protects the Groq quota
def ask():
    payload = request.get_json(silent=True) or {}
    q = (payload.get("question") or "").strip()
    history = payload.get("history") or []               # recent turns for follow-up questions
    if not q:
        return jsonify({"answer": "Ask me something.", "route": "-", "grounded": None, "sources": [], "spans": []})
    try:
        return jsonify(traced_ask(q, history))
    except Exception as e:                                # error safety net (C2)
        print("ask error:", repr(e), file=sys.stderr)
        return jsonify({"answer": "Sorry — something went wrong (the model service may be busy or rate-limited). Please try again.",
                        "route": "error", "grounded": None, "sources": [], "spans": []})


def _sse(obj):
    """One Server-Sent Event line."""
    return f"data: {json.dumps(obj)}\n\n"


def stream_ask(question: str, history=None):
    """Streaming pipeline: yields SSE events so the UI renders the answer as it is generated. Events:
    {type:'meta', route, sources, spans} before generation, many {type:'token', text} during, and one
    {type:'done', route, grounded, sources, spans} at the end. Shares _classify / _rag_context with the
    non-streaming traced_ask so the two can't drift apart. The groundedness check runs AFTER the stream
    (we can't un-say streamed text), so a RAG answer is verified and flagged, not silently re-drafted."""
    spans = []

    def span(stage, t0, **info):
        spans.append({"stage": stage, "ms": round((time.time() - t0) * 1000), **info})

    def whole(text, route, grounded=None, sources=None, meta_sources=None):
        """Emit a non-token-streamed answer (blocked / decline / cached / no-hits / API) through the
        same protocol. A meta event is sent only when meta_sources is given."""
        if meta_sources is not None:
            yield _sse({"type": "meta", "route": route, "sources": meta_sources, "spans": spans})
        yield _sse({"type": "token", "text": text})
        yield _sse({"type": "done", "route": route, "grounded": grounded, "sources": sources or [], "spans": spans})

    decision, q = _classify(question, history, span)

    if decision == "BLOCKED":
        yield from whole("That looks like a prompt-injection attempt, so I won't act on it.", "BLOCKED"); return
    if decision == "DECLINE":
        yield from whole("That's outside what I can help with — GitHub docs and live GitHub data only.", decision); return
    if decision == "CHAT":
        yield _sse({"type": "meta", "route": decision, "sources": [], "spans": spans})
        t = time.time()
        for tok in generate.stream_chat(question):
            yield _sse({"type": "token", "text": tok})
        span("chat", t)
        yield _sse({"type": "done", "route": decision, "grounded": None, "sources": [], "spans": spans}); return
    if decision == "API":
        srcs = [{"id": "live GitHub API", "url": None}]
        yield _sse({"type": "meta", "route": decision, "sources": srcs, "spans": spans})
        t = time.time(); ans = github_api.answer_api(q); span("live_api", t)
        yield _sse({"type": "token", "text": ans})          # API answer isn't token-streamed; send whole
        yield _sse({"type": "done", "route": decision, "grounded": None, "sources": srcs, "spans": spans}); return

    # --- RAG ---
    cached, context, source_list, candidates, hits = _rag_context(q, span)
    if cached:
        yield from whole(cached["answer"], decision, grounded=cached.get("grounded"),
                         sources=source_list, meta_sources=source_list); return
    if not hits:                                            # nothing retrieved → honest no-answer
        yield from whole(NO_HITS, decision, meta_sources=[]); return

    # Sources go out BEFORE the tokens so the UI can turn [1],[2]… into live links as they stream.
    yield _sse({"type": "meta", "route": decision, "sources": source_list, "spans": spans})

    t = time.time(); acc = ""
    for tok in generate.stream_draft(context, q):
        acc += tok
        yield _sse({"type": "token", "text": tok})
    grounded, verdict = generate.check_groundedness(context, acc)  # verify AFTER streaming
    span("generate(stream)+check", t, grounded=grounded, verdict=verdict)
    generate.cache_put(q, acc, source_list, grounded)             # cache the streamed answer
    yield _sse({"type": "done", "route": decision, "grounded": grounded, "sources": source_list, "spans": spans})


@app.route("/ask_stream", methods=["POST"])
@limiter.limit("15 per minute; 150 per hour")
@limiter.limit("40 per minute; 400 per hour", key_func=lambda: "global")
def ask_stream():
    payload = request.get_json(silent=True) or {}
    q = (payload.get("question") or "").strip()
    history = payload.get("history") or []

    def gen():
        if not q:
            yield _sse({"type": "token", "text": "Ask me something."})
            yield _sse({"type": "done", "route": "-", "grounded": None, "sources": [], "spans": []})
            return
        try:
            yield from stream_ask(q, history)
        except Exception as e:                               # error safety net, mid-stream
            print("ask_stream error:", repr(e), file=sys.stderr)
            yield _sse({"type": "token", "text": "Sorry — something went wrong (the model service may be busy or rate-limited). Please try again."})
            yield _sse({"type": "done", "route": "error", "grounded": None, "sources": [], "spans": []})

    return Response(gen(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


if __name__ == "__main__":
    # Containers/Hugging Face Spaces inject PORT; locally we default to 5000.
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, threaded=True)
