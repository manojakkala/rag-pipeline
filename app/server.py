"""
app/server.py  —  a small web backend for the chat UI.

Serves the chat page and a /ask endpoint that runs the pipeline WITH tracing,
returning the answer PLUS the per-step trace so the UI can show observability.
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
os.chdir(ROOT)                                  # so .env.local / data resolve
sys.path.insert(0, os.path.join(ROOT, "query"))

from flask import Flask, request, jsonify, send_file
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


def traced_ask(question: str, history=None) -> dict:
    """Run the real pipeline while recording a span per step.
    `history` (recent {role, content} turns) is what makes follow-up questions work."""
    spans = []

    def span(stage, t0, **info):
        spans.append({"stage": stage, "ms": round((time.time() - t0) * 1000), **info})

    if guardrails.looks_like_injection(question):        # input guardrail (C3) — on the RAW question
        return {"answer": "That looks like a prompt-injection attempt, so I won't act on it.",
                "route": "BLOCKED", "grounded": None, "sources": [], "spans": spans}

    # Follow-up support: turn "what about python?" into a standalone question using the history.
    q = question
    if history:
        t = time.time(); q = contextualize.contextualize(history, question)
        span("contextualize", t, standalone=q)

    t = time.time(); decision = router.route(q); span("route", t, decision=decision)

    if decision == "CHAT":
        t = time.time(); ans = generate.chat_reply(q); span("chat", t)
        return {"answer": ans, "route": decision, "grounded": None, "sources": [], "spans": spans}

    if decision == "DECLINE":
        return {"answer": "That's outside what I can help with — GitHub docs and live GitHub data only.",
                "route": decision, "grounded": None, "sources": [], "spans": spans}

    if decision == "API":
        t = time.time(); ans = github_api.answer_api(q); span("live_api", t)
        return {"answer": ans, "route": decision, "grounded": None,
                "sources": [{"id": "live GitHub API", "url": None}], "spans": spans}

    # --- RAG path ---
    t = time.time(); rq = rewrite.rewrite_query(q); span("rewrite", t, rewritten=rq)

    t = time.time()
    if LITE:                                             # memory-constrained host: hybrid only
        hits = retrieve.retrieve_hybrid_text(rq)
        stage = "retrieve (hybrid)"
    else:                                                # full quality: hybrid + cross-encoder rerank
        hits = retrieve.retrieve_rerank(rq)
        stage = "retrieve+rerank"
    span(stage, t, chunks=[{"doc": h[1], "score": round(float(h[2]), 3)} for h in hits])

    blocks = [f"[{i}] (source: {h[1]})\n{h[3]}" for i, h in enumerate(hits, 1)]
    context = "\n\n".join(blocks)

    t = time.time(); out, grounded, _ = generate.answer_with_check(context, q)  # shared logic
    span("generate+check", t, model=generate.MODEL.split("/")[-1], grounded=grounded)

    source_list = [{"id": h[1], "url": source_url(h[1])} for h in hits]   # clickable citations
    return {"answer": out, "route": decision, "grounded": grounded,
            "sources": source_list, "spans": spans}


@app.route("/")
def index():
    return send_file(os.path.join(HERE, "index.html"))


@app.route("/ask", methods=["POST"])
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


if __name__ == "__main__":
    # Containers/Hugging Face Spaces inject PORT; locally we default to 5000.
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, threaded=True)
