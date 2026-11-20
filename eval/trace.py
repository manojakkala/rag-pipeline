"""
eval/trace.py  —  Stage 7: tracing / observability.

Runs ONE question through the pipeline, recording a SPAN per stage (what went
in/out, how long), writes one JSON trace to eval/traces.jsonl, and prints it.
This is the "black box": when an answer is wrong, you replay the trace to see
exactly which stage failed. Logs chunk IDs + scores, NEVER chunk text.

Run:  .venv\\Scripts\\python.exe eval\\trace.py "your question"
"""
import json
import os
import sys
import time
import uuid

sys.path.insert(0, "query")
sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

load_dotenv(".env.local")

from router import route
from rewrite import rewrite_query
from retrieve import retrieve_rerank
import generate

TRACES = os.path.join("eval", "traces.jsonl")


def trace(question: str, k: int = 5) -> dict:
    rec = {"request_id": uuid.uuid4().hex[:8], "question": question, "spans": []}

    def span(stage, t0, **fields):                       # record one stage
        rec["spans"].append({"stage": stage, "ms": round((time.time() - t0) * 1000), **fields})

    t = time.time(); decision = route(question); span("route", t, decision=decision)

    if decision == "RAG":
        t = time.time(); rq = rewrite_query(question); span("rewrite", t, rewritten=rq)

        t = time.time(); hits = retrieve_rerank(rq, k)
        span("retrieve_rerank", t,
             chunk_ids=[h[0] for h in hits],              # IDs, not text
             doc_ids=[h[1] for h in hits],
             scores=[round(float(h[2]), 3) for h in hits])

        blocks = [f"[{i}] (source: {h[1]})\n{h[3]}" for i, h in enumerate(hits, 1)]
        context = "\n\n".join(blocks)

        t = time.time()
        out = generate.client.chat.completions.create(
            model=generate.MODEL, temperature=0,
            messages=[{"role": "system", "content": generate.ANSWER_SYSTEM},
                      {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}],
        ).choices[0].message.content
        span("generate", t, model=generate.MODEL.split("/")[-1], answer_preview=out[:120])

        t = time.time(); grounded, _ = generate.check_groundedness(context, out)
        span("groundedness", t, grounded=grounded)

    rec["total_ms"] = sum(s["ms"] for s in rec["spans"])
    with open(TRACES, "a", encoding="utf-8") as f:        # one JSON line per request
        f.write(json.dumps(rec) + "\n")
    with open(TRACES, encoding="utf-8") as f:             # retention: keep only the last 500 traces
        lines = f.read().splitlines()
    if len(lines) > 500:
        with open(TRACES, "w", encoding="utf-8") as f:
            f.write("\n".join(lines[-500:]) + "\n")
    return rec


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else \
        "How do I let a workflow access cloud resources without storing long-lived keys?"
    print(json.dumps(trace(q), indent=2))
