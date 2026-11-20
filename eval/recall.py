"""
eval/recall.py  —  Stage 2/3: measure RETRIEVAL quality.

Usage:
  .venv\\Scripts\\python.exe eval\\recall.py          # vector-only (default)
  .venv\\Scripts\\python.exe eval\\recall.py hybrid   # vector + keyword (RRF)

Reports recall@5, recall@10, MRR over answerable questions, plus a no-answer
check (top cosine score should be LOW for out-of-corpus questions).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "query"))
from retrieve import retrieve, retrieve_hybrid, retrieve_rerank, retrieve_rewrite

MODE = sys.argv[1] if len(sys.argv) > 1 else "vector"
retriever = {"hybrid": retrieve_hybrid, "rerank": retrieve_rerank,
             "rewrite": retrieve_rewrite}.get(MODE, retrieve)

GOLDEN = Path(__file__).with_name("golden_set.jsonl")
K = 10
GUARD = 0.30

items = [json.loads(l) for l in GOLDEN.read_text(encoding="utf-8").splitlines() if l.strip()]
answerable = [it for it in items if it["answer_doc_ids"]]
no_answer = [it for it in items if not it["answer_doc_ids"]]

print(f"=== retriever: {MODE} ===")
hits5 = hits10 = 0
rr_sum = 0.0
for it in answerable:
    gold = set(it["answer_doc_ids"])
    results = retriever(it["question"], K)          # rows where [1] is always doc_id
    doc_ids = [row[1] for row in results]
    rank = next((i + 1 for i, d in enumerate(doc_ids) if d in gold), None)
    if rank and rank <= 5:
        hits5 += 1
    if rank and rank <= 10:
        hits10 += 1
    rr_sum += (1.0 / rank) if rank else 0.0
    print(f"{(str(rank) if rank else 'miss'):>4}  {it['question'][:62]}")

n = len(answerable)
print("-" * 72)
print(f"recall@5 : {hits5}/{n} = {hits5 / n:.0%}")
print(f"recall@10: {hits10}/{n} = {hits10 / n:.0%}")
print(f"MRR      : {rr_sum / n:.3f}")

# No-answer check always uses the vector cosine score (meaningful threshold).
if no_answer:
    print("\nNo-answer (vector cosine top score should be < 0.30 to refuse):")
    for it in no_answer:
        results = retrieve(it["question"], K)
        top = results[0][2] if results else 0.0
        verdict = "OK (would refuse)" if top < GUARD else "LEAK (would answer!)"
        print(f"  top={top:.3f}  {verdict:<20}  {it['question'][:44]}")
