"""
eval/gate.py  —  Stage 7: the CI quality gate.

Runs the golden-set eval and EXITS NON-ZERO if recall@10 falls below THRESHOLD,
so a change that regresses retrieval FAILS the build instead of silently shipping.
CI (GitHub Actions) runs this on every push — a red build means "don't merge".
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "query")
from retrieve import retrieve_rerank

THRESHOLD = 0.85   # just below our current baseline (rerank recall@10 ~0.90)
K = 10
GOLDEN = Path("eval/golden_set.jsonl")

items = [json.loads(l) for l in GOLDEN.read_text(encoding="utf-8").splitlines() if l.strip()]
answerable = [it for it in items if it["answer_doc_ids"]]

hits = 0
for it in answerable:
    gold = set(it["answer_doc_ids"])
    doc_ids = [row[1] for row in retrieve_rerank(it["question"], K)]
    if any(d in gold for d in doc_ids):
        hits += 1

recall = hits / len(answerable)
print(f"recall@{K} = {recall:.0%}  (threshold {THRESHOLD:.0%}, n={len(answerable)})")

if recall < THRESHOLD:
    print("FAIL - quality regressed; blocking the change.")
    sys.exit(1)       # non-zero exit = red build = merge blocked
print("PASS - quality gate OK.")
sys.exit(0)
