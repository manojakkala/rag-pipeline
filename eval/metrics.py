"""
eval/metrics.py  —  Stage 7: a simple health DASHBOARD built from the traces.

Reads eval/traces.jsonl and prints the few numbers you watch to know the system
is healthy: how many requests, which paths were used, time per step, and how
often it could NOT answer. A rising "can't answer" rate is often the first sign
something broke (e.g. documents stopped loading).
"""
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

TRACES = Path("eval/traces.jsonl")
rows = [json.loads(l) for l in TRACES.read_text(encoding="utf-8").splitlines() if l.strip()]
print(f"requests traced: {len(rows)}\n")

# --- which path did each request take? ---
routes = Counter(next((s.get("decision") for s in r["spans"] if s["stage"] == "route"), "?") for r in rows)
print("route mix:", dict(routes))

# --- time per step (average and p95 = "almost the worst case") ---
stage_ms = defaultdict(list)
for r in rows:
    for s in r["spans"]:
        stage_ms[s["stage"]].append(s["ms"])
print("\ntime per step (ms):")
for stage, vals in stage_ms.items():
    v = sorted(vals)
    p95 = v[min(len(v) - 1, int(0.95 * len(v)))]
    print(f"  {stage:16} avg={round(statistics.mean(vals)):>6}  p95={p95:>6}  n={len(vals)}")

# --- the key health gauge: how often it could NOT answer ---
grounded = [s["grounded"] for r in rows for s in r["spans"] if s["stage"] == "groundedness"]
if grounded:
    cant = sum(1 for g in grounded if not g)
    print(f'\n"can\'t answer" rate: {cant / len(grounded):.0%}  ({cant}/{len(grounded)})  <- WATCH THIS; a spike = trouble')

# --- alerts: draw a red line on the gauges and warn if crossed ---
print("\nalerts:")
alerts = []
if grounded and (cant / len(grounded)) > 0.20:
    alerts.append(f'"can\'t answer" rate {cant / len(grounded):.0%} is over the 20% line')
slow = stage_ms.get("retrieve_rerank", [])
if slow and max(slow) > 8000:
    alerts.append(f"retrieve_rerank hit {max(slow)}ms (over the 8000ms line)")
for a in alerts:
    print(f"  ALERT: {a}")
if not alerts:
    print("  all gauges green")
