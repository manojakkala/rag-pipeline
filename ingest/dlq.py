"""
ingest/dlq.py  —  Stage 8: the dead-letter queue.

Process documents one at a time; if one FAILS, set it aside in `dead_letter`
and keep going — one bad file must not stop the whole ingest.
"""
import os

import psycopg
from dotenv import load_dotenv

load_dotenv(".env.local")
conn_str = os.environ["DATABASE_URL"]


def record_failure(doc_id, error) -> None:
    with psycopg.connect(conn_str) as c:
        with c.cursor() as cur:
            cur.execute("INSERT INTO dead_letter (doc_id, error) VALUES (%s, %s)",
                        (doc_id, str(error)[:300]))


def safe_process(items, process_fn):
    """items: list of (doc_id, data). process_fn may raise.
    Returns (ok_count, failed_ids); failures are logged to dead_letter, the run continues."""
    ok, failed = 0, []
    for doc_id, data in items:
        try:
            process_fn(doc_id, data)
            ok += 1
        except Exception as e:
            record_failure(doc_id, e)   # drop in the problems bin, keep going
            failed.append(doc_id)
    return ok, failed
