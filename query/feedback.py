"""
query/feedback.py  —  Stage 7: collect user feedback (thumbs up/down).

Each thumb is saved. Over time this data shows which retrieved chunks actually
helped, and can be used to RETRAIN the reranker so ranking keeps improving (the
compounding loop). Retraining itself is a bigger job; here we build the COLLECTION.
"""
import os

import psycopg
from dotenv import load_dotenv

load_dotenv(".env.local")
conn_str = os.environ["DATABASE_URL"]


def record(question: str, chunk_id: str, thumb: str) -> None:
    """Save one piece of feedback. thumb is 'up' or 'down'."""
    with psycopg.connect(conn_str) as c:
        with c.cursor() as cur:
            cur.execute("INSERT INTO feedback (question, chunk_id, thumb) VALUES (%s, %s, %s)",
                        (question, chunk_id, thumb))


def summary() -> dict:
    """Totals so far — the raw material for retraining the reranker later."""
    with psycopg.connect(conn_str) as c:
        with c.cursor() as cur:
            cur.execute("SELECT thumb, count(*) FROM feedback GROUP BY thumb")
            return dict(cur.fetchall())
