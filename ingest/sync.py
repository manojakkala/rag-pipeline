"""
ingest/sync.py  —  Stage 8: keep the index in sync with the source, incl. DELETES.

reconcile(seen_ids): compare the FULL set of source IDs we see NOW against what's
in the index. Anything in the index but NOT seen = deleted at the source -> remove
it (and its chunks). This is how a deletion propagates instead of lingering forever.

Safety: an EMPTY crawl is treated as a failure, not "delete everything".
"""
import hashlib
import os

import psycopg
from dotenv import load_dotenv

load_dotenv(".env.local")
conn_str = os.environ["DATABASE_URL"]


def reconcile(seen_ids) -> list[str]:
    """Delete docs (and their chunks) that are in the index but no longer at the source."""
    seen = list(seen_ids)
    if not seen:
        return []   # empty crawl = probably a failure, NOT a signal to delete all

    with psycopg.connect(conn_str) as c:
        with c.cursor() as cur:
            cur.execute("SELECT doc_id FROM documents WHERE doc_id <> ALL(%s)", (seen,))
            gone = [r[0] for r in cur.fetchall()]
            for doc_id in gone:
                cur.execute("DELETE FROM chunks WHERE doc_id = %s", (doc_id,))     # chunks first
                cur.execute("DELETE FROM documents WHERE doc_id = %s", (doc_id,))  # then the doc
    return gone


# --- change detection: a content fingerprint, so unchanged docs are skipped ---

def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def is_changed(doc_id: str, text: str):
    """True if this doc's content differs from its stored fingerprint. Returns (changed, new_hash)."""
    h = content_hash(text)
    with psycopg.connect(conn_str) as c:
        with c.cursor() as cur:
            cur.execute("SELECT content_hash FROM documents WHERE doc_id = %s", (doc_id,))
            row = cur.fetchone()
    old = row[0] if row else None
    return (old != h), h


def set_hash(doc_id: str, h: str) -> None:
    """Store the fingerprint after a doc has been processed."""
    with psycopg.connect(conn_str) as c:
        with c.cursor() as cur:
            cur.execute("UPDATE documents SET content_hash = %s WHERE doc_id = %s", (h, doc_id))
