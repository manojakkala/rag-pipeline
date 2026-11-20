"""
ingest/load.py  —  Stage 1: RAW LOAD (Pipeline A, step 1)

Reads every .md file in data/corpus/ and stores it UNCHANGED in the
`documents` table. No chunking, no cleaning yet — we store the raw truth first,
so later we can measure what cleaning/chunking actually buys us.

Run it:  .venv\\Scripts\\python.exe ingest\\load.py   (from the project root)
Safe to run twice — see the UPSERT note below.
"""
import hashlib
import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv

# Load DATABASE_URL from .env.local (created by `neon init`, gitignored).
load_dotenv(".env.local")
# For normal app reads/writes we use the POOLED url (fine for lots of short queries).
conn_str = os.environ["DATABASE_URL"]

CORPUS_DIR = Path("data/corpus")
files = sorted(CORPUS_DIR.glob("*.md"))
print(f"Found {len(files)} markdown files in {CORPUS_DIR}")

loaded = skipped = 0
with psycopg.connect(conn_str) as conn:
    with conn.cursor() as cur:
        for f in files:
            # errors="replace": if a byte is weird, don't crash — swap it for a placeholder.
            raw_text = f.read_text(encoding="utf-8", errors="replace")
            doc_id = f.name                 # stable id = the (unique) filename
            source = f.as_posix()           # where it came from, for citations later
            content_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()  # fingerprint

            # Change detection: if the content hash matches what's stored, skip this doc.
            cur.execute("SELECT content_hash FROM documents WHERE doc_id = %s", (doc_id,))
            row = cur.fetchone()
            if row and row[0] == content_hash:
                skipped += 1
                continue

            # UPSERT (idempotent) — store the new content hash too.
            # %s placeholders are filled by psycopg SAFELY — never string-format SQL yourself.
            cur.execute(
                """
                INSERT INTO documents (doc_id, source, raw_text, content_hash)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (doc_id) DO UPDATE
                    SET raw_text     = EXCLUDED.raw_text,
                        source       = EXCLUDED.source,
                        content_hash = EXCLUDED.content_hash,
                        loaded_at    = now();
                """,
                (doc_id, source, raw_text, content_hash),
            )
            loaded += 1

print(f"Loaded {loaded} documents ({skipped} unchanged, skipped).")
