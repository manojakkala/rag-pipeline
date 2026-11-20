"""
ingest/embed.py  —  Stage 1: EMBEDDINGS (Pipeline A, step 3)

Turns each chunk's text into a 384-dimension vector using a local model
(BAAI/bge-small-en-v1.5 via `fastembed` — free, no API key, runs on CPU),
and stores that vector on the chunk row.

Idempotent + cheap: we only embed chunks whose `embedding` is still NULL, so
re-running skips work already done (that's the "skip unchanged content" idea).

First run downloads the model (~130 MB) once, then caches it.
Run:  .venv\\Scripts\\python.exe ingest\\embed.py
"""
import os

import psycopg
from pgvector.psycopg import register_vector  # maps Python vectors <-> the vector column
from fastembed import TextEmbedding
from dotenv import load_dotenv

load_dotenv(".env.local")
conn_str = os.environ["DATABASE_URL"]

BATCH = 64  # embed in batches, not one-at-a-time — far better throughput

# Load the model once (downloads on first run, then cached locally).
model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

with psycopg.connect(conn_str) as conn:
    register_vector(conn)
    with conn.cursor() as cur:
        # Only embed chunks we haven't embedded yet.
        cur.execute("SELECT chunk_id, text FROM chunks WHERE embedding IS NULL ORDER BY chunk_id")
        rows = cur.fetchall()
        print(f"{len(rows)} chunks need embedding")

        for start in range(0, len(rows), BATCH):
            batch = rows[start:start + BATCH]
            texts = [text for _, text in batch]
            vectors = list(model.embed(texts))  # one numpy vector per input text
            for (chunk_id, _), vec in zip(batch, vectors):
                cur.execute(
                    "UPDATE chunks SET embedding = %s WHERE chunk_id = %s",
                    (vec, chunk_id),
                )
            print(f"  embedded {min(start + BATCH, len(rows))}/{len(rows)}")

print("Done.")
