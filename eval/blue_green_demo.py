"""
eval/blue_green_demo.py  —  Stage 7: blue-green reindexing (the safe rebuild).

Build a NEW index next to the live one, (pretend to) test it, FLIP a pointer to
it, and show that rollback is just flipping the pointer back. Self-contained; cleans up.
"""
import os

import psycopg
from dotenv import load_dotenv

load_dotenv(".env.local")
conn_str = os.environ["DATABASE_URL"]

with psycopg.connect(conn_str) as c:
    with c.cursor() as cur:
        # a tiny pointer that says which table is "live"
        cur.execute("CREATE TABLE IF NOT EXISTS app_config (key TEXT PRIMARY KEY, value TEXT)")
        cur.execute("INSERT INTO app_config VALUES ('active_index','chunks') ON CONFLICT (key) DO NOTHING")
        cur.execute("SELECT value FROM app_config WHERE key='active_index'")
        print("live index:", cur.fetchone()[0])

        # 1) BUILD the new index next to the live one (here a copy; in real life, re-embedded)
        cur.execute("DROP TABLE IF EXISTS chunks_v2")
        cur.execute("CREATE TABLE chunks_v2 (LIKE chunks INCLUDING ALL)")
        # copy only real columns — `tsv` is GENERATED and fills itself, so we can't write it
        cur.execute("INSERT INTO chunks_v2 (chunk_id, doc_id, chunk_index, text, n_chars, embedding, tenant_id) "
                    "SELECT chunk_id, doc_id, chunk_index, text, n_chars, embedding, tenant_id FROM chunks")
        cur.execute("SELECT count(*) FROM chunks_v2")
        print("built chunks_v2 with", cur.fetchone()[0], "rows  (live index untouched, users unaffected)")

        # 2) you'd run eval against chunks_v2 here; flip ONLY if recall holds
        print("eval chunks_v2 ...  assume recall@10 holds  ->  safe to flip")

        # 3) FLIP the pointer (instant switch to the new index)
        cur.execute("UPDATE app_config SET value='chunks_v2' WHERE key='active_index'")
        cur.execute("SELECT value FROM app_config WHERE key='active_index'")
        print("flipped. live index now:", cur.fetchone()[0])

        # 4) rollback = flip the pointer back (instant, no rebuild)
        cur.execute("UPDATE app_config SET value='chunks' WHERE key='active_index'")
        print("rolled back instantly by flipping the pointer back")

        # cleanup demo artifacts
        cur.execute("DROP TABLE IF EXISTS chunks_v2")
        cur.execute("DELETE FROM app_config WHERE key='active_index'")
print("cleaned up demo.")
