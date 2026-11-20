"""
storage/apply_schema.py
Applies storage/schema.sql to our Neon Postgres database.

Run once (safe to re-run) to:
  - enable the pgvector extension
  - create the `documents` table
Then it prints a quick health check, so we KNOW the database actually works.
"""
import os
from pathlib import Path

import psycopg                 # the Postgres driver
from dotenv import load_dotenv # reads key=value lines from a file into the environment

# `neon init` wrote our secrets into .env.local (gitignored). Load them.
load_dotenv(".env.local")

# For schema changes (DDL), Neon recommends the DIRECT (unpooled) connection.
# Fall back to the normal pooled URL if the unpooled one isn't present.
conn_str = os.environ.get("DATABASE_URL_UNPOOLED") or os.environ.get("DATABASE_URL")
if not conn_str:
    raise SystemExit("No DATABASE_URL found. Is .env.local present and filled?")

# Read the SQL file that sits right next to this script.
schema_sql = Path(__file__).with_name("schema.sql").read_text(encoding="utf-8")

# Connect, run the schema, then verify. `with` commits and closes for us.
with psycopg.connect(conn_str) as conn:
    with conn.cursor() as cur:
        cur.execute(schema_sql)  # CREATE EXTENSION vector; + CREATE TABLE documents

        # --- health checks -------------------------------------------------
        cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector';")
        row = cur.fetchone()
        pgvector_version = row[0] if row else None

        cur.execute("SELECT to_regclass('public.documents');")  # NULL if table missing
        documents_table = cur.fetchone()[0]

        cur.execute("SELECT count(*) FROM documents;")
        doc_count = cur.fetchone()[0]

print("Database connection : OK")
print(f"pgvector extension  : {'enabled v' + pgvector_version if pgvector_version else 'MISSING'}")
print(f"documents table     : {documents_table or 'MISSING'}")
print(f"documents rows      : {doc_count}")
