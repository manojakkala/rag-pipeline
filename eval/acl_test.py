"""
eval/acl_test.py  —  Stage 6 security: multi-tenant isolation (the cross-tenant test).

Seeds two synthetic tenants (acme, globex), each with a secret chunk, then proves a
search scoped to one tenant NEVER returns the other's chunk — even when the other's
chunk is the best semantic match. Cleans up afterward.
"""
import os
import sys

sys.path.insert(0, "query")

import psycopg
from pgvector.psycopg import register_vector
from dotenv import load_dotenv

from retrieve import model, retrieve_scoped

load_dotenv(".env.local")
conn_str = os.environ["DATABASE_URL"]

SECRETS = {
    "acme":   "ACME internal: the product launch code is ALPHA-SEVEN.",
    "globex": "GLOBEX internal: the vault PIN is 4821.",
}

# --- seed the two tenants' secret chunks ---
with psycopg.connect(conn_str) as c:
    register_vector(c)
    with c.cursor() as cur:
        for tenant, text in SECRETS.items():
            vec = list(model.embed([text]))[0]
            cur.execute("INSERT INTO documents (doc_id, source, raw_text) VALUES (%s,'acl-demo',%s) "
                        "ON CONFLICT (doc_id) DO UPDATE SET raw_text=EXCLUDED.raw_text",
                        (f"ACL_{tenant}", text))
            cur.execute("DELETE FROM chunks WHERE doc_id=%s", (f"ACL_{tenant}",))
            cur.execute("INSERT INTO chunks (chunk_id, doc_id, chunk_index, text, n_chars, embedding, tenant_id) "
                        "VALUES (%s,%s,0,%s,%s,%s,%s)",
                        (f"ACL_{tenant}::0", f"ACL_{tenant}", text, len(text), vec, tenant))

print("Seeded two tenants: acme, globex.\n")

# --- THE CROSS-TENANT TEST: as ACME, search for GLOBEX's secret. Must leak ZERO globex chunks. ---
rows = retrieve_scoped("what is the vault PIN?", allowed_tenants=["acme", "public"], k=5)
leaked = [r for r in rows if r[2] == "globex"]
print("As ACME, searching for GLOBEX's secret ('vault PIN'):")
for r in rows[:3]:
    print(f"   tenant={r[2]:7} score={r[3]:.3f}  {r[0]}")
print(f"   -> globex chunks leaked: {len(leaked)}  =>  {'PASS (isolated)' if not leaked else 'FAIL (LEAK!)'}\n")

# --- positive control: ACME can still find its OWN secret ---
rows2 = retrieve_scoped("what is the launch code?", allowed_tenants=["acme", "public"], k=5)
print(f"As ACME, searching own secret ('launch code'): found acme chunk = {any(r[2]=='acme' for r in rows2)}")

# --- cleanup ---
with psycopg.connect(conn_str) as c:
    with c.cursor() as cur:
        for tenant in SECRETS:
            cur.execute("DELETE FROM chunks WHERE doc_id=%s", (f"ACL_{tenant}",))
            cur.execute("DELETE FROM documents WHERE doc_id=%s", (f"ACL_{tenant}",))
print("\nCleaned up demo tenants.")
