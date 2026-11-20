"""
query/retrieve.py  —  Stage 1: VECTOR SEARCH (Pipeline B, step 1)

Given a question, embed it with the SAME model, then find the nearest chunks
by cosine similarity. This is "semantic search": we match by meaning, not words.

Production detail baked in: we embed the QUESTION with `query_embed`, not
`embed`. bge models expect a short instruction on the query side only
(query ≠ passage). Getting this asymmetry right measurably improves recall.

Usage:  .venv\\Scripts\\python.exe query\\retrieve.py "how do I authenticate?"
"""
import sys

from fastembed import TextEmbedding

from db import pool as DB          # shared connection pool (register_vector runs per connection)

TOP_K = 5
model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")  # MUST match the ingest model


def retrieve(question: str, k: int = TOP_K):
    # query_embed adds the retrieval instruction bge wants on the query side.
    qvec = list(model.query_embed([question]))[0]
    with DB.connection() as conn:
        with conn.cursor() as cur:
            # `<=>` is pgvector's cosine-DISTANCE operator (smaller = closer).
            # score = 1 - distance = cosine SIMILARITY (bigger = more similar).
            cur.execute(
                """
                SELECT chunk_id, doc_id,
                       1 - (embedding <=> %s) AS score,
                       text
                FROM chunks
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                (qvec, qvec, k),
            )
            return cur.fetchall()


# --- Hybrid search: vector + keyword, fused with RRF --------------------------

def _vector_ids(cur, qvec, n):
    """Top-n chunk ids by vector (meaning) similarity."""
    cur.execute("SELECT chunk_id, doc_id FROM chunks ORDER BY embedding <=> %s LIMIT %s", (qvec, n))
    return cur.fetchall()


def _keyword_ids(cur, question, n):
    """Top-n chunk ids by keyword (exact-word) match, Postgres full-text."""
    cur.execute(
        "SELECT chunk_id, doc_id FROM chunks "
        "WHERE tsv @@ plainto_tsquery('english', %s) "
        "ORDER BY ts_rank_cd(tsv, plainto_tsquery('english', %s)) DESC LIMIT %s",
        (question, question, n),
    )
    return cur.fetchall()


def retrieve_hybrid(question: str, k: int = 5, pool: int = 50, rrf_k: int = 60):
    """Run BOTH searches (wide net = `pool`), fuse by rank with RRF, return top k.
    RRF: a doc's score = sum of 1/(rrf_k + rank) over each list it appears in."""
    qvec = list(model.query_embed([question]))[0]
    with DB.connection() as conn:
        with conn.cursor() as cur:
            lists = [_vector_ids(cur, qvec, pool), _keyword_ids(cur, question, pool)]

    scores, doc_of = {}, {}
    for ranked_list in lists:
        for rank, (cid, did) in enumerate(ranked_list, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
            doc_of[cid] = did
    top = sorted(scores, key=scores.get, reverse=True)[:k]
    return [(cid, doc_of[cid], scores[cid]) for cid in top]


# --- Reranking: a cross-encoder re-reads the candidates and reorders them ------

_reranker = None


def _get_reranker():
    """Load the reranker once, only when first needed (downloads ~90MB on first run)."""
    global _reranker
    if _reranker is None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        _reranker = TextCrossEncoder(model_name="Xenova/ms-marco-MiniLM-L-6-v2")
    return _reranker


def retrieve_rerank(question: str, k: int = 5, pool: int = 50, cand: int = 30):
    """Wide net via hybrid search, then a cross-encoder re-reads the top `cand`
    candidates (question + chunk together) and keeps the best k."""
    fused = retrieve_hybrid(question, k=cand, pool=pool)   # [(chunk_id, doc_id, score), ...]
    cids = [cid for cid, _, _ in fused]
    if not cids:
        return []
    with DB.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT chunk_id, doc_id, text FROM chunks WHERE chunk_id = ANY(%s)", (cids,))
            rows = {cid: (did, text) for cid, did, text in cur.fetchall()}
    cands = [(cid, rows[cid][0], rows[cid][1]) for cid in cids if cid in rows]
    scores = list(_get_reranker().rerank(question, [text for _, _, text in cands]))
    order = sorted(zip(cands, scores), key=lambda x: x[1], reverse=True)[:k]
    return [(cid, did, float(score), text) for (cid, did, text), score in order]


def retrieve_hybrid_text(question: str, k: int = 5, pool: int = 50):
    """Hybrid search (vector+BM25+RRF) that ALSO returns each chunk's text — but NO reranker.
    Used on memory-constrained hosts (e.g. a 512MB free tier) where we skip the 2nd model.
    Returns the same 4-tuple shape as retrieve_rerank: (chunk_id, doc_id, score, text)."""
    fused = retrieve_hybrid(question, k=k, pool=pool)          # [(chunk_id, doc_id, score), ...]
    cids = [cid for cid, _, _ in fused]
    if not cids:
        return []
    with DB.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT chunk_id, text FROM chunks WHERE chunk_id = ANY(%s)", (cids,))
            text_of = {cid: text for cid, text in cur.fetchall()}
    return [(cid, did, score, text_of.get(cid, "")) for cid, did, score in fused]


def retrieve_rewrite(question: str, k: int = 5, pool: int = 50, cand: int = 30):
    """Rewrite the question into the docs' terminology first, then retrieve+rerank."""
    from rewrite import rewrite_query
    better_query = rewrite_query(question)
    return retrieve_rerank(better_query, k, pool, cand)


def retrieve_scoped(question: str, allowed_tenants: list[str], k: int = 5):
    """Vector search with a PRE-FILTER: only chunks whose tenant_id is allowed.
    The filter lives INSIDE the query, so forbidden chunks are never even considered."""
    qvec = list(model.query_embed([question]))[0]
    with DB.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT chunk_id, doc_id, tenant_id, 1 - (embedding <=> %s) AS score "
                "FROM chunks WHERE tenant_id = ANY(%s) ORDER BY embedding <=> %s LIMIT %s",
                (qvec, allowed_tenants, qvec, k),
            )
            return cur.fetchall()


if __name__ == "__main__":
    question = sys.argv[1] if len(sys.argv) > 1 else "How do I authenticate to the REST API?"
    print(f"Q: {question}\n")
    for chunk_id, doc_id, score, text in retrieve(question):
        print(f"[{score:.3f}] {doc_id}")
        print(f"        {text.strip().replace(chr(10), ' ')[:150]}...\n")
