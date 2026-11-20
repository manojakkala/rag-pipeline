-- storage/schema.sql
-- The shared "storage layer" between our two pipelines.
-- Pipeline A (ingestion) WRITES here. Pipeline B (query) only READS.
-- Running this file is safe to repeat: every statement uses IF NOT EXISTS.

-- pgvector: adds a `vector` column type so we can store embeddings later (Stage 1 "embed" step).
-- On Neon/most Postgres this just works; it's a one-time enable per database.
CREATE EXTENSION IF NOT EXISTS vector;

-- Raw documents: one row per source file, stored exactly as we read it.
-- Stage 1 only fills THIS table. No chunking, no embeddings yet — on purpose.
CREATE TABLE IF NOT EXISTS documents (
    doc_id     TEXT PRIMARY KEY,                       -- stable id; we'll use the file's relative path
    source     TEXT        NOT NULL,                   -- where it came from (path or URL) — for citations later
    raw_text   TEXT        NOT NULL,                   -- the full, unparsed text of the file
    loaded_at  TIMESTAMPTZ NOT NULL DEFAULT now()      -- when we ingested it
);

-- Chunks: the retrievable pieces we cut each document into.
-- This is what we embed and search. One document -> many chunks.
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id    TEXT PRIMARY KEY,                         -- e.g. "actions__artifacts.md::3"
    doc_id      TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,  -- delete a doc -> its chunks go too
    chunk_index INT  NOT NULL,                            -- order within the document (0,1,2,...)
    text        TEXT NOT NULL,                            -- the chunk's text
    n_chars     INT  NOT NULL                             -- size, so we can inspect our chunking
);

-- Embeddings live on each chunk. bge-small-en-v1.5 outputs 384-dim vectors,
-- so the column must be exactly vector(384) — the dimension MUST match the model.
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS embedding vector(384);

-- HNSW: pgvector's fast approximate-nearest-neighbour index for similarity search.
-- vector_cosine_ops = rank by cosine similarity (the usual choice for text embeddings).
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops);

-- Keyword search (the "exact words" half of hybrid search).
-- A generated tsvector column holds the full-text form of each chunk, kept in
-- sync automatically; the GIN index makes keyword search fast.
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS tsv tsvector
    GENERATED ALWAYS AS (to_tsvector('english', text)) STORED;
CREATE INDEX IF NOT EXISTS chunks_tsv_gin ON chunks USING gin (tsv);

-- Multi-tenant isolation: who owns each chunk. Existing chunks default to 'public'.
-- The query pre-filters on this, so one tenant never even sees another's chunks.
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS tenant_id TEXT NOT NULL DEFAULT 'public';
CREATE INDEX IF NOT EXISTS chunks_tenant_idx ON chunks (tenant_id);

-- Semantic cache: past questions + their answers, keyed by the question's MEANING.
-- On a new question we embed it and look for a very-similar past question here;
-- a close hit returns the stored answer instantly (no retrieval, no LLM calls).
CREATE TABLE IF NOT EXISTS qa_cache (
    id          BIGSERIAL PRIMARY KEY,
    question    TEXT NOT NULL,
    q_embedding vector(384) NOT NULL,          -- the question's meaning-vector (same embedder)
    answer      TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS qa_cache_hnsw ON qa_cache USING hnsw (q_embedding vector_cosine_ops);

-- User feedback (thumbs up/down) on answers. Saved now; used later to improve ranking.
CREATE TABLE IF NOT EXISTS feedback (
    id         BIGSERIAL PRIMARY KEY,
    question   TEXT NOT NULL,
    chunk_id   TEXT,
    thumb      TEXT NOT NULL,                 -- 'up' or 'down'
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Change detection: a fingerprint of each document's content, so unchanged docs are skipped.
ALTER TABLE documents ADD COLUMN IF NOT EXISTS content_hash TEXT;

-- Dead-letter queue: documents that FAILED to process, set aside so one bad file
-- doesn't crash the whole ingest. Inspect / retry later.
CREATE TABLE IF NOT EXISTS dead_letter (
    id         BIGSERIAL PRIMARY KEY,
    doc_id     TEXT,
    error      TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
