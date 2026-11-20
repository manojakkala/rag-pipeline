---
title: RAG Assistant
emoji: 🔎
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# 🔎 RAG Assistant — Documentation & Live-API Assistant

A **production-grade Retrieval-Augmented Generation (RAG)** system, built stage by stage.
It answers questions over **GitHub's documentation**, fetches **live GitHub data** when needed, and is also
exposed as an **MCP server** so it can plug straight into Claude. It ships with a chat UI that shows a **live
trace** of every step (routing, retrieval, timings, grounding).

> Built as a deep, hands-on study of how real RAG systems work — not a toy demo.

## ✨ What makes it production-grade

- **Hybrid retrieval** — semantic (vector) **+** keyword (BM25) search, fused with **Reciprocal Rank Fusion**.
- **Cross-encoder reranking** — re-reads the top candidates (question + chunk together) for sharper ordering.
- **Query rewriting + routing** — a cheap model cleans the question and routes it: **RAG / live-API / chat / decline**.
- **Honest answers** — grounded generation with **citations**, a **groundedness check** (a 2nd model verifies
  every claim), and a real **"I don't know"** when the docs don't cover it.
- **Semantic cache + model tiering** — repeat questions return fast; cheap model for easy steps, strong model
  for the final answer.
- **Security** — ACL pre-filter / tenant isolation (cross-tenant test passes) and a prompt-injection guard.
- **Observability & ops** — per-step **tracing**, a metrics **dashboard**, threshold **alerts**, a **CI gate**
  (GitHub Actions blocks merges if recall drops), and a **blue-green reindex** demo.
- **Real ingestion** — content-hash **change detection**, **deletes propagate**, a **dead-letter queue**, idempotency.

## 📊 Measured quality
- **recall@10: 90%** · **MRR: 0.813** on a hand-built golden set (21 answerable + 2 "no-answer" cases).

## 🏗️ Architecture (two pipelines, one store)
```
INGEST (offline)   docs → load → chunk (structure-aware) → embed (bge-small, 384-d) → Postgres
STORE              Neon Postgres + pgvector: chunks(vector(384) + tsvector), HNSW + GIN indexes, qa_cache
QUERY (online)     question → route → rewrite → hybrid(vector+BM25+RRF) → rerank → Groq LLM → grounded answer
```

## 🧰 Tech stack
**Python · Flask · Postgres (Neon) + pgvector · fastembed (bge-small-en-v1.5) · ms-marco cross-encoder ·
Groq (gpt-oss-120b / 20b) · MCP · Docker · GitHub Actions**

## 🚀 Run it locally
1. `python -m venv .venv && .venv\Scripts\activate && pip install -r requirements.txt`
2. Copy `.env.example` → `.env.local` and fill in `DATABASE_URL` (Neon) and `GROQ_API_KEY` (free at console.groq.com).
3. Apply the schema: `python storage/apply_schema.py`
4. Ingest: `python ingest/load.py` → `python ingest/chunk.py` → `python ingest/embed.py`
5. Start the web app: `python app/server.py` → open http://localhost:5000

## ☁️ Deploy
See **[DEPLOY.md](DEPLOY.md)** for step-by-step Hugging Face Spaces deployment (Docker).

## 📚 More
- **[PROJECT.md](PROJECT.md)** — the full build log: plan, decisions, metrics, coverage checklist.
- **[INTERVIEW_PREP.md](INTERVIEW_PREP.md)** — a deep concept-by-concept study sheet for the whole pipeline.
