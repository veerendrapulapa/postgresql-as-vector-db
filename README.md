# PostgreSQL as a Vector DB — Day 1 & Day 2

[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-17.x-336791?logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![pgvector](https://img.shields.io/badge/pgvector-0.x-blue)](https://github.com/pgvector/pgvector)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)

This repo tracks my **21-Day Journey** of turning PostgreSQL into a Vector Database for AI/RAG workloads.  
Each day adds new features, experiments, and benchmarks.

---

## 📅 Day 1 — Basics
- Setup PostgreSQL 17.5 + `pgvector`
- Designed schema (`docs`, `doc_embeddings`)
- Built minimal **RAG pipeline** in `rag_min.py`
- Added citations to answers for grounding

---

## 📅 Day 2 — ANN Indexes (IVFFlat & HNSW)
- Created **IVFFlat index** (`lists=256`) and tuned `ivfflat.probes`
- Created **HNSW index** (`m=16, ef_construction=200`) and tuned `hnsw.ef_search`
- Compared **Exact vs ANN search** with `EXPLAIN ANALYZE`
- Added benchmark script `bench_day2.py` to measure latency + recall@k

### Run the benchmark
```bash
python bench_day2.py

## Example output
{
  "k": 8,
  "exact_ms_avg": 24.49,
  "ivf8_ms_avg": 11.2,
  "ivf8_recall@k": 1.0,
  "ivf16_ms_avg": 9.3,
  "ivf16_recall@k": 1.0,
  "ivf24_ms_avg": 10.28,
  "ivf24_recall@k": 1.0
}

⚡ ANN was ~2× faster than exact search, with perfect recall at this dataset size. Larger datasets will show bigger differences.

## What’s inside
- `schema.sql` — tables `docs` and `doc_embeddings`.
- `rag_min.py` — minimal ingestion + retrieval + answers with citations.
- `ingest_pdf.py` — PDF ingestion with batching, chunking.
- `app.py` — FastAPI endpoint (`/ask?q=...`) returning `{ "answer", "citations" }`.
- `requirements.txt` — Python dependencies.
- `.env.example` — template for environment variables.
- `Makefile` — `init`, `schema`, `seed`, `ask`, and `api` targets.

## Quick start
```bash
python3 -m venv ~/venv/pg-ai && source ~/venv/pg-ai/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Setup env
cp .env.example .env
# edit .env and set OPENAI_API_KEY

# Create DB
psql postgresql://localhost:5432/postgres -c "CREATE DATABASE ragdb;"
psql postgresql://localhost:5432/ragdb -f schema.sql
