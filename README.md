
# PostgreSQL as a Vector DB — Day 1

This repo is a minimal, working Day 1 setup for running **PostgreSQL + pgvector** with a tiny RAG pipeline.

## What’s inside
- `schema.sql` — tables for `docs` and `doc_embeddings` (1536-dim embeddings).
- `rag_min.py` — minimal ingestion + retrieval + JSON answers with citations.
- `ingest_pdf.py` — PDF ingestion with batching, chunking, and doc_id handling.
- `app.py` — FastAPI endpoint (`/ask?q=...`) returning JSON `{answer, citations}`.
- `requirements.txt` — Python dependencies.
- `.env.example` — template for environment variables.
- `Makefile` — quick tasks for init, seed, ask, and running the API.

## Quick start
```bash
python3 -m venv ~/venv/pg-ai && source ~/venv/pg-ai/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Copy env template and set your own key
cp .env.example .env
# Edit .env to set OPENAI_API_KEY and (optionally) models

# Initialize DB schema
psql postgresql://localhost:5432/postgres -c "CREATE DATABASE ragdb;"
psql postgresql://localhost:5432/ragdb    -f schema.sql

# Seed tiny corpus + test
python rag_min.py

# Run API
uvicorn app:app --host 0.0.0.0 --port 8000
# http://127.0.0.1:8000/ask?q=What%20does%20pgvector%20do%3F
```

## License
MIT
