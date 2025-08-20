# PostgreSQL as a Vector DB — Day 1

[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-17.x-336791?logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![pgvector](https://img.shields.io/badge/pgvector-0.x-blue)](https://github.com/pgvector/pgvector)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)

Minimal, working **Day 1** setup to run **PostgreSQL + pgvector** with a tiny **RAG** pipeline (ingest → embed → store → retrieve → answer).

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
