# PostgreSQL as a Vector DB

[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-17.x-336791?logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![pgvector](https://img.shields.io/badge/pgvector-0.x-blue)](https://github.com/pgvector/pgvector)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)

This repo tracks my **21‑Day Journey** turning PostgreSQL into a **Vector Database** for AI/RAG workloads.  
Each day adds features, experiments, and benchmarks.

---

## 🧱 What’s inside

- `schema.sql` — tables `docs` and `doc_embeddings` (`vector(1536)` for OpenAI small embeddings)
- `rag_min.py` — minimal ingestion → retrieval → **answers with citations**
- `ingest_pdf.py` — PDF → chunks → embeddings → Postgres (replace‑on‑reingest)
- `app.py` — FastAPI endpoint `/ask?q=...` returns `{ "answer", "citations" }`
- `bench_day2.py` — Day‑2 benchmark (Exact vs **IVFFlat/HNSW**) with **recall@k** and latency
- `bench_day3.py` — Day‑3 benchmark (recall@k vs latency curves across probes/ef_search)
- `.env.example` — environment template (`PG_DSN`, `OPENAI_API_KEY`, model names)
- `requirements.txt` — Python deps
- `Makefile` — convenience targets (`init`, `schema`, `seed`, `ask`, `api`)

---

## ⚙️ Prerequisites

- macOS/Linux, Python **3.11+**
- PostgreSQL **17.x** with `pgvector` extension
- An **OpenAI API key** (or swap in another embedding provider in code)

---

## 🚀 Quick start

```bash
# 1) Python env
python3 -m venv ~/venv/pg-ai && source ~/venv/pg-ai/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 2) Env vars
cp .env.example .env
# edit .env and set: OPENAI_API_KEY=sk-....

# 3) Database
psql postgresql://localhost:5432/postgres -c "CREATE DATABASE ragdb;" || true
psql postgresql://localhost:5432/ragdb -f schema.sql

# 4) Sanity seed & test retrieval
python rag_min.py

# 5) (Optional) Run API
uvicorn app:app --host 0.0.0.0 --port 8000
# → http://127.0.0.1:8000/ask?q=What%20does%20pgvector%20do%3F
```

---

## 📅 Day 1 — Basics (pgvector, memory model, exact ops)

- Installed PostgreSQL 17.5 + enabled `pgvector`
- Designed schema:
  - `docs(doc_id, chunk_no, content)` (PK)
  - `doc_embeddings(doc_id, chunk_no, embedding vector(1536))` (PK, FK to `docs`)
- Verified **exact similarity** using distance operator and `ORDER BY ... LIMIT k`
- Built **minimal RAG** in `rag_min.py`:
  - ingest text → embed → store → retrieve → answer with **citations**

**Takeaways**
- `vector(N)` stores fixed‑length float arrays (4 bytes each) → `1536 × 4 ≈ 6 KB/row`
- Exact search is fine for small data, but at scale you need ANN

---

## 📅 Day 2 — ANN Indexes (IVFFlat & HNSW), Indexing Strategies

### Create IVFFlat (cosine)
```sql
CREATE INDEX IF NOT EXISTS doc_emb_ivfcos
ON doc_embeddings USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 256);

ANALYZE doc_embeddings;
```

**Query‑time breadth (recall vs latency):**
```sql
SET ivfflat.probes = 8;   -- try 8 / 16 / 24
```

### (Optional) Create HNSW (cosine)
```sql
CREATE INDEX IF NOT EXISTS doc_emb_hnswcos
ON doc_embeddings USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 200);

SET hnsw.ef_search = 64;  -- try 32 / 64 / 128
```

### Use the **correct distance operator**
Your indexes use **cosine**, so query with **`<=>`**:

```sql
-- For testing, nudge planner off seq scan
SET enable_seqscan = off;
SET ivfflat.probes = 10;

EXPLAIN ANALYZE
SELECT doc_id, chunk_no
FROM doc_embeddings
ORDER BY embedding <=> (
  ARRAY(SELECT (random()*2 - 1)::float4 FROM generate_series(1,1536))
)::vector
LIMIT 8;
```

**Operator ↔ Opclass cheat‑sheet**

| Distance      | Opclass              | Operator |
|---------------|----------------------|----------|
| L2 / Euclidean| `vector_l2_ops`      | `<->`    |
| Cosine        | `vector_cosine_ops`  | `<=>`    |
| Inner product | `vector_ip_ops`      | `<#>`    |

### Run the Day‑2 benchmark
```bash
python bench_day2.py
```

**Example output**
```json
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
```

**Interpretation**
- ANN was ~2× faster than exact on this small dataset; on larger corpora the gap is much bigger
- Tune **`probes`** (IVFFlat) or **`ef_search`** (HNSW) to reach your target **recall@k**
- If recall plateaus, rebuild with higher **`lists`** (IVFFlat) or tune **`m`/`ef_construction`** (HNSW)

---

## 📅 Day 3 — Formal Benchmarks: Recall@k vs Latency

- Extended Day‑2 work with a new script `bench_day3.py`
- Benchmarks **IVFFlat (probes)** and **HNSW (ef_search)** against exact search
- Measures **latency (ms)** and **recall@k** for different parameter values
- Run with:
  ```bash
  python bench_day3.py
  ```

**Example results**
```json
[
  {"method": "ivf", "probes": 4, "ms": 5.3, "recall": 0.82},
  {"method": "ivf", "probes": 16, "ms": 9.1, "recall": 0.95},
  {"method": "ivf", "probes": 32, "ms": 15.2, "recall": 0.99},
  {"method": "hnsw", "ef": 32, "ms": 6.4, "recall": 0.88},
  {"method": "hnsw", "ef": 64, "ms": 10.8, "recall": 0.97},
  {"method": "hnsw", "ef": 128, "ms": 18.1, "recall": 0.995}
]
```

**Interpretation**
- Exact = 100% recall but slowest at scale
- IVFFlat = recall improves with `probes`, latency increases
- HNSW = recall improves with `ef_search`, usually higher recall at similar latency
- You can now plot **recall vs latency curves** to decide optimal parameters for your workload

---

## 🧪 Ingest a PDF (optional)

```bash
python ingest_pdf.py "path/to/your.pdf" mydoc
# or omit doc_id to derive from filename
```

---

## 🩺 Troubleshooting

- **Seq Scan instead of index**  
  - You likely used the wrong operator. Cosine index → use `<=>` (not `<->`).
  - While testing: `SET enable_seqscan = off;`
  - Run `ANALYZE doc_embeddings;` after big ingests.

- **Dimension mismatch**  
  - Column is `vector(1536)` → queries must pass **1536** numbers.
  - Use the SQL random vector generator shown above, or a real embedding from `rag_min.py`.

- **OpenAI errors (429/insufficient_quota)**  
  - Set a valid `OPENAI_API_KEY` and ensure billing/quota.
  - Or swap to a local embedding provider.

---

## 🗺️ Roadmap (next)

- **Day 4:** RAG architecture deep‑dive (chunking strategies, hybrid search)  
- **Day 5:** Grounding & hallucination control; better citations  

---

## 📜 License

MIT
