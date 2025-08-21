# bench_day2.py
import os, time, json, psycopg
from dotenv import load_dotenv
from rag_min import embed

load_dotenv()
DSN = os.getenv("PG_DSN", "postgresql://localhost:5432/ragdb")

# config you may tweak
K = 8
QUERY_TEXTS = [
    "What is Kafka?",
    "What is Debezium?",
    "What does ZooKeeper do in Kafka?",
    "Change data capture",
    "RAG architecture",
    "PostgreSQL vector search",
    "Index tuning",
    "Cosine vs L2 distance",
]

def topk(conn, qvec, k=K):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT doc_id, chunk_no
            FROM doc_embeddings
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """, (qvec, k))
        return [tuple(r) for r in cur.fetchall()]

def run_mode(conn, mode):
    """Set planner/index knobs, then run top-k for QUERY_TEXTS and measure time."""
    with conn.cursor() as cur:
        if mode == "exact":
            cur.execute("SET enable_seqscan = on;")
        elif mode.startswith("ivf"):
            # ivf10, ivf16, ivf24, etc.
            probes = int(mode.replace("ivf", ""))
            cur.execute("SET enable_seqscan = off;")
            cur.execute(f"SET ivfflat.probes = {probes};")
        elif mode.startswith("hnsw"):
            # hnsw32, hnsw64, hnsw128
            ef = int(mode.replace("hnsw", ""))
            cur.execute("SET enable_seqscan = off;")
            cur.execute(f"SET hnsw.ef_search = {ef};")
        else:
            raise ValueError(f"unknown mode: {mode}")

    lat_ms = []
    results = []
    for q in QUERY_TEXTS:
        qvec = embed([q])[0]
        t0 = time.time()
        rows = topk(conn, qvec, k=K)
        lat_ms.append((time.time() - t0) * 1000.0)
        results.append(rows)

    return results, sum(lat_ms) / len(lat_ms)

def recall_at_k(gold, approx):
    """Average overlap between exact and approx top-k sets per query."""
    assert len(gold) == len(approx)
    acc = 0.0
    for g, a in zip(gold, approx):
        gs, as_ = set(g), set(a)
        acc += len(gs & as_) / max(1, len(gs))
    return acc / len(gold)

if __name__ == "__main__":
    modes = ["exact", "ivf8", "ivf16", "ivf24"]
    # try HNSW if you created an HNSW index:
    # modes += ["hnsw32", "hnsw64", "hnsw128"]

    with psycopg.connect(DSN) as con:
        # exact baseline
        exact_res, exact_ms = run_mode(con, "exact")

        report = {
            "k": K,
            "exact_ms_avg": round(exact_ms, 2),
        }

        for m in modes[1:]:
            approx_res, approx_ms = run_mode(con, m)
            report[f"{m}_ms_avg"] = round(approx_ms, 2)
            report[f"{m}_recall@k"] = round(recall_at_k(exact_res, approx_res), 3)

    print(json.dumps(report, indent=2))
