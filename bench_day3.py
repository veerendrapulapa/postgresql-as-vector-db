import os, time, json, psycopg, numpy as np
from dotenv import load_dotenv

load_dotenv()
dsn = os.getenv("PG_DSN", "postgresql://localhost:5432/ragdb")

def random_vec(dim=1536):
    return np.random.uniform(-1,1,dim).astype("float32").tolist()

def recall(con, q, k, method):
    cur = con.cursor()
    # exact
    cur.execute("SELECT doc_id FROM doc_embeddings ORDER BY embedding <=> %s::vector LIMIT %s", (q,k))
    exact = [r[0] for r in cur.fetchall()]
    # approx
    cur.execute(f"SET {method}")
    cur.execute("SELECT doc_id FROM doc_embeddings ORDER BY embedding <=> %s::vector LIMIT %s", (q,k))
    approx = [r[0] for r in cur.fetchall()]
    return len(set(exact) & set(approx)) / k

def bench(nq=30, k=10):
    results = []
    with psycopg.connect(dsn) as con:
        for probes in [4,8,16,32]:
            lat = []
            rec = []
            for _ in range(nq):
                q = random_vec()
                t0 = time.time()
                r = recall(con, q, k, f"ivfflat.probes = {probes}")
                lat.append((time.time()-t0)*1000)
                rec.append(r)
            results.append({"method":"ivf", "probes":probes, "ms":np.mean(lat), "recall":np.mean(rec)})
        for ef in [32,64,128]:
            lat = []
            rec = []
            for _ in range(nq):
                q = random_vec()
                t0 = time.time()
                r = recall(con, q, k, f"hnsw.ef_search = {ef}")
                lat.append((time.time()-t0)*1000)
                rec.append(r)
            results.append({"method":"hnsw", "ef":ef, "ms":np.mean(lat), "recall":np.mean(rec)})
    print(json.dumps(results, indent=2))

if __name__=="__main__":
    bench()
