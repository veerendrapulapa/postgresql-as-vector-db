
import os, sys
import psycopg
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

load_dotenv()
DSN   = os.getenv("PG_DSN","postgresql://localhost:5432/ragdb")
EMBED = os.getenv("EMBED_MODEL","text-embedding-3-small")
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def chunk_text(t, size=800, overlap=120):
    out = []
    i, step = 0, max(1, size - overlap)
    while i < len(t):
        piece = t[i:i+size].strip()
        if piece: out.append(piece)
        i += step
    return out

def read_pdf_text(path):
    r = PdfReader(path)
    parts = []
    for p in r.pages:
        txt = p.extract_text() or ""
        txt = txt.replace("\x00", "").strip()
        if txt:
            parts.append(txt)
    return "\n".join(parts)

def embed_batch(texts, batch=64):
    embs = []
    for i in range(0, len(texts), batch):
        sub = texts[i:i+batch]
        resp = client.embeddings.create(model=EMBED, input=sub)
        embs.extend([d.embedding for d in resp.data])
    return embs

def ingest_pdf(path, doc_id):
    full = read_pdf_text(path)
    if not full:
        raise SystemExit(f"No extractable text in: {path}")

    chunks = chunk_text(full, size=800, overlap=120)
    if not chunks:
        raise SystemExit("Chunker produced 0 chunks (unexpected).")

    embs = embed_batch(chunks, batch=64)
    if len(embs) != len(chunks):
        raise SystemExit("Embedding count mismatch.")

    with psycopg.connect(DSN) as con, con.cursor() as cur:
        cur.execute("BEGIN;")
        # replace document if re-ingested
        cur.execute("DELETE FROM doc_embeddings WHERE doc_id=%s", (doc_id,))
        cur.execute("DELETE FROM docs WHERE doc_id=%s", (doc_id,))
        for i, c in enumerate(chunks):
            cur.execute("INSERT INTO docs(doc_id,chunk_no,content) VALUES(%s,%s,%s)", (doc_id,i,c))
        for i, e in enumerate(embs):
            cur.execute("INSERT INTO doc_embeddings(doc_id,chunk_no,embedding) VALUES(%s,%s,%s)", (doc_id,i,e))
        cur.execute("COMMIT;")
    print(f"Ingested {len(chunks)} chunks for doc_id='{doc_id}'")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ingest_pdf.py <pdf_path> [doc_id]")
        sys.exit(2)
    pdf_path = sys.argv[1]
    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        sys.exit(2)
    doc_id = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(os.path.basename(pdf_path))[0].lower().replace(" ","_")
    ingest_pdf(pdf_path, doc_id)
