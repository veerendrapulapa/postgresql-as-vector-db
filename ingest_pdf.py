#!/usr/bin/env python3
"""
ingest_pdf.py (safe, verbose) - fixed version:
- Deletes existing doc rows (prints counts)
- Uses plain INSERT for docs (no ON CONFLICT) because docs table lacks a (doc_id,chunk_no) unique constraint
- Uses upsert (ON CONFLICT) for doc_embeddings (which has PK (doc_id,chunk_no))
- Shows progress logs
Usage:
  python ingest_pdf.py <pdf_path> [doc_id]
"""

import os, sys, time
from typing import List
from dotenv import load_dotenv
import psycopg
from openai import OpenAI
from pypdf import PdfReader

# ---- config / env ----
load_dotenv()
DSN = os.getenv("PG_DSN", "postgresql://localhost:5432/ragdb")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
BATCH_SIZE = int(os.getenv("INGEST_BATCH_SIZE", "32"))
CHUNK_CHARS = int(os.getenv("CHUNK_CHARS", "1500"))

if not OPENAI_API_KEY or OPENAI_API_KEY == "replace_me":
    print("ERROR: OPENAI_API_KEY not set. Edit .env or set env var.")
    sys.exit(1)

client = OpenAI(api_key=OPENAI_API_KEY)

# ---- helpers ----
def extract_text_from_pdf(path: str) -> str:
    reader = PdfReader(path)
    pages = []
    for p in reader.pages:
        try:
            pages.append(p.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n\n".join(pages)

def chunk_text(text: str, approx_chars: int = CHUNK_CHARS) -> List[str]:
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    cur = []
    curlen = 0
    for p in paras:
        l = len(p)
        if curlen + l + 2 <= approx_chars:
            cur.append(p); curlen += l + 2
        else:
            if cur:
                chunks.append("\n\n".join(cur).strip())
            if l > approx_chars:
                start = 0
                while start < l:
                    part = p[start:start+approx_chars].strip()
                    if part:
                        chunks.append(part)
                    start += approx_chars
                cur = []
                curlen = 0
            else:
                cur = [p]; curlen = l + 2
    if cur:
        chunks.append("\n\n".join(cur).strip())
    return chunks

def batch_iter(it, n):
    it = iter(it)
    while True:
        chunk = []
        try:
            for _ in range(n):
                chunk.append(next(it))
        except StopIteration:
            if chunk:
                yield chunk
            break
        yield chunk

def embed_texts(texts: List[str]):
    embs = []
    for batch in batch_iter(texts, BATCH_SIZE):
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        for d in resp.data:
            embs.append(d.embedding)
        time.sleep(0.01)
    return embs

# ---- ingest (safe) ----
def ingest_pdf(pdf_path: str, doc_id: str):
    print(f"[+] Ingest start: {pdf_path} -> doc_id='{doc_id}'")
    txt = extract_text_from_pdf(pdf_path)
    if not txt.strip():
        print("ERROR: PDF extraction produced no text.")
        return
    chunks = chunk_text(txt)
    print(f"[+] Chunked into {len(chunks)} chunks (approx {CHUNK_CHARS} chars each)")

    print(f"[+] Generating embeddings with model {EMBED_MODEL} (batch_size={BATCH_SIZE})")
    embs = embed_texts(chunks)
    if len(embs) != len(chunks):
        print(f"WARNING: embeddings ({len(embs)}) != chunks ({len(chunks)})")

    # Persist with safety checks and upsert for embeddings
    with psycopg.connect(DSN) as conn:
        with conn.cursor() as cur:
            # show counts before
            cur.execute("SELECT count(*) FROM docs WHERE doc_id = %s", (doc_id,))
            docs_before = cur.fetchone()[0]
            cur.execute("SELECT count(*) FROM doc_embeddings WHERE doc_id = %s", (doc_id,))
            emb_before = cur.fetchone()[0]
            print(f"[db] before: docs={docs_before}, embeddings={emb_before}")

            # delete old rows first
            print("[db] deleting old rows for doc_id (if any)...")
            cur.execute("DELETE FROM doc_embeddings WHERE doc_id = %s", (doc_id,))
            cur.execute("DELETE FROM docs WHERE doc_id = %s", (doc_id,))

            # verify deletion
            cur.execute("SELECT count(*) FROM docs WHERE doc_id = %s", (doc_id,))
            docs_mid = cur.fetchone()[0]
            cur.execute("SELECT count(*) FROM doc_embeddings WHERE doc_id = %s", (doc_id,))
            emb_mid = cur.fetchone()[0]
            print(f"[db] after delete: docs={docs_mid}, embeddings={emb_mid}")

            # insert docs rows (plain INSERT; docs table lacks unique constraint on (doc_id, chunk_no))
            print("[db] inserting docs rows (plain INSERT)...")
            for i, c in enumerate(chunks):
                cur.execute("""
                    INSERT INTO docs(doc_id, chunk_no, content)
                    VALUES (%s, %s, %s)
                """, (doc_id, i, c))

            # insert embeddings (upsert on conflict since doc_embeddings has PK (doc_id, chunk_no))
            print("[db] inserting embeddings rows (upsert)...")
            for i, e in enumerate(embs):
                cur.execute("""
                    INSERT INTO doc_embeddings(doc_id, chunk_no, embedding)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (doc_id, chunk_no) DO UPDATE SET embedding = EXCLUDED.embedding
                """, (doc_id, i, e))

            # final counts
            cur.execute("SELECT count(*) FROM docs WHERE doc_id = %s", (doc_id,))
            docs_after = cur.fetchone()[0]
            cur.execute("SELECT count(*) FROM doc_embeddings WHERE doc_id = %s", (doc_id,))
            emb_after = cur.fetchone()[0]
            print(f"[db] after insert: docs={docs_after}, embeddings={emb_after}")

        conn.commit()
    print("[+] Ingest complete.")

# ---- CLI ----
def main():
    if len(sys.argv) < 2:
        print("usage: python ingest_pdf.py <pdf_path> [doc_id]")
        sys.exit(1)
    pdf_path = sys.argv[1]
    if not os.path.exists(pdf_path):
        print("ERROR: file not found:", pdf_path)
        sys.exit(1)
    doc_id = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(os.path.basename(pdf_path))[0]
    ingest_pdf(pdf_path, doc_id)

if __name__ == "__main__":
    main()
