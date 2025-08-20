
import os, json
from dotenv import load_dotenv
import psycopg
from openai import OpenAI

# load env
load_dotenv()
dsn = os.getenv("PG_DSN", "postgresql://localhost:5432/ragdb")
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
EMBED = os.getenv("EMBED_MODEL","text-embedding-3-small")
GEN   = os.getenv("GEN_MODEL","gpt-4o-mini")

def embed(texts):
    resp = client.embeddings.create(model=EMBED, input=texts)
    return [d.embedding for d in resp.data]

def ingest(doc_id, chunks, replace=True):
    with psycopg.connect(dsn) as con, con.cursor() as cur:
        cur.execute("BEGIN;")
        if replace:
            cur.execute("DELETE FROM doc_embeddings WHERE doc_id=%s", (doc_id,))
            cur.execute("DELETE FROM docs WHERE doc_id=%s", (doc_id,))
        for i,c in enumerate(chunks):
            cur.execute("INSERT INTO docs(doc_id,chunk_no,content) VALUES(%s,%s,%s)",
                        (doc_id,i,c))
        embs = embed(chunks)
        for i,e in enumerate(embs):
            cur.execute("INSERT INTO doc_embeddings(doc_id,chunk_no,embedding) VALUES(%s,%s,%s)",
                        (doc_id,i,e))
        cur.execute("COMMIT;")

def retrieve(q, k=5):
    """Return [(doc_id, chunk_no, content)] for top-k chunks."""
    v = embed([q])[0]
    with psycopg.connect(dsn) as con, con.cursor() as cur:
        cur.execute("""
            SELECT d.doc_id, d.chunk_no, d.content
            FROM doc_embeddings e
            JOIN docs d USING(doc_id,chunk_no)
            ORDER BY e.embedding <-> %s::vector
            LIMIT %s
        """, (v, k))
        return cur.fetchall()

def answer(q, k=5):
    """Return JSON-able dict with answer + citations."""
    ctx_rows = retrieve(q, k=k)  # [(doc_id, chunk_no, content), ...]
    # Build tagged context and citation list
    context_blocks, citations = [], []
    for doc_id, chunk_no, content in ctx_rows:
        tag = f"{doc_id}:{chunk_no}"
        citations.append(tag)
        context_blocks.append(f"[{tag}]\n{content}")

    context = "\n\n".join(context_blocks)
    prompt = (
        "Answer using ONLY the provided context. If the answer is not in the context, "
        "reply exactly: 'Not in the context.'\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {q}\n\n"
        "Return a compact JSON object with keys:\n"
        '  "answer": string,\n'
        '  "citations": array of strings like "doc_id:chunk_no"\n'
        "Do not include any text before or after the JSON."
    )

    chat = client.chat.completions.create(
        model=GEN,
        messages=[{"role":"user","content":prompt}],
        temperature=0.0,
    )
    text = chat.choices[0].message.content.strip()
    try:
        obj = json.loads(text)
        if "citations" not in obj or not obj["citations"]:
            obj["citations"] = citations[:k]
        return obj
    except Exception:
        return {"answer": text, "citations": citations[:k]}

if __name__=="__main__":
    ingest("intro", [
        "PostgreSQL with pgvector enables vector similarity search.",
        "RAG retrieves relevant chunks and asks an LLM to answer grounded questions."
    ])
    print("Top hits:", retrieve("What does pgvector enable?"))
    print("Answer JSON:", answer("What is RAG in one sentence?"))
