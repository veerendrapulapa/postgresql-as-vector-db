
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS docs (
  doc_id   text NOT NULL,
  chunk_no int  NOT NULL,
  content  text NOT NULL,
  PRIMARY KEY (doc_id, chunk_no)
);

-- Adjust 1536 to your embedding model's dimension
CREATE TABLE IF NOT EXISTS doc_embeddings (
  doc_id   text NOT NULL,
  chunk_no int  NOT NULL,
  embedding vector(1536),
  PRIMARY KEY (doc_id, chunk_no),
  FOREIGN KEY (doc_id, chunk_no) REFERENCES docs(doc_id,chunk_no)
);
