
from fastapi import FastAPI, Query
from rag_min import answer

app = FastAPI()

@app.get("/ask")
def ask(q: str = Query(..., min_length=3, description="Your question")):
    return answer(q)
