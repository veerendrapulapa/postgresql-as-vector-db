
venv := $(HOME)/venv/pg-ai
PY := $(venv)/bin/python

init:
	python3 -m venv $(venv); \
	$(PY) -m pip install --upgrade pip; \
	$(PY) -m pip install -r requirements.txt

schema:
	psql postgresql://localhost:5432/postgres -c "CREATE DATABASE ragdb;" || true
	psql postgresql://localhost:5432/ragdb -f schema.sql

seed:
	$(PY) rag_min.py

ask:
	$(PY) - <<'PY'
from rag_min import answer
print(answer("What does pgvector enable?"))
PY

api:
	$(venv)/bin/uvicorn app:app --host 0.0.0.0 --port 8000
