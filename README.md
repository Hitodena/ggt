# Specialist Knowledge Base (FastAPI + Neon pgvector)

Per-specialist vector knowledge base. No client binding — knowledge is shared
for a specialist and can be added from chat via "add to knowledge base".

## Stack

- FastAPI + SQLAlchemy async + asyncpg
- Neon Lakebase Postgres with `pgvector` (`vector(1536)`, HNSW cosine)
- Existing tables: `kb_documents`, `kb_chunks` (+ captures/imports/usage)
- OpenAI-compatible embeddings (`text-embedding-3-small`)

## Setup

```bash
cd knowledge-base-fastapi
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -e ".[dev]"
# or with uv:
# uv sync

cp .env.example .env
# fill DATABASE_URL, DATABASE_URL_DIRECT, OPENAI_API_KEY
```

`DATABASE_URL` — pooled Neon URL (`-pooler` hostname) for the API.  
`DATABASE_URL_DIRECT` — non-pooler URL for Alembic (optional but recommended).

## Run

```bash
uvicorn app.main:app --reload --port 8000
```

Docs: http://127.0.0.1:8000/docs

## Inspect DB

```bash
python scripts/inspect_db.py
```

## Alembic

Schema already exists on Neon. Baseline revision is a no-op:

```bash
alembic upgrade head
```

## API

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/knowledge` | Manual add |
| `POST` | `/knowledge/from-message` | Chat button (idempotent by `message_id`) |
| `POST` | `/knowledge/search` | Semantic search |
| `GET` | `/knowledge?specialist_id=` | List |
| `GET` | `/knowledge/{id}` | Get one |
| `DELETE` | `/knowledge/{id}` | Delete document + chunks |

### Add from chat

```bash
curl -X POST http://127.0.0.1:8000/knowledge/from-message \
  -H "Content-Type: application/json" \
  -d "{
    \"specialist_id\": \"spec-1\",
    \"message_id\": \"msg-42\",
    \"title\": \"Протокол очистки\",
    \"content\": \"Для чувствительной кожи использовать мягкий энзимный пилинг...\"
  }"
```

### Search

```bash
curl -X POST http://127.0.0.1:8000/knowledge/search \
  -H "Content-Type: application/json" \
  -d "{
    \"specialist_id\": \"spec-1\",
    \"query\": \"пилинг для чувствительной кожи\",
    \"limit\": 5
  }"
```

## Tests

```bash
pytest
```
