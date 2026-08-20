# Specialist Knowledge Base (FastAPI + Neon pgvector)

Per-specialist vector knowledge base. No client binding — knowledge is shared
for a specialist and can be added from chat via "add to knowledge base".

## Stack

- FastAPI + SQLAlchemy async + asyncpg
- Neon Lakebase Postgres with `pgvector` (`vector(1536)`, HNSW cosine)
- Existing tables: `kb_documents`, `kb_chunks` (+ captures/imports/usage)
- OpenAI-compatible embeddings + chat (RAG)
- File extraction: PDF / DOC / DOCX / XLS / XLSX (+ Tesseract OCR fallback for scanned PDFs)

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
# fill DATABASE_URL, DATABASE_URL_DIRECT, OPENAI_API_KEY, CHAT_MODEL
```

`DATABASE_URL` — pooled Neon URL (`-pooler` hostname) for the API.  
`DATABASE_URL_DIRECT` — non-pooler URL for Alembic (optional but recommended).

### Windows: OCR and LibreOffice (for local runs)

For scanned PDFs and legacy `.doc` conversion:

```bash
winget install UB-Mannheim.TesseractOCR
winget install TheDocumentFoundation.LibreOffice
```

Then in `.env`:

```env
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
LIBREOFFICE_PATH=C:\Program Files\LibreOffice\program\soffice.exe
OCR_LANGUAGES=rus+eng
```

Docker image already includes Tesseract + LibreOffice.

## Run

```bash
uvicorn app.main:app --reload --port 8000
```

Docs: http://127.0.0.1:8000/docs

## Docker

Postgres остаётся в Neon — контейнер только API (+ OCR/LibreOffice).

```bash
cp .env.example .env
# заполнить DATABASE_URL / OPENAI_API_KEY / CHAT_MODEL

docker compose up --build -d
```

API: http://127.0.0.1:8000/docs  

Без compose:

```bash
docker build -t knowledge-base-fastapi .
docker run --rm -p 8000:8000 --env-file .env knowledge-base-fastapi
```

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
| `POST` | `/knowledge/upload` | Upload pdf/doc/docx/xls/xlsx → extract → embed |
| `POST` | `/knowledge/search` | Semantic search (raw chunks) |
| `POST` | `/knowledge/answer` | RAG: search + chat model answer |
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

### Upload a file

```bash
curl -X POST http://127.0.0.1:8000/knowledge/upload \
  -F "specialist_id=spec-1" \
  -F "title=Протокол после процедур" \
  -F "file=@./protocol.pdf"
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

Segmented notes use `tags.audience` and are hidden from default search.
Pass `filter_tags` to include them (together with general notes):

```bash
curl -X POST http://127.0.0.1:8000/knowledge/search \
  -H "Content-Type: application/json" \
  -d "{
    \"specialist_id\": \"spec-1\",
    \"query\": \"протокол\",
    \"limit\": 5,
    \"filter_tags\": {\"audience\": {\"gender\": \"male\", \"age_min\": 40}}
  }"
```

### RAG answer

```bash
curl -X POST http://127.0.0.1:8000/knowledge/answer \
  -H "Content-Type: application/json" \
  -d "{
    \"specialist_id\": \"spec-1\",
    \"query\": \"что делать после пилинга?\",
    \"limit\": 5
  }"
```

## How vectors + answers work

1. **Embedding model** (`EMBEDDING_MODEL`) turns text into vectors for storage/search.
2. **Search** returns similar chunks (raw text + distance).
3. **Answer / RAG** takes those chunks as context and asks a separate **chat model** (`CHAT_MODEL`) to compose a readable reply.

Embedding models cannot generate answers by themselves.

### File chunking

Upload pipeline: extract → pack paragraphs into chunks (`CHUNK_SIZE` / `CHUNK_OVERLAP`) → embed.

DOCX paragraphs are **packed together** up to `CHUNK_SIZE` (default 1500), so a section heading stays with the following list/body instead of becoming a standalone chunk. Already imported files keep old cuts until re-uploaded.

Upload filenames are normalized to UTF-8 (fixes common multipart mojibake).

## Tests

```bash
pytest
```
