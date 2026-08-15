from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import knowledge_router
from app.core.config import get_settings
from app.db.session import dispose_engine


@asynccontextmanager
async def lifespan(_app: FastAPI):
    get_settings()
    yield
    await dispose_engine()


app = FastAPI(
    title="Specialist Knowledge Base",
    description=(
        "Per-specialist vector knowledge base on Neon Postgres + pgvector. "
        "No client binding — shared KB for a specialist."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(knowledge_router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
