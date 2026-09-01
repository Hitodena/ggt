from contextlib import asynccontextmanager
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, Request, Response
from loguru import logger

from app.api import knowledge_router
from app.core.config import get_settings
from app.core.logging import setup_logging
from app.db.session import dispose_engine


@asynccontextmanager
async def lifespan(_app: FastAPI):
    settings = get_settings()
    setup_logging(settings)
    logger.info(
        "App starting | env={} chat_model={} embedding_model={} dims={}",
        settings.app_env,
        settings.chat_model,
        settings.embedding_model,
        settings.embedding_dimensions,
    )
    yield
    logger.info("App shutting down")
    await dispose_engine()


app = FastAPI(
    title="Specialist Knowledge Base",
    description=(
        "Per-specialist vector knowledge base on Neon Postgres + pgvector. "
        "No client binding — shared KB for a specialist.\n\n"
        "## Tags\n\n"
        "Preferred shape uses flat string lists:\n\n"
        "- **`audience`** — segment tags "
        '(e.g. `["sex:female", "age_bucket:26_35"]`). '
        "Chunks with a non-empty audience are **hidden** from "
        "`/knowledge/search` and `/knowledge/answer` unless the request "
        "passes matching `filter_tags.audience`.\n"
        "- **`clinical`** / **`labels`** — free-form string lists "
        "(no whitelist; unknown values are accepted).\n"
        "- **`system`** — technical metadata (filename, chunk_index). "
        "Does **not** hide a chunk from search.\n\n"
        "With `filter_tags.audience`, results include general chunks "
        "(no audience) **and** chunks whose audience list contains the "
        "filter strings (Postgres JSONB `@>`). Optional `clinical` / "
        "`labels` filters AND-restrict further.\n\n"
        "Example create/upload tags:\n"
        '`{"audience":["sex:female","age_bucket:26_35"],'
        '"clinical":["procedure:rf_face"],"labels":["manual:spf_лето"]}`\n\n'
        "Example search filter:\n"
        '`{"filter_tags":{"audience":["sex:female","age_bucket:26_35"]}}`\n\n'
        "Re-uploading the same filename for a specialist replaces the "
        "existing document (same id)."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next) -> Response:
    request_id = request.headers.get("x-request-id") or uuid4().hex[:12]
    started = perf_counter()
    with logger.contextualize(request_id=request_id):
        logger.info(
            "HTTP {} {} | client={}",
            request.method,
            request.url.path,
            request.client.host if request.client else "-",
        )
        try:
            response = await call_next(request)
        except Exception:
            elapsed_ms = (perf_counter() - started) * 1000
            logger.exception(
                "HTTP {} {} failed after {:.1f}ms",
                request.method,
                request.url.path,
                elapsed_ms,
            )
            raise

        elapsed_ms = (perf_counter() - started) * 1000
        logger.info(
            "HTTP {} {} -> {} | {:.1f}ms",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        response.headers["X-Request-ID"] = request_id
        return response


app.include_router(knowledge_router)


@app.get("/health")
async def health() -> dict[str, str]:
    logger.debug("Health check")
    return {"status": "ok"}
