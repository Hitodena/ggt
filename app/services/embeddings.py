from time import perf_counter

from loguru import logger
from openai import AsyncOpenAI

from app.core.config import Settings, get_settings
from app.services.openai_client import build_openai_client


class EmbeddingError(RuntimeError):
    pass


class EmbeddingService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        if not self.settings.openai_api_key:
            raise EmbeddingError(
                "OPENAI_API_KEY is not set; cannot generate embeddings"
            )
        self._client = build_openai_client(self.settings)
        logger.debug(
            "EmbeddingService ready | model={} dims={} send_dims={} "
            "base_url={} timeout={}s",
            self.settings.embedding_model,
            self.settings.embedding_dimensions,
            self.settings.embedding_send_dimensions,
            self.settings.openai_base_url,
            self.settings.openai_timeout_sec,
        )

    async def embed(self, text: str) -> list[float]:
        cleaned = text.strip()
        if not cleaned:
            raise EmbeddingError("Cannot embed empty text")

        kwargs: dict = {
            "model": self.settings.embedding_model,
            "input": cleaned,
        }
        if self.settings.embedding_send_dimensions:
            kwargs["dimensions"] = self.settings.embedding_dimensions

        started = perf_counter()
        logger.info(
            "Embed start | model={} chars={} send_dims={}",
            self.settings.embedding_model,
            len(cleaned),
            self.settings.embedding_send_dimensions,
        )
        try:
            response = await self._client.embeddings.create(**kwargs)
        except Exception as exc:
            elapsed_ms = (perf_counter() - started) * 1000
            logger.exception(
                "Embed failed | model={} chars={} elapsed_ms={:.0f} err={}",
                self.settings.embedding_model,
                len(cleaned),
                elapsed_ms,
                exc,
            )
            raise EmbeddingError(
                f"Embedding request failed after {elapsed_ms:.0f}ms: {exc}"
            ) from exc

        vector = response.data[0].embedding
        elapsed_ms = (perf_counter() - started) * 1000
        if len(vector) != self.settings.embedding_dimensions:
            raise EmbeddingError(
                f"Expected {self.settings.embedding_dimensions} dims, "
                f"got {len(vector)}. If using Timeweb, set "
                f"EMBEDDING_DIMENSIONS={len(vector)} or "
                f"EMBEDDING_SEND_DIMENSIONS=false"
            )
        logger.info(
            "Embed ok | dims={} elapsed_ms={:.0f}",
            len(vector),
            elapsed_ms,
        )
        return list(vector)
