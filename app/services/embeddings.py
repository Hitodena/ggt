from loguru import logger
from openai import AsyncOpenAI

from app.core.config import Settings, get_settings


class EmbeddingError(RuntimeError):
    pass


class EmbeddingService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        if not self.settings.openai_api_key:
            raise EmbeddingError(
                "OPENAI_API_KEY is not set; cannot generate embeddings"
            )
        self._client = AsyncOpenAI(
            api_key=self.settings.openai_api_key,
            base_url=self.settings.openai_base_url,
        )
        logger.debug(
            "EmbeddingService ready | model={} dims={} base_url={}",
            self.settings.embedding_model,
            self.settings.embedding_dimensions,
            self.settings.openai_base_url,
        )

    async def embed(self, text: str) -> list[float]:
        cleaned = text.strip()
        if not cleaned:
            raise EmbeddingError("Cannot embed empty text")

        logger.debug(
            "Embed request | model={} chars={}",
            self.settings.embedding_model,
            len(cleaned),
        )
        try:
            response = await self._client.embeddings.create(
                model=self.settings.embedding_model,
                input=cleaned,
                dimensions=self.settings.embedding_dimensions,
            )
        except Exception:
            logger.exception(
                "Embed failed | model={} chars={}",
                self.settings.embedding_model,
                len(cleaned),
            )
            raise

        vector = response.data[0].embedding
        if len(vector) != self.settings.embedding_dimensions:
            raise EmbeddingError(
                f"Expected {self.settings.embedding_dimensions} dims, "
                f"got {len(vector)}"
            )
        logger.debug("Embed ok | dims={}", len(vector))
        return list(vector)
