from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.dao.knowledge import KnowledgeDAO
from app.models.kb import KBDocument
from app.schemas import (
    KnowledgeCreate,
    KnowledgeFromMessage,
    KnowledgeSearchHit,
    KnowledgeSearchRequest,
    KnowledgeSearchResponse,
)
from app.services.embeddings import EmbeddingService


class KnowledgeService:
    def __init__(
        self,
        session: AsyncSession,
        embeddings: EmbeddingService | None = None,
    ) -> None:
        self.session = session
        self.embeddings = embeddings or EmbeddingService()

    async def create(self, payload: KnowledgeCreate) -> KBDocument:
        logger.info(
            "Create knowledge | specialist_id={} title={!r}",
            payload.specialist_id,
            payload.title,
        )
        vector = await self.embeddings.embed(payload.content)
        document = await KnowledgeDAO.create_document_with_chunk(
            self.session,
            specialist_id=payload.specialist_id,
            title=payload.title,
            content=payload.content,
            embedding=vector,
            source_type=payload.source_type,
            source_origin=payload.source_origin,
            tags=payload.tags,
        )
        logger.info("Create knowledge done | document_id={}", document.id)
        return document

    async def create_from_message(
        self,
        payload: KnowledgeFromMessage,
    ) -> tuple[KBDocument, bool]:
        """Return (document, created). created=False when already present."""
        existing = await KnowledgeDAO.get_by_origin_message(
            self.session,
            specialist_id=payload.specialist_id,
            origin_message_id=payload.message_id,
        )
        if existing is not None:
            logger.info(
                "From-message idempotent hit | specialist_id={} "
                "message_id={} document_id={}",
                payload.specialist_id,
                payload.message_id,
                existing.id,
            )
            return existing, False

        logger.info(
            "From-message create | specialist_id={} message_id={}",
            payload.specialist_id,
            payload.message_id,
        )
        vector = await self.embeddings.embed(payload.content)
        document = await KnowledgeDAO.create_document_with_chunk(
            self.session,
            specialist_id=payload.specialist_id,
            title=payload.title,
            content=payload.content,
            embedding=vector,
            source_type=payload.source_type,
            source_origin=payload.source_origin,
            origin_message_id=payload.message_id,
            tags=payload.tags,
        )
        logger.info(
            "From-message created | document_id={}",
            document.id,
        )
        return document, True

    async def search(
        self,
        payload: KnowledgeSearchRequest,
    ) -> KnowledgeSearchResponse:
        logger.info(
            "Search | specialist_id={} query={!r} limit={}",
            payload.specialist_id,
            payload.query,
            payload.limit,
        )
        vector = await self.embeddings.embed(payload.query)
        rows = await KnowledgeDAO.search_similar(
            self.session,
            specialist_id=payload.specialist_id,
            query_embedding=vector,
            limit=payload.limit,
        )
        hits = [
            KnowledgeSearchHit(
                document_id=document.id,
                document_title=document.title,
                chunk_id=chunk.id,
                content=chunk.content,
                distance=distance,
                tags=chunk.tags,
            )
            for chunk, document, distance in rows
        ]
        logger.info(
            "Search done | specialist_id={} hits={}",
            payload.specialist_id,
            len(hits),
        )
        if hits:
            logger.debug(
                "Search top hit | document_id={} distance={:.4f}",
                hits[0].document_id,
                hits[0].distance,
            )
        return KnowledgeSearchResponse(
            query=payload.query,
            specialist_id=payload.specialist_id,
            hits=hits,
        )
