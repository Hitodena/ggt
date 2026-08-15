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
        vector = await self.embeddings.embed(payload.content)
        return await KnowledgeDAO.create_document_with_chunk(
            self.session,
            specialist_id=payload.specialist_id,
            title=payload.title,
            content=payload.content,
            embedding=vector,
            source_type=payload.source_type,
            source_origin=payload.source_origin,
            tags=payload.tags,
        )

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
            return existing, False

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
        return document, True

    async def search(
        self,
        payload: KnowledgeSearchRequest,
    ) -> KnowledgeSearchResponse:
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
        return KnowledgeSearchResponse(
            query=payload.query,
            specialist_id=payload.specialist_id,
            hits=hits,
        )
