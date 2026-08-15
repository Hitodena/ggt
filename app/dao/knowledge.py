from typing import Any
from uuid import uuid4

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.kb import KBChunk, KBDocument, KBUsageEvent


def new_id() -> str:
    return str(uuid4())


class KnowledgeDAO:
    @staticmethod
    async def get_by_origin_message(
        session: AsyncSession,
        *,
        specialist_id: str,
        origin_message_id: str,
    ) -> KBDocument | None:
        stmt = (
            select(KBDocument)
            .where(
                KBDocument.specialist_id == specialist_id,
                KBDocument.origin_message_id == origin_message_id,
            )
            .options(selectinload(KBDocument.chunks))
        )
        result = await session.execute(stmt)
        return result.scalar_one_or_none()

    @staticmethod
    async def get_by_id(
        session: AsyncSession,
        document_id: str,
        *,
        specialist_id: str | None = None,
    ) -> KBDocument | None:
        stmt = (
            select(KBDocument)
            .where(KBDocument.id == document_id)
            .options(selectinload(KBDocument.chunks))
        )
        if specialist_id is not None:
            stmt = stmt.where(KBDocument.specialist_id == specialist_id)
        result = await session.execute(stmt)
        return result.scalar_one_or_none()

    @staticmethod
    async def list_for_specialist(
        session: AsyncSession,
        *,
        specialist_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[KBDocument], int]:
        count_stmt = select(func.count()).select_from(KBDocument).where(
            KBDocument.specialist_id == specialist_id
        )
        total = int((await session.execute(count_stmt)).scalar_one())

        stmt = (
            select(KBDocument)
            .where(KBDocument.specialist_id == specialist_id)
            .options(selectinload(KBDocument.chunks))
            .order_by(KBDocument.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        rows = list((await session.execute(stmt)).scalars().all())
        return rows, total

    @staticmethod
    async def create_document_with_chunk(
        session: AsyncSession,
        *,
        specialist_id: str,
        title: str,
        content: str,
        embedding: list[float],
        source_type: str,
        source_origin: str,
        origin_message_id: str | None = None,
        tags: list[str] | dict[str, Any] | None = None,
    ) -> KBDocument:
        document_id = new_id()
        chunk_id = new_id()

        document = KBDocument(
            id=document_id,
            specialist_id=specialist_id,
            title=title,
            source_type=source_type,
            source_origin=source_origin,
            origin_message_id=origin_message_id,
        )
        chunk = KBChunk(
            id=chunk_id,
            specialist_id=specialist_id,
            document_id=document_id,
            content=content,
            tags=tags,
            embedding=embedding,
        )
        session.add(document)
        session.add(chunk)
        await session.flush()

        usage = KBUsageEvent(
            id=new_id(),
            specialist_id=specialist_id,
            event_type="knowledge_created",
            payload={
                "document_id": document_id,
                "chunk_id": chunk_id,
                "source_type": source_type,
                "origin_message_id": origin_message_id,
            },
        )
        session.add(usage)
        await session.commit()

        return await KnowledgeDAO.get_by_id(session, document_id)  # type: ignore[return-value]

    @staticmethod
    async def delete_document(
        session: AsyncSession,
        document_id: str,
        *,
        specialist_id: str | None = None,
    ) -> bool:
        document = await KnowledgeDAO.get_by_id(
            session,
            document_id,
            specialist_id=specialist_id,
        )
        if document is None:
            return False

        await session.execute(
            delete(KBChunk).where(KBChunk.document_id == document_id)
        )
        await session.delete(document)
        session.add(
            KBUsageEvent(
                id=new_id(),
                specialist_id=document.specialist_id,
                event_type="knowledge_deleted",
                payload={"document_id": document_id},
            )
        )
        await session.commit()
        return True

    @staticmethod
    async def search_similar(
        session: AsyncSession,
        *,
        specialist_id: str,
        query_embedding: list[float],
        limit: int = 5,
    ) -> list[tuple[KBChunk, KBDocument, float]]:
        distance = KBChunk.embedding.cosine_distance(query_embedding)
        stmt = (
            select(KBChunk, KBDocument, distance)
            .join(KBDocument, KBChunk.document_id == KBDocument.id)
            .where(
                KBChunk.specialist_id == specialist_id,
                KBChunk.embedding.is_not(None),
            )
            .order_by(distance)
            .limit(limit)
        )
        result = await session.execute(stmt)
        return [
            (chunk, document, float(dist))
            for chunk, document, dist in result.all()
        ]
