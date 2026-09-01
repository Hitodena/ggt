from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from loguru import logger
from sqlalchemy import and_, delete, func, or_, select, type_coerce
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from sqlalchemy.sql import ColumnElement

from app.core.tags import (
    extract_filter_audience,
    extract_filter_list,
    normalize_knowledge_tags,
)
from app.models.kb import KBChunk, KBDocument, KBImportJob, KBUsageEvent

_TAG_LIST_KEYS = ("audience", "clinical", "labels")


def new_id() -> str:
    return str(uuid4())


class KnowledgeDAO:
    @staticmethod
    def _merge_chunk_tags(
        base_tags: dict[str, Any] | None,
        chunk_tags: list[str] | dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Merge document-level tags with per-chunk system metadata."""
        normalized_chunk = normalize_knowledge_tags(chunk_tags)
        if base_tags is None:
            return normalized_chunk
        if normalized_chunk is None:
            return base_tags

        merged: dict[str, Any] = dict(base_tags)
        base_system = (
            dict(base_tags["system"])
            if isinstance(base_tags.get("system"), dict)
            else {}
        )
        chunk_system = (
            dict(normalized_chunk["system"])
            if isinstance(normalized_chunk.get("system"), dict)
            else {}
        )
        if base_system or chunk_system:
            merged["system"] = {**base_system, **chunk_system}

        for key in _TAG_LIST_KEYS:
            if key in normalized_chunk:
                merged[key] = normalized_chunk[key]
            elif key in base_tags:
                merged[key] = base_tags[key]

        for key, value in normalized_chunk.items():
            if key in {"system", *_TAG_LIST_KEYS}:
                continue
            merged[key] = value
        return merged or None

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
    async def get_by_source_origin(
        session: AsyncSession,
        *,
        specialist_id: str,
        source_origin: str,
    ) -> KBDocument | None:
        """Find a file-upload document by specialist + original filename."""
        stmt = (
            select(KBDocument)
            .where(
                KBDocument.specialist_id == specialist_id,
                KBDocument.source_origin == source_origin,
            )
            .options(selectinload(KBDocument.chunks))
            .order_by(KBDocument.updated_at.desc())
            .limit(1)
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
        logger.debug(
            "DAO list | specialist_id={} returned={} total={}",
            specialist_id,
            len(rows),
            total,
        )
        return rows, total

    @staticmethod
    async def create_document_with_chunks(
        session: AsyncSession,
        *,
        specialist_id: str,
        title: str,
        chunks: list[tuple[str, list[float]]]
        | list[tuple[str, list[float], dict[str, Any] | list[Any] | None]],
        source_type: str,
        source_origin: str,
        origin_message_id: str | None = None,
        tags: list[str] | dict[str, Any] | None = None,
    ) -> KBDocument:
        if not chunks:
            raise ValueError("At least one chunk is required")

        document_id = new_id()
        document = KBDocument(
            id=document_id,
            specialist_id=specialist_id,
            title=title,
            source_type=source_type,
            source_origin=source_origin,
            origin_message_id=origin_message_id,
        )
        session.add(document)

        base_tags = normalize_knowledge_tags(tags)
        chunk_ids: list[str] = []
        for item in chunks:
            if len(item) == 3:
                content, embedding, chunk_tags = item  # type: ignore[misc]
            else:
                content, embedding = item  # type: ignore[misc]
                chunk_tags = base_tags
            merged_tags = KnowledgeDAO._merge_chunk_tags(base_tags, chunk_tags)
            chunk_id = new_id()
            chunk_ids.append(chunk_id)
            session.add(
                KBChunk(
                    id=chunk_id,
                    specialist_id=specialist_id,
                    document_id=document_id,
                    content=content,
                    tags=merged_tags,
                    embedding=embedding,
                )
            )

        await session.flush()
        session.add(
            KBUsageEvent(
                id=new_id(),
                specialist_id=specialist_id,
                event_type="knowledge_created",
                payload={
                    "document_id": document_id,
                    "chunk_ids": chunk_ids,
                    "source_type": source_type,
                    "origin_message_id": origin_message_id,
                },
            )
        )
        await session.commit()
        logger.info(
            "DAO created document | id={} specialist_id={} chunks={} "
            "source_type={}",
            document_id,
            specialist_id,
            len(chunk_ids),
            source_type,
        )
        return await KnowledgeDAO.get_by_id(session, document_id)  # type: ignore[return-value]

    @staticmethod
    async def replace_document_chunks(
        session: AsyncSession,
        *,
        document_id: str,
        title: str,
        chunks: list[tuple[str, list[float]]]
        | list[tuple[str, list[float], dict[str, Any] | list[Any] | None]],
        tags: list[str] | dict[str, Any] | None = None,
        specialist_id: str | None = None,
    ) -> KBDocument:
        """Replace all chunks of an existing document (same document id)."""
        if not chunks:
            raise ValueError("At least one chunk is required")

        document = await KnowledgeDAO.get_by_id(
            session,
            document_id,
            specialist_id=specialist_id,
        )
        if document is None:
            raise ValueError(f"Document not found: {document_id}")

        await session.execute(
            delete(KBChunk).where(KBChunk.document_id == document_id)
        )
        # Expire cached relationship so reload picks up new chunks.
        session.expire(document, ["chunks"])

        document.title = title
        document.updated_at = datetime.now(timezone.utc)

        base_tags = normalize_knowledge_tags(tags)
        chunk_ids: list[str] = []
        for item in chunks:
            if len(item) == 3:
                content, embedding, chunk_tags = item  # type: ignore[misc]
            else:
                content, embedding = item  # type: ignore[misc]
                chunk_tags = base_tags
            merged_tags = KnowledgeDAO._merge_chunk_tags(base_tags, chunk_tags)
            chunk_id = new_id()
            chunk_ids.append(chunk_id)
            session.add(
                KBChunk(
                    id=chunk_id,
                    specialist_id=document.specialist_id,
                    document_id=document_id,
                    content=content,
                    tags=merged_tags,
                    embedding=embedding,
                )
            )

        await session.flush()
        session.add(
            KBUsageEvent(
                id=new_id(),
                specialist_id=document.specialist_id,
                event_type="knowledge_updated",
                payload={
                    "document_id": document_id,
                    "chunk_ids": chunk_ids,
                    "source_type": document.source_type,
                    "source_origin": document.source_origin,
                },
            )
        )
        await session.commit()
        logger.info(
            "DAO replaced document chunks | id={} specialist_id={} chunks={}",
            document_id,
            document.specialist_id,
            len(chunk_ids),
        )
        return await KnowledgeDAO.get_by_id(session, document_id)  # type: ignore[return-value]

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
        return await KnowledgeDAO.create_document_with_chunks(
            session,
            specialist_id=specialist_id,
            title=title,
            chunks=[(content, embedding)],
            source_type=source_type,
            source_origin=source_origin,
            origin_message_id=origin_message_id,
            tags=tags,
        )

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
        logger.info("DAO deleted document | id={}", document_id)
        return True

    @staticmethod
    def _has_nonempty_tag_list(key: str) -> ColumnElement[bool]:
        """True when ``tags[key]`` is a non-empty JSON array."""
        tag_value = KBChunk.tags[key]
        return and_(
            KBChunk.tags.is_not(None),
            KBChunk.tags.has_key(key),
            func.jsonb_typeof(tag_value) == "array",
            func.jsonb_array_length(tag_value) > 0,
        )

    @staticmethod
    def _has_no_tag_list(key: str) -> ColumnElement[bool]:
        """True when ``tags[key]`` is missing, null, or an empty array."""
        tag_value = KBChunk.tags[key]
        return or_(
            KBChunk.tags.is_(None),
            ~KBChunk.tags.has_key(key),
            func.jsonb_typeof(tag_value) == "null",
            and_(
                func.jsonb_typeof(tag_value) == "array",
                func.jsonb_array_length(tag_value) == 0,
            ),
        )

    @staticmethod
    def _tag_visibility_clause(
        filter_tags: dict[str, Any] | None,
    ) -> ColumnElement[bool]:
        """Build SQL visibility rules for flat string-list tags.

        Without audience criteria: hide chunks with a non-empty
        ``tags.audience`` array.
        With audience criteria: keep general chunks and chunks whose
        ``tags.audience`` JSON contains the filter list (``@>``).
        Optional ``clinical`` / ``labels`` filters AND-restrict further.
        """
        audience_filter = extract_filter_audience(filter_tags)
        has_no_audience = KnowledgeDAO._has_no_tag_list("audience")

        if audience_filter is None:
            clause: ColumnElement[bool] = has_no_audience
        else:
            audience_match = KBChunk.tags["audience"].contains(
                type_coerce(audience_filter, JSONB)
            )
            clause = or_(has_no_audience, audience_match)

        for key in ("clinical", "labels"):
            filter_list = extract_filter_list(filter_tags, key)
            if filter_list is None:
                continue
            clause = and_(
                clause,
                and_(
                    KnowledgeDAO._has_nonempty_tag_list(key),
                    KBChunk.tags[key].contains(
                        type_coerce(filter_list, JSONB)
                    ),
                ),
            )
        return clause

    # Back-compat alias used by older call sites / tests.
    @staticmethod
    def _audience_visibility_clause(
        filter_tags: dict[str, Any] | None,
    ) -> ColumnElement[bool]:
        return KnowledgeDAO._tag_visibility_clause(filter_tags)

    @staticmethod
    async def search_similar(
        session: AsyncSession,
        *,
        specialist_id: str,
        query_embedding: list[float],
        limit: int = 5,
        filter_tags: dict[str, Any] | None = None,
    ) -> list[tuple[KBChunk, KBDocument, float]]:
        distance = KBChunk.embedding.cosine_distance(query_embedding)
        stmt = (
            select(KBChunk, KBDocument, distance)
            .join(KBDocument, KBChunk.document_id == KBDocument.id)
            .where(
                KBChunk.specialist_id == specialist_id,
                KBChunk.embedding.is_not(None),
                KnowledgeDAO._tag_visibility_clause(filter_tags),
            )
            .order_by(distance)
            .limit(limit)
        )
        result = await session.execute(stmt)
        rows = [
            (chunk, document, float(dist))
            for chunk, document, dist in result.all()
        ]
        logger.debug(
            "DAO search_similar | specialist_id={} limit={} hits={} "
            "filter_tags={}",
            specialist_id,
            limit,
            len(rows),
            filter_tags,
        )
        return rows

    @staticmethod
    async def create_import_job(
        session: AsyncSession,
        *,
        specialist_id: str,
        filename: str,
        content_type: str | None,
        byte_size: int | None,
    ) -> KBImportJob:
        job = KBImportJob(
            id=new_id(),
            specialist_id=specialist_id,
            filename=filename,
            status="processing",
            progress_pct=0,
            step="queued",
            content_type=content_type,
            byte_size=byte_size,
        )
        session.add(job)
        await session.commit()
        await session.refresh(job)
        logger.info(
            "DAO import job created | job_id={} specialist_id={} file={!r}",
            job.id,
            specialist_id,
            filename,
        )
        return job

    @staticmethod
    async def update_import_job(
        session: AsyncSession,
        job_id: str,
        *,
        status: str | None = None,
        step: str | None = None,
        progress_pct: float | None = None,
        error_message: str | None = None,
    ) -> None:
        job = await session.get(KBImportJob, job_id)
        if job is None:
            logger.warning("DAO import job missing | job_id={}", job_id)
            return
        if status is not None:
            job.status = status
        if step is not None:
            job.step = step
        if progress_pct is not None:
            job.progress_pct = progress_pct
        if error_message is not None:
            job.error_message = error_message
        await session.commit()
        logger.debug(
            "DAO import job update | job_id={} status={} step={} progress={}",
            job_id,
            job.status,
            job.step,
            job.progress_pct,
        )
