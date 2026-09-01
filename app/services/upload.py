from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.filenames import normalize_upload_filename
from app.core.tags import normalize_knowledge_tags
from app.dao.knowledge import KnowledgeDAO
from app.models.kb import KBDocument
from app.services.chunking import chunk_text, chunk_text_structured
from app.services.embeddings import EmbeddingService
from app.services.extraction import ExtractionError, TextExtractor


class UploadError(RuntimeError):
    pass


class UploadService:
    def __init__(
        self,
        session: AsyncSession,
        *,
        embeddings: EmbeddingService | None = None,
        extractor: TextExtractor | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.session = session
        self.settings = settings or get_settings()
        self.embeddings = embeddings or EmbeddingService(self.settings)
        self.extractor = extractor or TextExtractor(self.settings)

    async def upload(
        self,
        *,
        specialist_id: str,
        filename: str,
        data: bytes,
        content_type: str | None = None,
        title: str | None = None,
        tags: dict[str, Any] | list[Any] | None = None,
    ) -> tuple[KBDocument, str]:
        if not specialist_id.strip():
            raise UploadError("specialist_id is required")
        if not filename.strip():
            raise UploadError("filename is required")
        if len(data) > self.settings.max_upload_bytes:
            raise UploadError(
                f"File exceeds max size of {self.settings.max_upload_size_mb} MB"
            )

        safe_name = normalize_upload_filename(filename)
        safe_title = (
            normalize_upload_filename(title, fallback="")
            if title
            else None
        )
        if safe_title == "":
            safe_title = None

        logger.info(
            "Upload start | specialist_id={} file={!r} bytes={} content_type={}",
            specialist_id,
            safe_name,
            len(data),
            content_type,
        )
        job = await KnowledgeDAO.create_import_job(
            self.session,
            specialist_id=specialist_id,
            filename=safe_name,
            content_type=content_type,
            byte_size=len(data),
        )
        logger.debug("Import job created | job_id={}", job.id)

        try:
            await KnowledgeDAO.update_import_job(
                self.session,
                job.id,
                step="extracting",
                progress_pct=10,
            )
            blocks = self.extractor.extract(safe_name, data)
            logger.info(
                "Upload extracted | job_id={} blocks={} chars={}",
                job.id,
                len(blocks),
                sum(
                    len(b if isinstance(b, str) else str(b.get("content", "")))
                    for b in blocks
                ),
            )

            await KnowledgeDAO.update_import_job(
                self.session,
                job.id,
                step="chunking",
                progress_pct=40,
            )
            chunk_payloads = self._build_chunks(blocks, filename=safe_name)
            if not chunk_payloads:
                raise UploadError("No text chunks produced from file")
            logger.info(
                "Upload chunked | job_id={} chunks={}",
                job.id,
                len(chunk_payloads),
            )

            await KnowledgeDAO.update_import_job(
                self.session,
                job.id,
                step="embedding",
                progress_pct=60,
            )
            embedded: list[
                tuple[str, list[float], dict[str, Any] | None]
            ] = []
            total = len(chunk_payloads)
            for index, payload in enumerate(chunk_payloads):
                vector = await self.embeddings.embed(payload["content"])
                meta = payload.get("metadata") or {}
                chunk_tags: dict[str, Any] | None = None
                if meta:
                    chunk_tags = {
                        "system": {
                            "chunk_index": meta.get("chunk_index", index),
                            "section_title": meta.get("section_title"),
                            "is_heading_only": bool(
                                meta.get("is_heading_only")
                            ),
                        }
                    }
                embedded.append((payload["content"], vector, chunk_tags))
                if total and index % max(1, total // 5) == 0:
                    progress = 60 + int(30 * (index + 1) / total)
                    await KnowledgeDAO.update_import_job(
                        self.session,
                        job.id,
                        progress_pct=min(progress, 90),
                    )
                    logger.debug(
                        "Upload embedding progress | job_id={} {}/{}",
                        job.id,
                        index + 1,
                        total,
                    )

            doc_title = (
                (safe_title or Path(safe_name).stem).strip() or safe_name
            )
            document_tags = self._merge_upload_tags(
                tags,
                filename=safe_name,
                content_type=content_type,
            )
            existing = await KnowledgeDAO.get_by_source_origin(
                self.session,
                specialist_id=specialist_id,
                source_origin=safe_name,
            )
            if existing is not None:
                logger.info(
                    "Upload upsert | job_id={} existing_document_id={} "
                    "file={!r}",
                    job.id,
                    existing.id,
                    safe_name,
                )
                document = await KnowledgeDAO.replace_document_chunks(
                    self.session,
                    document_id=existing.id,
                    title=doc_title,
                    chunks=embedded,
                    tags=document_tags,
                    specialist_id=specialist_id,
                )
            else:
                document = await KnowledgeDAO.create_document_with_chunks(
                    self.session,
                    specialist_id=specialist_id,
                    title=doc_title,
                    chunks=embedded,
                    source_type="file_upload",
                    source_origin=safe_name,
                    tags=document_tags,
                )

            await KnowledgeDAO.update_import_job(
                self.session,
                job.id,
                status="done",
                step="done",
                progress_pct=100,
            )
            logger.info(
                "Upload done | job_id={} document_id={} chunks={}",
                job.id,
                document.id,
                len(document.chunks),
            )
            return document, job.id
        except (ExtractionError, UploadError) as exc:
            logger.warning(
                "Upload failed | job_id={} err={}",
                job.id,
                exc,
            )
            await KnowledgeDAO.update_import_job(
                self.session,
                job.id,
                status="failed",
                step="failed",
                error_message=str(exc),
            )
            raise UploadError(str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            logger.exception("Upload crashed | job_id={} err={}", job.id, exc)
            await KnowledgeDAO.update_import_job(
                self.session,
                job.id,
                status="failed",
                step="failed",
                error_message=str(exc),
            )
            raise UploadError(f"Upload failed: {exc}") from exc

    def _build_chunks(
        self,
        blocks: list[str] | list[dict[str, Any]],
        *,
        filename: str | None = None,
    ) -> list[dict[str, Any]]:
        # TXT: store the entire file as one chunk (no size-based splitting).
        if filename and Path(filename).suffix.lower() == ".txt":
            if not blocks:
                return []
            first = blocks[0]
            text = first if isinstance(first, str) else str(first.get("content", ""))
            if not text.strip():
                return []
            return [
                {
                    "content": text,
                    "metadata": {
                        "chunk_index": 0,
                        "section_title": None,
                        "is_heading_only": False,
                    },
                }
            ]

        if (
            isinstance(blocks, list)
            and blocks
            and isinstance(blocks[0], dict)
            and "content" in blocks[0]
        ):
            return chunk_text_structured(blocks, settings=self.settings)

        plain = chunk_text(blocks, settings=self.settings)  # type: ignore[arg-type]
        return [
            {
                "content": content,
                "metadata": {
                    "chunk_index": index,
                    "section_title": None,
                    "is_heading_only": False,
                },
            }
            for index, content in enumerate(plain)
        ]

    @staticmethod
    def _merge_upload_tags(
        tags: dict[str, Any] | list[Any] | None,
        *,
        filename: str,
        content_type: str | None,
    ) -> dict[str, Any]:
        system: dict[str, Any] = {
            "filename": filename,
            "content_type": content_type,
        }
        if tags is None:
            return {"system": system}

        normalized = normalize_knowledge_tags(tags)
        if normalized is None:
            return {"system": system}

        merged = dict(normalized)
        existing_system = merged.get("system")
        if isinstance(existing_system, dict):
            system = {
                **system,
                **{
                    key: value
                    for key, value in existing_system.items()
                    if key != "labels"
                },
            }
        # Keep upload filename/content_type authoritative for file origin.
        system["filename"] = filename
        system["content_type"] = content_type
        merged["system"] = system
        return merged


def parse_upload_tags(raw: str | None) -> dict[str, Any] | list[Any] | None:
    """Parse optional multipart ``tags`` JSON field."""
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise UploadError(f"Invalid tags JSON: {exc}") from exc
    if isinstance(parsed, (dict, list)):
        return parsed
    raise UploadError("tags must be a JSON object or array")
