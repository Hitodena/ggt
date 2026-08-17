from pathlib import Path

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.dao.knowledge import KnowledgeDAO
from app.models.kb import KBDocument
from app.services.chunking import chunk_text
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
    ) -> tuple[KBDocument, str]:
        if not specialist_id.strip():
            raise UploadError("specialist_id is required")
        if not filename.strip():
            raise UploadError("filename is required")
        if len(data) > self.settings.max_upload_bytes:
            raise UploadError(
                f"File exceeds max size of {self.settings.max_upload_size_mb} MB"
            )

        safe_name = Path(filename).name
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
                sum(len(b) for b in blocks),
            )

            await KnowledgeDAO.update_import_job(
                self.session,
                job.id,
                step="chunking",
                progress_pct=40,
            )
            chunks = chunk_text(blocks, settings=self.settings)
            if not chunks:
                raise UploadError("No text chunks produced from file")
            logger.info(
                "Upload chunked | job_id={} chunks={}",
                job.id,
                len(chunks),
            )

            await KnowledgeDAO.update_import_job(
                self.session,
                job.id,
                step="embedding",
                progress_pct=60,
            )
            embedded: list[tuple[str, list[float]]] = []
            for index, chunk in enumerate(chunks):
                vector = await self.embeddings.embed(chunk)
                embedded.append((chunk, vector))
                if chunks and index % max(1, len(chunks) // 5) == 0:
                    progress = 60 + int(30 * (index + 1) / len(chunks))
                    await KnowledgeDAO.update_import_job(
                        self.session,
                        job.id,
                        progress_pct=min(progress, 90),
                    )
                    logger.debug(
                        "Upload embedding progress | job_id={} {}/{}",
                        job.id,
                        index + 1,
                        len(chunks),
                    )

            doc_title = (title or Path(safe_name).stem).strip() or safe_name
            document = await KnowledgeDAO.create_document_with_chunks(
                self.session,
                specialist_id=specialist_id,
                title=doc_title,
                chunks=embedded,
                source_type="file_upload",
                source_origin=safe_name,
                tags={"filename": safe_name, "content_type": content_type},
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
