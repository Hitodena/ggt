from pathlib import Path

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
        job = await KnowledgeDAO.create_import_job(
            self.session,
            specialist_id=specialist_id,
            filename=safe_name,
            content_type=content_type,
            byte_size=len(data),
        )

        try:
            await KnowledgeDAO.update_import_job(
                self.session,
                job.id,
                step="extracting",
                progress_pct=10,
            )
            blocks = self.extractor.extract(safe_name, data)

            await KnowledgeDAO.update_import_job(
                self.session,
                job.id,
                step="chunking",
                progress_pct=40,
            )
            chunks = chunk_text(blocks, settings=self.settings)
            if not chunks:
                raise UploadError("No text chunks produced from file")

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
            return document, job.id
        except (ExtractionError, UploadError) as exc:
            await KnowledgeDAO.update_import_job(
                self.session,
                job.id,
                status="failed",
                step="failed",
                error_message=str(exc),
            )
            raise UploadError(str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            await KnowledgeDAO.update_import_job(
                self.session,
                job.id,
                status="failed",
                step="failed",
                error_message=str(exc),
            )
            raise UploadError(f"Upload failed: {exc}") from exc
