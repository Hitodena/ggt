"""Unit tests for upload upsert-by-filename behavior."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.config import Settings
from app.dao import knowledge as knowledge_mod
from app.services.upload import UploadService


def _settings() -> Settings:
    return Settings(  # type: ignore[arg-type]
        database_url="postgresql+asyncpg://u:p@localhost/db",
        chunk_size=1500,
        chunk_overlap=200,
        max_upload_size_mb=25,
    )


@pytest.mark.asyncio
async def test_upload_creates_when_filename_missing() -> None:
    session = AsyncMock()
    embeddings = MagicMock()
    embeddings.embed = AsyncMock(return_value=[0.1] * 1536)
    extractor = MagicMock()
    extractor.extract = MagicMock(return_value=["hello world"])

    created = MagicMock()
    created.id = "doc-new"
    created.chunks = [MagicMock()]

    job = MagicMock()
    job.id = "job-1"

    original_job = knowledge_mod.KnowledgeDAO.create_import_job
    original_update = knowledge_mod.KnowledgeDAO.update_import_job
    original_get = knowledge_mod.KnowledgeDAO.get_by_source_origin
    original_create = knowledge_mod.KnowledgeDAO.create_document_with_chunks
    original_replace = knowledge_mod.KnowledgeDAO.replace_document_chunks

    async def fake_job(*_a, **_k):
        return job

    async def fake_update(*_a, **_k):
        return None

    async def fake_get(*_a, **_k):
        return None

    async def fake_create(*_a, **_k):
        assert _k["source_origin"] == "note.txt"
        assert _k["tags"]["labels"] == ["manual:a"]
        return created

    async def fake_replace(*_a, **_k):
        raise AssertionError("replace should not be called")

    knowledge_mod.KnowledgeDAO.create_import_job = staticmethod(fake_job)
    knowledge_mod.KnowledgeDAO.update_import_job = staticmethod(fake_update)
    knowledge_mod.KnowledgeDAO.get_by_source_origin = staticmethod(fake_get)
    knowledge_mod.KnowledgeDAO.create_document_with_chunks = staticmethod(
        fake_create
    )
    knowledge_mod.KnowledgeDAO.replace_document_chunks = staticmethod(
        fake_replace
    )
    try:
        service = UploadService(
            session,
            embeddings=embeddings,
            extractor=extractor,
            settings=_settings(),
        )
        document, job_id = await service.upload(
            specialist_id="spec-1",
            filename="note.txt",
            data=b"hello world",
            content_type="text/plain",
            tags={"labels": ["manual:a"]},
        )
        assert document.id == "doc-new"
        assert job_id == "job-1"
    finally:
        knowledge_mod.KnowledgeDAO.create_import_job = original_job
        knowledge_mod.KnowledgeDAO.update_import_job = original_update
        knowledge_mod.KnowledgeDAO.get_by_source_origin = original_get
        knowledge_mod.KnowledgeDAO.create_document_with_chunks = original_create
        knowledge_mod.KnowledgeDAO.replace_document_chunks = original_replace


@pytest.mark.asyncio
async def test_upload_replaces_when_filename_exists() -> None:
    session = AsyncMock()
    embeddings = MagicMock()
    embeddings.embed = AsyncMock(return_value=[0.2] * 1536)
    extractor = MagicMock()
    extractor.extract = MagicMock(return_value=["updated body"])

    existing = MagicMock()
    existing.id = "doc-existing"

    replaced = MagicMock()
    replaced.id = "doc-existing"
    replaced.chunks = [MagicMock(), MagicMock()]

    job = MagicMock()
    job.id = "job-2"

    original_job = knowledge_mod.KnowledgeDAO.create_import_job
    original_update = knowledge_mod.KnowledgeDAO.update_import_job
    original_get = knowledge_mod.KnowledgeDAO.get_by_source_origin
    original_create = knowledge_mod.KnowledgeDAO.create_document_with_chunks
    original_replace = knowledge_mod.KnowledgeDAO.replace_document_chunks

    async def fake_job(*_a, **_k):
        return job

    async def fake_update(*_a, **_k):
        return None

    async def fake_get(*_a, **_k):
        assert _k["source_origin"] == "guide.txt"
        return existing

    async def fake_create(*_a, **_k):
        raise AssertionError("create should not be called on upsert")

    async def fake_replace(*_a, **_k):
        assert _k["document_id"] == "doc-existing"
        assert _k["title"] == "guide"
        assert _k["tags"]["audience"] == ["sex:female"]
        assert len(_k["chunks"]) == 1
        assert _k["chunks"][0][0] == "updated body"
        return replaced

    knowledge_mod.KnowledgeDAO.create_import_job = staticmethod(fake_job)
    knowledge_mod.KnowledgeDAO.update_import_job = staticmethod(fake_update)
    knowledge_mod.KnowledgeDAO.get_by_source_origin = staticmethod(fake_get)
    knowledge_mod.KnowledgeDAO.create_document_with_chunks = staticmethod(
        fake_create
    )
    knowledge_mod.KnowledgeDAO.replace_document_chunks = staticmethod(
        fake_replace
    )
    try:
        service = UploadService(
            session,
            embeddings=embeddings,
            extractor=extractor,
            settings=_settings(),
        )
        document, job_id = await service.upload(
            specialist_id="spec-1",
            filename="guide.txt",
            data=b"updated body",
            content_type="text/plain",
            tags={"audience": ["sex:female"]},
        )
        assert document.id == "doc-existing"
        assert job_id == "job-2"
        assert len(document.chunks) == 2
    finally:
        knowledge_mod.KnowledgeDAO.create_import_job = original_job
        knowledge_mod.KnowledgeDAO.update_import_job = original_update
        knowledge_mod.KnowledgeDAO.get_by_source_origin = original_get
        knowledge_mod.KnowledgeDAO.create_document_with_chunks = original_create
        knowledge_mod.KnowledgeDAO.replace_document_chunks = original_replace
