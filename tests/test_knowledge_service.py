"""Unit tests that do not require Neon or OpenAI."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas import KnowledgeCreate, KnowledgeFromMessage, KnowledgeSearchRequest
from app.services.knowledge import KnowledgeService


def test_health() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_create_from_message_idempotent() -> None:
    session = AsyncMock()
    embeddings = MagicMock()
    embeddings.embed = AsyncMock(return_value=[0.1] * 1536)

    existing = MagicMock()
    existing.id = "doc-1"

    service = KnowledgeService(session, embeddings=embeddings)

    # Patch DAO methods used by the service
    from app.dao import knowledge as knowledge_mod

    original_get = knowledge_mod.KnowledgeDAO.get_by_origin_message
    original_create = knowledge_mod.KnowledgeDAO.create_document_with_chunk

    async def fake_get(*_args, **_kwargs):
        return existing

    knowledge_mod.KnowledgeDAO.get_by_origin_message = staticmethod(fake_get)
    try:
        doc, created = await service.create_from_message(
            KnowledgeFromMessage(
                specialist_id="spec-1",
                message_id="msg-1",
                title="t",
                content="c",
            )
        )
        assert created is False
        assert doc is existing
        embeddings.embed.assert_not_called()
    finally:
        knowledge_mod.KnowledgeDAO.get_by_origin_message = original_get
        knowledge_mod.KnowledgeDAO.create_document_with_chunk = original_create


@pytest.mark.asyncio
async def test_create_calls_embed() -> None:
    session = AsyncMock()
    embeddings = MagicMock()
    embeddings.embed = AsyncMock(return_value=[0.2] * 1536)

    created_doc = MagicMock()
    created_doc.id = "doc-new"

    from app.dao import knowledge as knowledge_mod

    original_create = knowledge_mod.KnowledgeDAO.create_document_with_chunk

    async def fake_create(*_args, **_kwargs):
        assert len(_kwargs["embedding"]) == 1536
        return created_doc

    knowledge_mod.KnowledgeDAO.create_document_with_chunk = staticmethod(
        fake_create
    )
    try:
        service = KnowledgeService(session, embeddings=embeddings)
        doc = await service.create(
            KnowledgeCreate(
                specialist_id="spec-1",
                title="Заголовок",
                content="Русский текст для базы знаний",
            )
        )
        assert doc is created_doc
        embeddings.embed.assert_awaited_once()
    finally:
        knowledge_mod.KnowledgeDAO.create_document_with_chunk = original_create


@pytest.mark.asyncio
async def test_search_maps_hits() -> None:
    session = AsyncMock()
    embeddings = MagicMock()
    embeddings.embed = AsyncMock(return_value=[0.3] * 1536)

    chunk = MagicMock()
    chunk.id = "chunk-1"
    chunk.content = "контент"
    chunk.tags = ["a"]
    document = MagicMock()
    document.id = "doc-1"
    document.title = "title"

    from app.dao import knowledge as knowledge_mod

    original_search = knowledge_mod.KnowledgeDAO.search_similar

    async def fake_search(*_args, **_kwargs):
        return [(chunk, document, 0.12)]

    knowledge_mod.KnowledgeDAO.search_similar = staticmethod(fake_search)
    try:
        service = KnowledgeService(session, embeddings=embeddings)
        result = await service.search(
            KnowledgeSearchRequest(
                specialist_id="spec-1",
                query="чувствительная кожа",
                limit=3,
            )
        )
        assert result.specialist_id == "spec-1"
        assert len(result.hits) == 1
        assert result.hits[0].distance == 0.12
        assert result.hits[0].content == "контент"
    finally:
        knowledge_mod.KnowledgeDAO.search_similar = original_search


@pytest.mark.asyncio
async def test_search_passes_filter_tags() -> None:
    session = AsyncMock()
    embeddings = MagicMock()
    embeddings.embed = AsyncMock(return_value=[0.3] * 1536)

    from app.dao import knowledge as knowledge_mod

    original_search = knowledge_mod.KnowledgeDAO.search_similar
    captured: dict = {}

    async def fake_search(*_args, **_kwargs):
        captured.update(_kwargs)
        return []

    knowledge_mod.KnowledgeDAO.search_similar = staticmethod(fake_search)
    try:
        service = KnowledgeService(session, embeddings=embeddings)
        filter_tags = {"audience": {"gender": "male", "age_min": 40}}
        await service.search(
            KnowledgeSearchRequest(
                specialist_id="spec-1",
                query="протокол",
                limit=3,
                filter_tags=filter_tags,
            )
        )
        assert captured["filter_tags"] == filter_tags
        assert captured["specialist_id"] == "spec-1"
        assert captured["limit"] == 3
    finally:
        knowledge_mod.KnowledgeDAO.search_similar = original_search
