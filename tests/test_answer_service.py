"""Tests for RAG answer service with mocked LLM/embeddings."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.schemas import KnowledgeAnswerRequest
from app.services.answer import NO_HITS_ANSWER, AnswerService


@pytest.mark.asyncio
async def test_answer_without_hits_skips_chat() -> None:
    session = AsyncMock()
    embeddings = MagicMock()
    embeddings.embed = AsyncMock(return_value=[0.1] * 1536)
    client = MagicMock()
    client.chat.completions.create = AsyncMock()

    from app.dao import knowledge as knowledge_mod

    original = knowledge_mod.KnowledgeDAO.search_similar

    async def fake_search(*_args, **_kwargs):
        return []

    knowledge_mod.KnowledgeDAO.search_similar = staticmethod(fake_search)
    try:
        service = AnswerService(
            session,
            embeddings=embeddings,
            client=client,
        )
        result = await service.answer(
            KnowledgeAnswerRequest(
                specialist_id="spec-1",
                query="что делать после пилинга?",
            )
        )
        assert result.answer == NO_HITS_ANSWER
        assert result.sources == []
        client.chat.completions.create.assert_not_called()
    finally:
        knowledge_mod.KnowledgeDAO.search_similar = original


@pytest.mark.asyncio
async def test_answer_with_hits_calls_chat() -> None:
    session = AsyncMock()
    embeddings = MagicMock()
    embeddings.embed = AsyncMock(return_value=[0.2] * 1536)

    chunk = MagicMock()
    chunk.id = "chunk-1"
    chunk.content = "После пилинга использовать SPF."
    chunk.tags = None
    document = MagicMock()
    document.id = "doc-1"
    document.title = "Протокол"

    completion = MagicMock()
    completion.choices = [
        MagicMock(message=MagicMock(content="Используйте SPF после процедуры."))
    ]
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=completion)

    from app.dao import knowledge as knowledge_mod

    original = knowledge_mod.KnowledgeDAO.search_similar

    async def fake_search(*_args, **_kwargs):
        return [(chunk, document, 0.05)]

    knowledge_mod.KnowledgeDAO.search_similar = staticmethod(fake_search)
    try:
        service = AnswerService(
            session,
            embeddings=embeddings,
            client=client,
        )
        result = await service.answer(
            KnowledgeAnswerRequest(
                specialist_id="spec-1",
                query="что после пилинга?",
                limit=3,
            )
        )
        assert "SPF" in result.answer
        assert len(result.sources) == 1
        assert result.sources[0].document_id == "doc-1"
        client.chat.completions.create.assert_awaited_once()
        call_kwargs = client.chat.completions.create.await_args.kwargs
        assert "После пилинга использовать SPF." in call_kwargs["messages"][1][
            "content"
        ]
    finally:
        knowledge_mod.KnowledgeDAO.search_similar = original
