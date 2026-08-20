from loguru import logger
from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.tags import extract_chunk_system_meta
from app.dao.knowledge import KnowledgeDAO
from app.schemas import (
    KnowledgeAnswerRequest,
    KnowledgeAnswerResponse,
    KnowledgeSearchHit,
)
from app.services.embeddings import EmbeddingService
from app.services.openai_client import build_openai_client

SYSTEM_PROMPT = (
    "Ты помощник специалиста. Отвечай на вопрос пользователя, используя "
    "только предоставленный контекст из базы знаний. Если в контексте нет "
    "достаточной информации, честно скажи, что в базе знаний этого нет. "
    "Отвечай на русском языке, кратко и по делу."
)

NO_HITS_ANSWER = (
    "В базе знаний специалиста нет релевантной информации по этому вопросу."
)


class AnswerService:
    def __init__(
        self,
        session: AsyncSession,
        *,
        embeddings: EmbeddingService | None = None,
        settings: Settings | None = None,
        client: AsyncOpenAI | None = None,
    ) -> None:
        self.session = session
        self.settings = settings or get_settings()
        self.embeddings = embeddings or EmbeddingService(self.settings)
        if client is not None:
            self._client = client
        else:
            if not self.settings.openai_api_key:
                raise RuntimeError(
                    "OPENAI_API_KEY is not set; cannot generate answers"
                )
            self._client = build_openai_client(self.settings)

    async def answer(
        self,
        payload: KnowledgeAnswerRequest,
    ) -> KnowledgeAnswerResponse:
        logger.info(
            "RAG answer start | specialist_id={} query={!r} limit={} "
            "model={} filter_tags={}",
            payload.specialist_id,
            payload.query,
            payload.limit,
            self.settings.chat_model,
            payload.filter_tags,
        )
        vector = await self.embeddings.embed(payload.query)
        rows = await KnowledgeDAO.search_similar(
            self.session,
            specialist_id=payload.specialist_id,
            query_embedding=vector,
            limit=payload.limit,
            filter_tags=payload.filter_tags,
        )
        sources = []
        for chunk, document, distance in rows:
            chunk_index, section_title, is_heading_only = (
                extract_chunk_system_meta(chunk.tags)
            )
            sources.append(
                KnowledgeSearchHit(
                    document_id=document.id,
                    document_title=document.title,
                    chunk_id=chunk.id,
                    content=chunk.content,
                    distance=distance,
                    tags=chunk.tags,
                    chunk_index=chunk_index,
                    section_title=section_title,
                    is_heading_only=is_heading_only,
                )
            )
        if not sources:
            logger.warning(
                "RAG no hits | specialist_id={} query={!r}",
                payload.specialist_id,
                payload.query,
            )
            return KnowledgeAnswerResponse(
                query=payload.query,
                specialist_id=payload.specialist_id,
                answer=NO_HITS_ANSWER,
                sources=[],
            )

        logger.info(
            "RAG hits | specialist_id={} count={} best_distance={:.4f}",
            payload.specialist_id,
            len(sources),
            sources[0].distance,
        )
        context = self._build_context(sources)
        completion = await self._client.chat.completions.create(
            model=self.settings.chat_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Вопрос: {payload.query}\n\n"
                        f"Контекст из базы знаний:\n{context}"
                    ),
                },
            ],
            temperature=0.2,
        )
        answer = (completion.choices[0].message.content or "").strip()
        if not answer:
            logger.warning("RAG chat returned empty answer")
            answer = NO_HITS_ANSWER

        logger.info(
            "RAG answer ready | specialist_id={} sources={} answer_chars={}",
            payload.specialist_id,
            len(sources),
            len(answer),
        )
        return KnowledgeAnswerResponse(
            query=payload.query,
            specialist_id=payload.specialist_id,
            answer=answer,
            sources=sources,
        )

    @staticmethod
    def _build_context(sources: list[KnowledgeSearchHit]) -> str:
        parts: list[str] = []
        for index, hit in enumerate(sources, start=1):
            parts.append(
                f"[{index}] {hit.document_title}\n{hit.content}"
            )
        return "\n\n".join(parts)
