from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.dao.knowledge import KnowledgeDAO
from app.schemas import (
    KnowledgeAnswerRequest,
    KnowledgeAnswerResponse,
    KnowledgeSearchHit,
)
from app.services.embeddings import EmbeddingService

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
            self._client = AsyncOpenAI(
                api_key=self.settings.openai_api_key,
                base_url=self.settings.openai_base_url,
            )

    async def answer(
        self,
        payload: KnowledgeAnswerRequest,
    ) -> KnowledgeAnswerResponse:
        vector = await self.embeddings.embed(payload.query)
        rows = await KnowledgeDAO.search_similar(
            self.session,
            specialist_id=payload.specialist_id,
            query_embedding=vector,
            limit=payload.limit,
        )
        sources = [
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
        if not sources:
            return KnowledgeAnswerResponse(
                query=payload.query,
                specialist_id=payload.specialist_id,
                answer=NO_HITS_ANSWER,
                sources=[],
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
            answer = NO_HITS_ANSWER

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
