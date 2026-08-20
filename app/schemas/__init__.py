from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

_TAGS_DESCRIPTION = (
    "Preferred shape: `{\"system\": {...}, \"audience\": {...}}`. "
    "`system` is technical metadata and does not hide the chunk. "
    "`audience` marks a segment note (e.g. men 40+) and hides it from "
    "default search/answer unless `filter_tags.audience` matches. "
    "A flat object without `system`/`audience` is stored as `audience`."
)

_FILTER_TAGS_DESCRIPTION = (
    "Optional audience filter. "
    "Without `filter_tags` / `filter_tags.audience`, chunks that have "
    "`tags.audience` are excluded. "
    "With `filter_tags.audience`, returns general chunks (no audience) "
    "plus chunks whose `tags.audience` contains this object "
    "(JSONB containment). "
    "Example: `{\"audience\": {\"gender\": \"male\", \"age_min\": 40}}`."
)


class KnowledgeCreate(BaseModel):
    model_config = ConfigDict(
        extra="ignore",
        json_schema_extra={
            "examples": [
                {
                    "specialist_id": "spec-1",
                    "title": "Протокол для мужчин 40+",
                    "content": "Рекомендации только для мужчин от 40 лет...",
                    "source_type": "manual",
                    "source_origin": "api",
                    "tags": {
                        "audience": {"gender": "male", "age_min": 40},
                    },
                }
            ]
        },
    )

    specialist_id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    source_type: str = Field(default="manual")
    source_origin: str = Field(default="api")
    tags: list[str] | dict[str, Any] | None = Field(
        default=None,
        description=_TAGS_DESCRIPTION,
        examples=[
            {"audience": {"gender": "male", "age_min": 40}},
            {"system": {"labels": ["protocol"]}, "audience": {"gender": "male"}},
        ],
    )


class KnowledgeFromMessage(BaseModel):
    model_config = ConfigDict(
        extra="ignore",
        json_schema_extra={
            "examples": [
                {
                    "specialist_id": "spec-1",
                    "message_id": "msg-42",
                    "title": "Протокол очистки",
                    "content": "Для чувствительной кожи использовать мягкий пилинг...",
                    "tags": None,
                }
            ]
        },
    )

    specialist_id: str = Field(..., min_length=1)
    message_id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    source_type: str = Field(default="chat_message")
    source_origin: str = Field(default="chat")
    tags: list[str] | dict[str, Any] | None = Field(
        default=None,
        description=_TAGS_DESCRIPTION,
        examples=[{"audience": {"gender": "male", "age_min": 40}}],
    )


class KnowledgeSearchRequest(BaseModel):
    model_config = ConfigDict(
        extra="ignore",
        json_schema_extra={
            "examples": [
                {
                    "specialist_id": "spec-1",
                    "query": "пилинг для чувствительной кожи",
                    "limit": 5,
                },
                {
                    "specialist_id": "spec-1",
                    "query": "протокол",
                    "limit": 5,
                    "filter_tags": {
                        "audience": {"gender": "male", "age_min": 40},
                    },
                },
            ]
        },
    )

    specialist_id: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)
    limit: int = Field(default=5, ge=1, le=50)
    filter_tags: dict[str, Any] | None = Field(
        default=None,
        description=_FILTER_TAGS_DESCRIPTION,
        examples=[{"audience": {"gender": "male", "age_min": 40}}],
    )


class KnowledgeAnswerRequest(BaseModel):
    model_config = ConfigDict(
        extra="ignore",
        json_schema_extra={
            "examples": [
                {
                    "specialist_id": "spec-1",
                    "query": "что делать после пилинга?",
                    "limit": 5,
                },
                {
                    "specialist_id": "spec-1",
                    "query": "какой протокол подходит?",
                    "limit": 5,
                    "filter_tags": {
                        "audience": {"gender": "male", "age_min": 40},
                    },
                },
            ]
        },
    )

    specialist_id: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)
    limit: int = Field(default=5, ge=1, le=20)
    filter_tags: dict[str, Any] | None = Field(
        default=None,
        description=_FILTER_TAGS_DESCRIPTION,
        examples=[{"audience": {"gender": "male", "age_min": 40}}],
    )


class KnowledgeChunkOut(BaseModel):
    id: str
    content: str
    tags: list[str] | dict[str, Any] | None = Field(
        default=None,
        description=_TAGS_DESCRIPTION,
    )
    distance: float | None = None

    model_config = {"from_attributes": True}


class KnowledgeDocumentOut(BaseModel):
    id: str
    specialist_id: str
    title: str
    source_type: str
    source_origin: str
    origin_message_id: str | None = None
    created_at: datetime
    updated_at: datetime
    chunks: list[KnowledgeChunkOut] = Field(default_factory=list)

    model_config = {"from_attributes": True}


class KnowledgeSearchHit(BaseModel):
    document_id: str
    document_title: str
    chunk_id: str
    content: str
    distance: float
    tags: list[str] | dict[str, Any] | None = Field(
        default=None,
        description=_TAGS_DESCRIPTION,
    )
    chunk_index: int | None = Field(
        default=None,
        description="0-based chunk order within the source document.",
    )
    section_title: str | None = Field(
        default=None,
        description="Heading/section title for this chunk when available.",
    )
    is_heading_only: bool | None = Field(
        default=None,
        description="True when the chunk is only a heading with no body.",
    )


class KnowledgeSearchResponse(BaseModel):
    query: str
    specialist_id: str
    hits: list[KnowledgeSearchHit]


class KnowledgeAnswerResponse(BaseModel):
    query: str
    specialist_id: str
    answer: str
    sources: list[KnowledgeSearchHit]


class KnowledgeListResponse(BaseModel):
    items: list[KnowledgeDocumentOut]
    total: int


class KnowledgeUploadResponse(BaseModel):
    document: KnowledgeDocumentOut
    import_job_id: str
    chunks_count: int
    filename: str
