from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class KnowledgeCreate(BaseModel):
    specialist_id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    source_type: str = Field(default="manual")
    source_origin: str = Field(default="api")
    tags: list[str] | dict[str, Any] | None = None


class KnowledgeFromMessage(BaseModel):
    specialist_id: str = Field(..., min_length=1)
    message_id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    source_type: str = Field(default="chat_message")
    source_origin: str = Field(default="chat")
    tags: list[str] | dict[str, Any] | None = None


class KnowledgeSearchRequest(BaseModel):
    specialist_id: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)
    limit: int = Field(default=5, ge=1, le=50)


class KnowledgeAnswerRequest(BaseModel):
    specialist_id: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)
    limit: int = Field(default=5, ge=1, le=20)


class KnowledgeChunkOut(BaseModel):
    id: str
    content: str
    tags: list[str] | dict[str, Any] | None = None
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
    tags: list[str] | dict[str, Any] | None = None


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
