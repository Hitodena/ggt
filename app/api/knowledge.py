from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Response,
    UploadFile,
    status,
)
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.filenames import normalize_upload_filename
from app.dao.knowledge import KnowledgeDAO
from app.db.session import get_session
from app.schemas import (
    KnowledgeAnswerRequest,
    KnowledgeAnswerResponse,
    KnowledgeCreate,
    KnowledgeDocumentOut,
    KnowledgeFromMessage,
    KnowledgeListResponse,
    KnowledgeSearchRequest,
    KnowledgeSearchResponse,
    KnowledgeUploadResponse,
)
from app.services.answer import AnswerService
from app.services.embeddings import EmbeddingError
from app.services.knowledge import KnowledgeService
from app.services.upload import UploadError, UploadService, parse_upload_tags

router = APIRouter(prefix="/knowledge", tags=["knowledge"])


def _to_out(document) -> KnowledgeDocumentOut:
    return KnowledgeDocumentOut.model_validate(document)


@router.post(
    "",
    response_model=KnowledgeDocumentOut,
    status_code=status.HTTP_201_CREATED,
    summary="Add knowledge manually",
    description=(
        "Create a knowledge document with one embedded chunk. "
        "Optional `tags.audience` marks a segment note "
        "(e.g. gender/age) that is hidden from default search/answer "
        "until a matching `filter_tags.audience` is passed."
    ),
)
async def create_knowledge(
    body: KnowledgeCreate,
    session: AsyncSession = Depends(get_session),
) -> KnowledgeDocumentOut:
    logger.info(
        "API create knowledge | specialist_id={} title={!r} chars={}",
        body.specialist_id,
        body.title,
        len(body.content),
    )
    service = KnowledgeService(session)
    try:
        document = await service.create(body)
    except EmbeddingError as exc:
        logger.error("API create embed error | err={}", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    logger.info("API create knowledge done | document_id={}", document.id)
    return _to_out(document)


@router.post(
    "/from-message",
    response_model=KnowledgeDocumentOut,
    summary="Add chat message to specialist knowledge base",
    description=(
        "Idempotent by `(specialist_id, message_id)`. "
        "Supports the same `tags.system` / `tags.audience` shape as "
        "manual create. Segment (`audience`) notes stay hidden from "
        "default search until `filter_tags` is provided."
    ),
)
async def create_from_message(
    body: KnowledgeFromMessage,
    response: Response,
    session: AsyncSession = Depends(get_session),
) -> KnowledgeDocumentOut:
    logger.info(
        "API from-message | specialist_id={} message_id={} title={!r}",
        body.specialist_id,
        body.message_id,
        body.title,
    )
    service = KnowledgeService(session)
    try:
        document, created = await service.create_from_message(body)
    except EmbeddingError as exc:
        logger.error("API from-message embed error | err={}", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    response.status_code = (
        status.HTTP_201_CREATED if created else status.HTTP_200_OK
    )
    logger.info(
        "API from-message done | document_id={} created={}",
        document.id,
        created,
    )
    return _to_out(document)


@router.post(
    "/upload",
    response_model=KnowledgeUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a file into the specialist knowledge base",
    description=(
        "Extract text from pdf/doc/docx/xls/xlsx, chunk, embed, and store. "
        "Technical metadata is stored under `tags.system` "
        "(`filename`, `content_type`, chunk metadata) and does **not** hide "
        "chunks from search. Optional multipart `tags` field accepts a JSON "
        "object/array (e.g. audience filters)."
    ),
)
async def upload_knowledge(
    specialist_id: str = Form(..., min_length=1),
    title: str | None = Form(None),
    tags: str | None = Form(
        None,
        description=(
            'Optional JSON tags, e.g. '
            '`{"audience": {"gender": "male"}}` or `{"system": {"labels": ["x"]}}`'
        ),
    ),
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_session),
) -> KnowledgeUploadResponse:
    settings = get_settings()
    data = await file.read()
    filename = normalize_upload_filename(file.filename)
    title_norm = (
        normalize_upload_filename(title, fallback="") if title else None
    )
    if title_norm == "":
        title_norm = None
    logger.info(
        "API upload | specialist_id={} filename={!r} content_type={} "
        "bytes={} title={!r}",
        specialist_id,
        filename,
        file.content_type,
        len(data),
        title_norm,
    )
    if len(data) > settings.max_upload_bytes:
        logger.warning(
            "API upload rejected: file too large | bytes={} max={}",
            len(data),
            settings.max_upload_bytes,
        )
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds max size of {settings.max_upload_size_mb} MB",
        )

    try:
        parsed_tags = parse_upload_tags(tags)
    except UploadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    service = UploadService(session, settings=settings)
    try:
        document, job_id = await service.upload(
            specialist_id=specialist_id,
            filename=filename,
            data=data,
            content_type=file.content_type,
            title=title_norm,
            tags=parsed_tags,
        )
    except UploadError as exc:
        logger.warning(
            "API upload failed | specialist_id={} err={}",
            specialist_id,
            exc,
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    logger.info(
        "API upload done | document_id={} job_id={} chunks={}",
        document.id,
        job_id,
        len(document.chunks),
    )
    return KnowledgeUploadResponse(
        document=_to_out(document),
        import_job_id=job_id,
        chunks_count=len(document.chunks),
        filename=filename,
    )


@router.post(
    "/search",
    response_model=KnowledgeSearchResponse,
    summary="Semantic search in specialist knowledge base",
    description=(
        "Cosine similarity search over chunk embeddings. "
        "By default, chunks with `tags.audience` are excluded. "
        "Pass `filter_tags.audience` to also include matching segment "
        "notes (plus general notes without audience)."
    ),
)
async def search_knowledge(
    body: KnowledgeSearchRequest,
    session: AsyncSession = Depends(get_session),
) -> KnowledgeSearchResponse:
    logger.info(
        "API search | specialist_id={} query={!r} limit={}",
        body.specialist_id,
        body.query,
        body.limit,
    )
    service = KnowledgeService(session)
    try:
        result = await service.search(body)
    except EmbeddingError as exc:
        logger.error("API search embed error | err={}", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    logger.info(
        "API search done | specialist_id={} hits={}",
        body.specialist_id,
        len(result.hits),
    )
    return result


@router.post(
    "/answer",
    response_model=KnowledgeAnswerResponse,
    summary="RAG answer from specialist knowledge base",
    description=(
        "Retrieve similar chunks (same audience visibility rules as "
        "`/knowledge/search`), then ask the chat model to answer using "
        "only that context. Use `filter_tags.audience` when the client "
        "profile should unlock segment-specific notes."
    ),
)
async def answer_knowledge(
    body: KnowledgeAnswerRequest,
    session: AsyncSession = Depends(get_session),
) -> KnowledgeAnswerResponse:
    logger.info(
        "API answer | specialist_id={} query={!r} limit={}",
        body.specialist_id,
        body.query,
        body.limit,
    )
    service = AnswerService(session)
    result = await service.answer(body)
    logger.info(
        "API answer done | specialist_id={} sources={} answer_chars={}",
        body.specialist_id,
        len(result.sources),
        len(result.answer),
    )
    return result


@router.get(
    "",
    response_model=KnowledgeListResponse,
    summary="List knowledge documents for a specialist",
)
async def list_knowledge(
    specialist_id: str = Query(..., min_length=1),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    session: AsyncSession = Depends(get_session),
) -> KnowledgeListResponse:
    logger.info(
        "API list | specialist_id={} limit={} offset={}",
        specialist_id,
        limit,
        offset,
    )
    items, total = await KnowledgeDAO.list_for_specialist(
        session,
        specialist_id=specialist_id,
        limit=limit,
        offset=offset,
    )
    logger.info(
        "API list done | specialist_id={} returned={} total={}",
        specialist_id,
        len(items),
        total,
    )
    return KnowledgeListResponse(
        items=[_to_out(item) for item in items],
        total=total,
    )


@router.get(
    "/{document_id}",
    response_model=KnowledgeDocumentOut,
    summary="Get a knowledge document",
)
async def get_knowledge(
    document_id: str,
    specialist_id: str | None = Query(None),
    session: AsyncSession = Depends(get_session),
) -> KnowledgeDocumentOut:
    logger.info(
        "API get | document_id={} specialist_id={}",
        document_id,
        specialist_id,
    )
    document = await KnowledgeDAO.get_by_id(
        session,
        document_id,
        specialist_id=specialist_id,
    )
    if document is None:
        logger.warning("API get: document not found | id={}", document_id)
        raise HTTPException(status_code=404, detail="Document not found")
    return _to_out(document)


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a knowledge document and its chunks",
)
async def delete_knowledge(
    document_id: str,
    specialist_id: str | None = Query(None),
    session: AsyncSession = Depends(get_session),
) -> Response:
    logger.info(
        "API delete | document_id={} specialist_id={}",
        document_id,
        specialist_id,
    )
    deleted = await KnowledgeDAO.delete_document(
        session,
        document_id,
        specialist_id=specialist_id,
    )
    if not deleted:
        logger.warning("API delete: document not found | id={}", document_id)
        raise HTTPException(status_code=404, detail="Document not found")
    logger.info("API delete done | document_id={}", document_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
