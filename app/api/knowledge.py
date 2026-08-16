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
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
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
from app.services.knowledge import KnowledgeService
from app.services.upload import UploadError, UploadService

router = APIRouter(prefix="/knowledge", tags=["knowledge"])


def _to_out(document) -> KnowledgeDocumentOut:
    return KnowledgeDocumentOut.model_validate(document)


@router.post(
    "",
    response_model=KnowledgeDocumentOut,
    status_code=status.HTTP_201_CREATED,
    summary="Add knowledge manually",
)
async def create_knowledge(
    body: KnowledgeCreate,
    session: AsyncSession = Depends(get_session),
) -> KnowledgeDocumentOut:
    service = KnowledgeService(session)
    document = await service.create(body)
    return _to_out(document)


@router.post(
    "/from-message",
    response_model=KnowledgeDocumentOut,
    summary="Add chat message to specialist knowledge base",
)
async def create_from_message(
    body: KnowledgeFromMessage,
    response: Response,
    session: AsyncSession = Depends(get_session),
) -> KnowledgeDocumentOut:
    service = KnowledgeService(session)
    document, created = await service.create_from_message(body)
    response.status_code = (
        status.HTTP_201_CREATED if created else status.HTTP_200_OK
    )
    return _to_out(document)


@router.post(
    "/upload",
    response_model=KnowledgeUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a file into the specialist knowledge base",
)
async def upload_knowledge(
    specialist_id: str = Form(..., min_length=1),
    title: str | None = Form(None),
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_session),
) -> KnowledgeUploadResponse:
    settings = get_settings()
    data = await file.read()
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds max size of {settings.max_upload_size_mb} MB",
        )

    service = UploadService(session, settings=settings)
    try:
        document, job_id = await service.upload(
            specialist_id=specialist_id,
            filename=file.filename or "upload.bin",
            data=data,
            content_type=file.content_type,
            title=title,
        )
    except UploadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return KnowledgeUploadResponse(
        document=_to_out(document),
        import_job_id=job_id,
        chunks_count=len(document.chunks),
        filename=file.filename or "upload.bin",
    )


@router.post(
    "/search",
    response_model=KnowledgeSearchResponse,
    summary="Semantic search in specialist knowledge base",
)
async def search_knowledge(
    body: KnowledgeSearchRequest,
    session: AsyncSession = Depends(get_session),
) -> KnowledgeSearchResponse:
    service = KnowledgeService(session)
    return await service.search(body)


@router.post(
    "/answer",
    response_model=KnowledgeAnswerResponse,
    summary="RAG answer from specialist knowledge base",
)
async def answer_knowledge(
    body: KnowledgeAnswerRequest,
    session: AsyncSession = Depends(get_session),
) -> KnowledgeAnswerResponse:
    service = AnswerService(session)
    return await service.answer(body)


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
    items, total = await KnowledgeDAO.list_for_specialist(
        session,
        specialist_id=specialist_id,
        limit=limit,
        offset=offset,
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
    document = await KnowledgeDAO.get_by_id(
        session,
        document_id,
        specialist_id=specialist_id,
    )
    if document is None:
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
    deleted = await KnowledgeDAO.delete_document(
        session,
        document_id,
        specialist_id=specialist_id,
    )
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
