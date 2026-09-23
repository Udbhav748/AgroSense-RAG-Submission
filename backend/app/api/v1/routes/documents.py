"""Routes for PDF document upload, listing, and management."""

import logging

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, Request, Response, UploadFile, status
from fastapi.responses import FileResponse

from app.api.v1.routes.query import get_image_vector_store, get_llm_client, get_vector_store
from app.core import permissions
from app.core.auth import require_auth
from app.core.config import settings
from app.core.exceptions import (
    ApprovalRequiredError,
    ConfirmationRequiredError,
    DocumentNotFoundError,
    TaskNotFoundError,
    VectorStoreNotFoundError,
)
from app.models.schemas import (
    AsyncTaskResponse,
    DocumentDeleteResponse,
    DocumentImageItem,
    DocumentImagesResponse,
    DocumentListResponse,
    DocumentProcessingResponse,
    HighlightResponse,
    TaskStatusResponse,
)
from app.services import approval_service, s3_sync_service
from app.services.document_processing_service import DocumentProcessingService
from app.services.document_repository import (
    delete_document_metadata,
    get_document_owner,
    list_documents,
)
from app.services.image_captioning_service import image_storage_dir, load_image_manifest
from app.services.task_service import execute_background_ingestion, get_task_service
from app.services.upload_service import UPLOAD_DIR, decrypt_upload_bytes, save_uploaded_file
from app.services.validation_service import validate_pdf_upload

router = APIRouter(tags=["Documents"], dependencies=[Depends(require_auth)])
logger = logging.getLogger(__name__)


@router.get("/documents", response_model=DocumentListResponse)
def list_uploaded_documents(
    request: Request, all_tenants: bool = False, collection: str | None = None
) -> DocumentListResponse:
    """List the caller's documents (tenant-scoped when the DB is enabled;
    empty when the DB is disabled).

    all_tenants=true is one of two permission-gated actions in this app
    (see app/core/permissions.py's registry — the other is DELETE
    /documents/{id}): an admin can opt into seeing every tenant's
    documents, for oversight, instead of only their own.
    list_documents(None) already returns everything unfiltered (see
    document_repository.py) — this route only needed to decide *when*
    passing None is allowed. A non-admin passing all_tenants=true is
    denied outright rather than silently falling back to their own
    scope, so a caller relying on this flag finds out immediately if it
    didn't take effect.
    """
    tenant_id = getattr(request.state, "tenant_id", None)

    if all_tenants:
        permissions.check_permission(
            request,
            permissions.DOCUMENT_LIST_ALL_TENANTS,
            message="Listing documents across all tenants requires the admin role.",
        )
        tenant_id = None

    docs = list_documents(tenant_id, collection=collection)

    logger.info(
        "audit_event",
        extra={
            "extra_fields": {
                "event": "documents_listed",
                "path": request.url.path,
                "count": len(docs),
                "client": getattr(request.state, "client_name", "unknown"),
                "tenant_id": tenant_id,
                "all_tenants": all_tenants,
                "collection": collection,
            }
        },
    )

    return DocumentListResponse(documents=docs, total=len(docs))


def _ensure_document_accessible(request: Request, document_id: str) -> None:
    """Best-effort ownership check, mirroring delete_document's: only
    enforced when it can actually be verified (a requesting tenant *and*
    a known owner — no DB row means the document predates tenant
    tracking, so nothing to check against). 404, not 403 — a non-owner
    gets the same response as "doesn't exist" rather than a signal that
    confirms the document is real."""
    requester_tenant_id = getattr(request.state, "tenant_id", None)
    if requester_tenant_id is None:
        return
    owner_tenant_id = get_document_owner(document_id)
    if owner_tenant_id is not None and owner_tenant_id != requester_tenant_id:
        raise DocumentNotFoundError(f"No document found with id {document_id}")


@router.get("/documents/{document_id}/images", response_model=DocumentImagesResponse)
def list_document_images(document_id: str, request: Request) -> DocumentImagesResponse:
    """List the images extracted from a document (multi-modal RAG, Phase
    1): embedded figures plus full-page rasters of low-text pages. Reads
    the per-document manifest written at ingestion, so it never
    re-extracts the PDF. Each item's `url` is where its bytes can be
    fetched (GET /documents/{document_id}/images/{image_id})."""
    _ensure_document_accessible(request, document_id)

    images = load_image_manifest(document_id)
    items = [
        DocumentImageItem(
            image_id=image.image_id,
            document_id=image.document_id,
            page_number=image.page_number,
            content_type=image.content_type,
            mime_type=image.mime_type,
            width=image.width,
            height=image.height,
            byte_size=image.byte_size,
            url=f"/documents/{document_id}/images/{image.image_id}",
        )
        for image in images
    ]

    logger.info(
        "audit_event",
        extra={
            "extra_fields": {
                "event": "document_images_listed",
                "path": request.url.path,
                "document_id": document_id,
                "count": len(items),
                "client": getattr(request.state, "client_name", "unknown"),
            }
        },
    )

    return DocumentImagesResponse(document_id=document_id, total=len(items), images=items)


@router.get("/documents/{document_id}/images/{image_id}")
def get_document_image(document_id: str, image_id: str, request: Request) -> FileResponse:
    """Serve one extracted image's bytes. 404s for an unknown document,
    an unknown image_id, or an image whose bytes are missing from disk."""
    _ensure_document_accessible(request, document_id)

    image = next(
        (i for i in load_image_manifest(document_id) if i.image_id == image_id),
        None,
    )
    if image is None:
        raise DocumentNotFoundError(f"No image found with id {image_id}")

    path = image_storage_dir() / image.storage_path
    if not path.is_file():
        raise DocumentNotFoundError(f"No image found with id {image_id}")

    # No `filename` argument — that would force a Content-Disposition:
    # attachment download; omitting it keeps the response inline so a
    # browser can render it directly in an <img>.
    return FileResponse(path, media_type=image.mime_type)


@router.post(
    "/upload",
    response_model=DocumentProcessingResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    collection: str | None = Form(None),
) -> DocumentProcessingResponse:
    contents = await file.read()
    await file.seek(0)

    validate_pdf_upload(file, contents)

    # get_vector_store() is the same cached instance the /chat route uses
    # (see query.py), so a document processed here is immediately visible
    # to chat without a reload.
    #
    # The LLM client is only built when captioning is enabled: building it
    # requires a configured provider (GEMINI_API_KEY), and captioning is
    # off by default — constructing it unconditionally would make upload
    # fail on a deployment without an LLM key even though the base
    # pipeline never needs one.
    llm_client = get_llm_client() if settings.image_captioning_enabled else None
    service = DocumentProcessingService(
        get_vector_store(),
        llm_client=llm_client,
        image_vector_store=get_image_vector_store() if settings.clip_embedding_enabled else None,
    )
    tenant_id = getattr(request.state, "tenant_id", None)
    response = await service.process(file, tenant_id=tenant_id, collection=collection)

    # Invalidate semantic cache so new documents are immediately discoverable
    from app.services.cache_service import cache_service

    cache_service.invalidate()

    logger.info(
        "audit_event",
        extra={
            "extra_fields": {
                "event": "document_uploaded",
                "path": request.url.path,
                "document_id": response.document_id,
                "client": getattr(request.state, "client_name", "unknown"),
                "tenant_id": tenant_id,
            }
        },
    )

    return response


@router.post(
    "/upload/async",
    response_model=AsyncTaskResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def upload_document_async(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    collection: str | None = Form(None),
) -> AsyncTaskResponse:
    """Asynchronously process large PDF uploads in the background (Recommendation 2).

    Saves the file to disk, queues the ingestion pipeline via FastAPI BackgroundTasks,
    and returns a task_id immediately. Callers poll GET /documents/tasks/{task_id}
    to observe stage transitions (extracting -> chunking -> embedding -> indexing -> completed).
    """
    contents = await file.read()
    await file.seek(0)

    validate_pdf_upload(file, contents)

    uploaded = await save_uploaded_file(file)
    tenant_id = getattr(request.state, "tenant_id", None)
    task_service = get_task_service()

    task = task_service.create_task(
        original_filename=uploaded["original_filename"],
        document_id=uploaded["document_id"],
        stored_filename=uploaded["stored_filename"],
        file_size=uploaded["file_size"],
        tenant_id=tenant_id,
        collection=collection,
    )

    background_tasks.add_task(
        execute_background_ingestion,
        task_id=task.task_id,
        document_id=uploaded["document_id"],
        original_filename=uploaded["original_filename"],
        stored_filename=uploaded["stored_filename"],
        file_size=uploaded["file_size"],
        tenant_id=tenant_id,
        collection=collection,
    )

    logger.info(
        "audit_event",
        extra={
            "extra_fields": {
                "event": "document_async_upload_queued",
                "path": request.url.path,
                "task_id": task.task_id,
                "document_id": uploaded["document_id"],
                "tenant_id": tenant_id,
                "client": getattr(request.state, "client_name", "unknown"),
            }
        },
    )

    return AsyncTaskResponse(
        task_id=task.task_id,
        document_id=uploaded["document_id"],
        original_filename=uploaded["original_filename"],
        status="queued",
        progress=0.0,
        message="Document upload queued for background processing.",
        created_at=task.created_at,
    )


@router.get("/documents/tasks/{task_id}", response_model=TaskStatusResponse)
def get_task_status(task_id: str, request: Request) -> TaskStatusResponse:
    """Check status and progress of a background ingestion task.

    Enforces tenant isolation: tasks owned by another tenant return 404.
    """
    tenant_id = getattr(request.state, "tenant_id", None)
    task = get_task_service().get_task(task_id, tenant_id=tenant_id)
    if task is None:
        raise TaskNotFoundError(f"No task found with id {task_id}")

    return task.to_response()


@router.get("/documents/tasks", response_model=list[TaskStatusResponse])
def list_tasks(
    request: Request, status: str | None = None, limit: int = 50
) -> list[TaskStatusResponse]:
    """List recent background tasks for the requesting tenant."""
    tenant_id = getattr(request.state, "tenant_id", None)
    tasks = get_task_service().list_tasks(tenant_id=tenant_id, status=status, limit=limit)
    return [t.to_response() for t in tasks]


@router.delete("/documents/{document_id}", response_model=DocumentDeleteResponse)
def delete_document(
    document_id: str, request: Request, confirm: bool = False, approval_id: str | None = None
) -> DocumentDeleteResponse:
    if not confirm:
        raise ConfirmationRequiredError(
            "Deleting a document is irreversible. Retry with ?confirm=true to proceed."
        )

    # Human-approval gate: a deployment policy (off by default), separate
    # from the confirm check above (mistake-prevention) and the role check
    # below (access control) — mirrors web_search_requires_approval's exact
    # shape in rag_service.py.
    #
    # PHASE 5 SECURITY FIX: this gate used to accept a bare client-supplied
    # `approved=true` query param as proof of approval — meaning any caller
    # with valid API-key credentials could self-satisfy the gate without an
    # operator ever resolving the registered Approval via
    # POST /api/v1/approvals/{id}/resolve (see docs/MODULE10_FINAL_AUDIT.md
    # §8, Finding 2, and docs/PHASE5_FINAL_GAP_CLOSURE_REPORT.md for the
    # full before/after). The gate now requires the caller to supply the
    # approval_id from the pending Approval this endpoint registers below,
    # and verifies — against the real ApprovalStore, not a client claim —
    # that it exists, is for THIS document_id, is genuinely APPROVED (not
    # pending/rejected), and has not expired. `approved=true` alone no
    # longer satisfies this gate; this is an intentional, documented
    # contract change, not an oversight.
    if settings.document_delete_requires_approval:
        approval = approval_service.get_approval_store().get(approval_id) if approval_id else None
        approval_valid = (
            approval is not None
            and approval.action == approval_service.APPROVAL_ACTION_DOCUMENT_DELETE
            and approval.payload.get("document_id") == document_id
            and approval.status == approval_service.STATUS_APPROVED
        )
        if not approval_valid:
            logger.info(
                "audit_event",
                extra={
                    "extra_fields": {
                        "event": "document_delete_pending_approval",
                        "path": request.url.path,
                        "document_id": document_id,
                        "client": getattr(request.state, "client_name", "unknown"),
                        "supplied_approval_id": approval_id,
                        "supplied_approval_status": approval.status if approval else None,
                    }
                },
            )
            # Feature #5 — record the gated deletion as a pending approval
            # (only if this exact one isn't already tracked) so an operator
            # can grant/reject it via the approval queue. Re-registering on
            # every retry of an already-pending request would otherwise
            # spam the queue with duplicates for the same document.
            if approval is None:
                approval_service.get_approval_store().register(
                    action=approval_service.APPROVAL_ACTION_DOCUMENT_DELETE,
                    requested_by=getattr(request.state, "client_name", None),
                    payload={"document_id": document_id},
                )
            raise ApprovalRequiredError(
                "Deleting this document requires an operator-approved request. Retry with "
                "?approval_id=<id> once an operator has approved it via "
                "POST /api/v1/approvals/{approval_id}/resolve."
            )

    # Permission check (see app/core/permissions.py): only enforced when
    # the DB is enabled (role has nowhere to live otherwise —
    # request.state.role is None, not "member", when the DB is disabled,
    # matching tenant_id's own None-when-disabled convention below).
    # Permission check (see app/core/permissions.py):
    # Members and admins have DOCUMENT_DELETE permission. Non-DB deployment
    # degrades to "allowed" when role is None.
    permissions.check_permission(
        request,
        permissions.DOCUMENT_DELETE,
        message="Deleting documents requires appropriate permissions.",
        document_id=document_id,
    )

    # Ownership check: only enforced when we can actually verify it (a
    # requesting tenant *and* a known owner). An unknown owner (no DB row —
    # DB disabled, or uploaded before tenant tracking existed) isn't
    # treated as a mismatch: the vector store, not this row, is the source
    # of truth for whether the document exists at all. 404, not 403 — a
    # non-owner gets the same response as "doesn't exist" rather than a
    # signal that confirms the document is real.
    requester_tenant_id = getattr(request.state, "tenant_id", None)
    if requester_tenant_id is not None:
        owner_tenant_id = get_document_owner(document_id)
        if owner_tenant_id is not None and owner_tenant_id != requester_tenant_id:
            logger.warning(
                "audit_event",
                extra={
                    "extra_fields": {
                        "event": "document_delete_denied_wrong_tenant",
                        "path": request.url.path,
                        "document_id": document_id,
                        "client": getattr(request.state, "client_name", "unknown"),
                    }
                },
            )
            raise DocumentNotFoundError(f"No document found with id {document_id}")

    vector_store = get_vector_store()
    removed_count = vector_store.delete_document(document_id, tenant_id=requester_tenant_id)

    if removed_count == 0:
        raise DocumentNotFoundError(f"No document found with id {document_id}")

    vector_store.save()

    # Same removal in the image vector store (Phase 4): CLIP image vectors
    # key their metadata by document_id too, so the same delete applies.
    # Best-effort — an uninitialized image store (nothing ever embedded)
    # or a store that never saw this document is not an error.
    image_chunks_removed = 0
    if settings.clip_embedding_enabled:
        image_store = get_image_vector_store()
        try:
            image_chunks_removed = image_store.delete_document(
                document_id, tenant_id=requester_tenant_id
            )
            image_store.save()
        except VectorStoreNotFoundError:
            pass

    # Best-effort cleanup of the uploaded file on disk — the document is
    # already gone from the vector store regardless of whether this finds
    # anything, so a missing file here isn't an error.
    for path in UPLOAD_DIR.glob(f"{document_id}.*"):
        path.unlink(missing_ok=True)
        # So the deleted file doesn't resurrect on the next cold start's
        # s3_sync_service.download_all() (no-op when sync is disabled).
        s3_sync_service.delete_file(path.name, settings.upload_dir_name)

    # Same best-effort cleanup for extracted images (multi-modal RAG,
    # Phase 1) — filenames start with the document_id (see
    # document_processing_service._persist_images), so a prefix glob
    # finds exactly this document's images.
    for path in settings.data_dir(settings.image_storage_dir_name).glob(f"{document_id}_*"):
        path.unlink(missing_ok=True)

    # Best-effort removal of the durable metadata row (no-op if DB disabled).
    delete_document_metadata(document_id)

    # Invalidate semantic query cache
    from app.services.cache_service import cache_service

    cache_service.invalidate()

    logger.info(
        "audit_event",
        extra={
            "extra_fields": {
                "event": "document_deleted",
                "path": request.url.path,
                "document_id": document_id,
                "chunks_removed": removed_count,
                "image_chunks_removed": image_chunks_removed,
                "client": getattr(request.state, "client_name", "unknown"),
            }
        },
    )

    return DocumentDeleteResponse(
        document_id=document_id,
        chunks_removed=removed_count,
        status="deleted",
    )


@router.get("/documents/{document_id}/file")
def get_document_file(document_id: str, request: Request) -> Response:
    """Serve the original uploaded PDF bytes (Agent 4.1 — in-app PDF
    citation preview). Tenant-scoped via _ensure_document_accessible.

    The file is encrypted at rest (Module 10 gap-closure) -- decrypted
    to plaintext bytes in memory and served directly (no tempfile) since
    the decrypted content is only ever the HTTP response body here, not
    something another process needs to open by path.
    """
    _ensure_document_accessible(request, document_id)
    matches = list(UPLOAD_DIR.glob(f"{document_id}.*"))
    if not matches:
        raise DocumentNotFoundError(f"No document found with id {document_id}")
    plaintext = decrypt_upload_bytes(matches[0].read_bytes(), document_id=document_id)
    return Response(content=plaintext, media_type="application/pdf")


@router.get(
    "/documents/{document_id}/pages/{page_number}/highlight",
    response_model=HighlightResponse,
)
def get_page_highlight(
    document_id: str, page_number: int, request: Request, text: str
) -> HighlightResponse:
    """Find where `text` appears on one page of the source PDF (PyMuPDF
    search_for) so the frontend can draw a highlight over the cited
    passage. Rects are in PDF page coordinates — the client scales them
    from page_width/page_height to its rendered canvas."""
    _ensure_document_accessible(request, document_id)
    matches = list(UPLOAD_DIR.glob(f"{document_id}.*"))
    if not matches:
        raise DocumentNotFoundError(f"No document found with id {document_id}")
    import fitz

    # Module 10 gap-closure: the file is encrypted at rest -- decrypt to
    # in-memory bytes and open PyMuPDF directly from the stream (no
    # tempfile needed; fitz supports this natively).
    plaintext = decrypt_upload_bytes(matches[0].read_bytes(), document_id=document_id)
    document = fitz.open(stream=plaintext, filetype="pdf")
    try:
        if page_number < 1 or page_number > document.page_count:
            raise DocumentNotFoundError(f"No page {page_number} in document {document_id}")
        page = document[page_number - 1]
        rects = page.search_for(text)
        return HighlightResponse(
            page_number=page_number,
            page_width=page.rect.width,
            page_height=page.rect.height,
            rects=[[r.x0, r.y0, r.x1, r.y1] for r in rects],
        )
    finally:
        document.close()
