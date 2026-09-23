"""Embeds extracted PDF figures in CLIP space and indexes them into the
image vector store (Phase 4 of multi-modal RAG — cross-modal retrieval).

Runs only when Settings.clip_embedding_enabled is true and images were
extracted (Settings.image_extraction_enabled). Each embedded figure
becomes an image-derived vector row in the *separate* image FAISS index
(image_index.faiss / image_metadata.json), tagged source="clip_image",
so hybrid search can fuse image similarity in as a third signal.

The retrieval object an image-store hit returns is a RetrievedChunk whose
text is the figure's caption when one was produced (image_captioning_enabled)
or a short placeholder otherwise — the caption text is what the answering
model cites from, so figures are best paired with captioning, but CLIP
embedding itself is independent of it.

A CLIP service failure (ClipServiceError after retries) degrades to "no
embedding for this image", never a failed upload — the same
degrade-don't-fail posture image captioning already uses.
"""

import logging
import time

from app.core.exceptions import ClipServiceError
from app.models.document import EmbeddedChunk, ExtractedImage
from app.services import clip_client
from app.services.image_captioning_service import image_storage_dir
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)


def _embed_one(
    image: ExtractedImage, *, document_id: str, tenant_id: int | None, caption: str | None
) -> EmbeddedChunk | None:
    """Embed one figure image via CLIP and return an image-derived
    EmbeddedChunk, or None on any failure (read error, CLIP unreachable,
    out-of-contract response) — logged, never raised."""
    start = time.perf_counter()
    try:
        image_bytes = (image_storage_dir() / image.storage_path).read_bytes()
    except OSError as exc:
        logger.warning(
            "clip_image_read_failed",
            extra={"extra_fields": {"image_id": image.image_id, "error": str(exc)}},
        )
        return None

    try:
        result = clip_client.embed_image(image_bytes, image.storage_path, image.mime_type)
    except ClipServiceError as exc:
        logger.warning(
            "clip_image_embed_skipped",
            extra={"extra_fields": {"image_id": image.image_id, "error": str(exc)}},
        )
        return None

    text = (
        caption.strip()
        if caption and caption.strip()
        else f"A figure on page {image.page_number} of the uploaded document."
    )
    logger.info(
        "clip_image_embedded",
        extra={
            "extra_fields": {
                "image_id": image.image_id,
                "page_number": image.page_number,
                "dimension": result.dimension,
                "processing_duration": round(time.perf_counter() - start, 4),
            }
        },
    )
    return EmbeddedChunk(
        chunk_id=image.image_id,
        document_id=document_id,
        embedding=result.embedding,
        metadata={
            "document_id": document_id,
            "image_id": image.image_id,
            "page_number": image.page_number,
            "source": "clip_image",
            "content_type": "image",
            "text": text,
            "tenant_id": tenant_id,
        },
    )


def embed_images_to_store(
    images: list[ExtractedImage],
    image_vector_store: VectorStore,
    *,
    document_id: str,
    tenant_id: int | None,
    captions_by_image_id: dict[str, str] | None = None,
) -> int:
    """Embed each figure image via CLIP and add the resulting vectors to
    the image vector store (creating the store's index on first use).
    Direction: image -> CLIP space, so query-time text embeddings can rank
    them. Returns the number of images actually embedded. Only
    content_type="figure" images are embedded — full-page rasters exist
    for vision QA, not for cross-modal figure retrieval.

    Degrades per-image: a failed embed is skipped and logged, never fatal
    to the upload. A wholly-empty batch is a valid no-op.
    """
    captions = captions_by_image_id or {}
    embedded: list[EmbeddedChunk] = []
    embedded_ids: set[str] = set()
    start = time.perf_counter()

    for image in images:
        if image.content_type != "figure":
            continue
        chunk = _embed_one(
            image,
            document_id=document_id,
            tenant_id=tenant_id,
            caption=captions.get(image.image_id),
        )
        if chunk is None or chunk.chunk_id in embedded_ids:
            continue
        embedded.append(chunk)
        embedded_ids.add(chunk.chunk_id)

    if not embedded:
        return 0

    try:
        image_vector_store.add_embeddings(embedded)
    except Exception:  # noqa: BLE001 — VectorStoreNotFoundError means no index yet
        image_vector_store.create_index(dimension=len(embedded[0].embedding))
        image_vector_store.add_embeddings(embedded)
    image_vector_store.save()

    logger.info(
        "clip_embedding_ingestion_completed",
        extra={
            "extra_fields": {
                "document_id": document_id,
                "images_embedded": len(embedded),
                "processing_duration": round(time.perf_counter() - start, 4),
            }
        },
    )
    return len(embedded)
