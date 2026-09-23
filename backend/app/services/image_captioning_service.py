"""Captions embedded PDF images via a vision-capable LLM and turns each
caption into an image-derived DocumentChunk for the existing text
pipeline (Phase 2 of multi-modal RAG).

Runs only when Settings.image_captioning_enabled is true, images were
extracted (Settings.image_extraction_enabled), and a vision-capable
LLMClient is available — and a caption failure or an unsupported client
degrades to "no caption chunk" for that image rather than failing or
slowing the upload (the same degrade-don't-fail posture OCR already
uses). The chunks this service produces are indistinguishable from
body-text chunks except for their metadata: source="image_caption",
content_type="image_caption", plus image_id/page_number for provenance,
so citations can point at "a figure on page N" instead of implying body
text.
"""

import json
import logging
import time
from pathlib import Path

from app.core.config import settings
from app.models.document import DocumentChunk, ExtractedImage
from app.services.chunking_service import chunk_text
from app.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

# Instructs the model to behave like a figure captioner, not a document
# summarizer — the caption becomes a searchable chunk, so it should
# describe *what is depicted* so that retrieval and the answering model
# can match it against questions that hinge on the image's content.
_CAPTION_PROMPT = (
    "Describe the content of this image in one or two concise sentences, "
    "as a caption for a figure in a document. State plainly what is "
    "visually depicted: objects, people, charts, labels, axes, and any "
    "visible text. Do not speculate about the surrounding document."
)


def image_storage_dir() -> Path:
    """Directory where extracted image bytes are persisted (see
    document_processing_service._persist_images)."""
    return settings.data_dir(settings.image_storage_dir_name)


def image_manifest_path(document_id: str) -> Path:
    """Where this document's extracted-image records live on disk. Named
    with the same {document_id}_ prefix as the image bytes so
    delete_document's prefix glob cleanup removes it alongside them."""
    return image_storage_dir() / f"{document_id}_images.json"


def write_image_manifest(images: list[ExtractedImage]) -> None:
    """Persist extracted-image metadata (bytes excluded) so the images can
    be listed without re-extracting the PDF. A write failure only loses
    the listing, never the images or the upload — logged, not raised, the
    same degrade-don't-fail posture the rest of this module uses."""
    if not images:
        return
    document_id = images[0].document_id
    try:
        path = image_manifest_path(document_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([image.model_dump() for image in images]))
    except OSError as exc:
        logger.warning(
            "image_manifest_write_failed",
            extra={"extra_fields": {"document_id": document_id, "error": str(exc)}},
        )


def load_image_manifest(document_id: str) -> list[ExtractedImage]:
    """Read back a document's extracted-image records, or [] when there's
    no manifest (no extracted images, or a document ingested before
    manifests existed). A corrupt/unreadable manifest degrades to [] with
    a log — listing is best-effort, never a 500."""
    path = image_manifest_path(document_id)
    try:
        payload = json.loads(path.read_text())
        return [ExtractedImage(**record) for record in payload]
    except FileNotFoundError:
        return []
    except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
        logger.warning(
            "image_manifest_read_failed",
            extra={"extra_fields": {"document_id": document_id, "error": str(exc)}},
        )
        return []


def _caption_one(image: ExtractedImage, llm_client: LLMClient) -> str | None:
    """Caption a single image, returning the caption text or None.

    Any failure — missing bytes on disk, an unsupported (non-vision)
    client, a provider error, an empty caption — is logged and degrades
    to None (no caption chunk for this image). Deliberately broad
    `except Exception`: the caller's contract is "captioning is an
    enhancement, never a failure mode for the upload," and this is the
    single chokepoint that guarantees it.
    """
    start = time.perf_counter()
    try:
        image_bytes = (image_storage_dir() / image.storage_path).read_bytes()
    except OSError as exc:
        logger.warning(
            "image_caption_read_failed",
            extra={"extra_fields": {"image_id": image.image_id, "error": str(exc)}},
        )
        return None

    prompt = (
        f"{_CAPTION_PROMPT}\n\nThis image is a figure on page "
        f"{image.page_number} of the uploaded document."
    )
    try:
        caption = llm_client.generate_with_image(prompt, image_bytes, image.mime_type)
    except Exception as exc:
        logger.warning(
            "image_caption_skipped",
            extra={
                "extra_fields": {
                    "image_id": image.image_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            },
        )
        return None

    caption = (caption or "").strip()
    if not caption:
        return None

    logger.info(
        "image_caption_completed",
        extra={
            "extra_fields": {
                "image_id": image.image_id,
                "page_number": image.page_number,
                "caption_chars": len(caption),
                "processing_duration": round(time.perf_counter() - start, 4),
            }
        },
    )
    return caption


def caption_images(
    images: list[ExtractedImage],
    llm_client: LLMClient,
    *,
    document_id: str,
    tenant_id: int | None,
    chunk_index_start: int = 0,
) -> list[DocumentChunk]:
    """Caption each figure image and return one chunk per successful
    caption, placed at chunk_index_start onward (so the caller can lay
    them after the body-text chunks). Only content_type="figure" images
    are captioned — full-page rasters (content_type="page") exist for
    vision QA, not for captioning. Failed captions are skipped, never
    fatal."""
    chunks: list[DocumentChunk] = []
    next_index = chunk_index_start
    captioned = 0
    start = time.perf_counter()

    for image in images:
        if image.content_type != "figure":
            continue
        caption = _caption_one(image, llm_client)
        if caption is None:
            continue
        captioned += 1
        caption_chunks = chunk_text(
            text=caption[: settings.image_caption_max_chars],
            document_id=document_id,
            tenant_id=tenant_id,
            chunk_index_start=next_index,
            source="image_caption",
            content_type="image_caption",
            page_number=image.page_number,
            image_id=image.image_id,
        )
        chunks.extend(caption_chunks)
        next_index += len(caption_chunks)

    logger.info(
        "image_captioning_completed",
        extra={
            "extra_fields": {
                "document_id": document_id,
                "images_seen": sum(1 for i in images if i.content_type == "figure"),
                "images_captioned": captioned,
                "chunks_produced": len(chunks),
                "processing_duration": round(time.perf_counter() - start, 4),
            }
        },
    )
    return chunks
