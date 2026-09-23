"""Vision-grounded QA on full-page images (Phase 3 of multi-modal RAG).

For pages that would otherwise OCR poorly — the same
Settings.ocr_min_chars_per_page decision point extraction uses — the
stored full-page raster is sent to Gemini directly instead of trusting a
thin or empty text layer. Config-gated behind Settings.vision_qa_enabled:
each attempt is a paid Gemini vision call, so it only fires when
retrieval came back weak/insufficient and the document actually has page
rasters persisted (produced at ingestion when image_extraction_enabled
is on). Any failure degrades to None and the caller falls through to its
normal weak-retrieval path (web search, etc.).
"""

import logging
from pathlib import Path

from app.core.config import settings
from app.models.document import RetrievedChunk
from app.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

_QA_PROMPT = (
    "Answer the user's question using ONLY what is visible in the "
    "provided page image(s) of the document. If the image(s) do not "
    "contain the answer, say so plainly. Ground every claim in the image."
)


def image_storage_dir() -> Path:
    """Directory where extracted image bytes are persisted (see
    document_processing_service._persist_images)."""
    return settings.data_dir(settings.image_storage_dir_name)


def _stored_page_images(document_id: str) -> list[Path]:
    """Full-page rasters (content_type="page") persisted for this
    document, ordered by page number. Persisted filenames follow the
    {document_id}_page_{n}.png shape (see
    document_processing_service._persist_images), so a glob on the
    document prefix finds exactly its page rasters."""
    return sorted(image_storage_dir().glob(f"{document_id}_page_*.png"))


def try_vision_qa(
    query: str,
    chunks: list[RetrievedChunk],
    llm_client: LLMClient,
) -> str | None:
    """Attempt a vision-grounded answer for query against the low-text
    pages of the most relevant document in chunks.

    Returns the answer text, or None (logged) when there's nothing to
    QA — no chunks, no stored page rasters for the top document, or the
    provider can't take images. Callers use None to fall through to
    their normal weak-retrieval path; they should only call this when
    Settings.vision_qa_enabled is true (each success is a paid call).
    """
    if not chunks:
        return None

    top_chunk = max(chunks, key=lambda c: c.score)
    document_id = top_chunk.document_id

    page_images = _stored_page_images(document_id)
    if not page_images:
        logger.info(
            "vision_qa_no_page_rasters",
            extra={"extra_fields": {"document_id": document_id}},
        )
        return None

    pages = page_images[: settings.vision_qa_max_pages]
    prompt = f"{_QA_PROMPT}\n\nDocument: {document_id}\nQuestion: {query}\n\nAnswer in plain text."

    for path in pages:
        try:
            image_bytes = path.read_bytes()
        except OSError as exc:
            logger.warning(
                "vision_qa_read_failed",
                extra={"extra_fields": {"path": str(path), "error": str(exc)}},
            )
            continue
        try:
            answer = llm_client.generate_with_image(prompt, image_bytes, "image/png")
        except Exception as exc:
            logger.warning(
                "vision_qa_failed",
                extra={
                    "extra_fields": {
                        "document_id": document_id,
                        "page_count": len(pages),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                },
            )
            continue
        answer = (answer or "").strip()
        if answer:
            logger.info(
                "vision_qa_completed",
                extra={
                    "extra_fields": {
                        "document_id": document_id,
                        "pages_sent": len(pages),
                    }
                },
            )
            return answer
    return None
