"""Extracts text from previously uploaded PDF files using PyMuPDF (fitz).

Pages whose own text falls short of Settings.ocr_min_chars_per_page — not
just totally blank pages — fall back to OCR via pytesseract, which shells
out to the tesseract binary. That binary is a system dependency, not a
pip package (see backend/Dockerfile), and it's genuinely optional: if
it's missing, OCR is skipped and whatever thin native text a page had is
kept as-is, logged once as a warning. Ingestion must never crash over a
missing optional dependency.
"""

import logging
import time
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any

import fitz
import pytesseract
from PIL import Image

from app.core.config import settings
from app.core.exceptions import CorruptedPDFError, DocumentNotFoundError, TextExtractionError

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _tesseract_available() -> bool:
    """Whether the tesseract binary is installed and callable.

    Cached, including a negative result — unlike a model load (see
    embedding_service.get_embedding_model), a missing system binary won't
    become available mid-process, so there's no reason to re-probe on
    every OCR'd page or every document.

    Deliberately `except Exception`, not `except TesseractNotFoundError`:
    a present-but-broken tesseract install (missing shared libraries, a
    bad PATH, wrong architecture) raises other exception types from the
    subprocess call, and those need to degrade the same way a missing
    binary does — don't narrow this without re-checking that case.
    """
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def is_tesseract_available() -> bool:
    """Public probe of whether OCR (the tesseract binary) is installed.

    Same cached probe as _tesseract_available — surfaces on /health so an
    operator can see, at a glance, that scanned/low-text PDFs will be
    ingested without OCR.
    """
    return _tesseract_available()


def _ocr_page(page: fitz.Page) -> str:
    """Rasterize a page at Settings.ocr_dpi and OCR it. Raises on failure
    — callers decide how to degrade (see extract_text_from_pdf)."""
    pixmap = page.get_pixmap(dpi=settings.ocr_dpi)
    image = Image.open(BytesIO(pixmap.tobytes("png")))
    # pytesseract ships no type stubs, so image_to_string() resolves to Any.
    # At runtime it's always a str here: we never pass output_type, and its
    # default (Output.STRING) is what returns plain text rather than the
    # dict/DataFrame shapes other output_type values produce.
    return str(pytesseract.image_to_string(image))


def extract_text_from_pdf(document_id: str, file_path: Path) -> dict[str, Any]:
    """Extract text from every page of a PDF, in order.

    Returns a dict with extracted_text, total_pages, extracted_characters,
    and pages_ocred. A page whose own text falls short of
    Settings.ocr_min_chars_per_page — not just a totally blank page — is
    tried via OCR (rasterize + pytesseract); real scans routinely carry a
    thin native text layer (a header, a page number, a few garbled
    characters from a bad prior OCR pass) that would otherwise pass as
    "has text" and ship a near-useless chunk to the index. If OCR isn't
    available or also comes up empty, whatever thin native text existed
    is kept rather than discarded — it's better than nothing even if it
    didn't clear the trust threshold on its own.
    """
    if not file_path.is_file():
        raise DocumentNotFoundError(f"No such file: {file_path}")

    start = time.perf_counter()

    try:
        document = fitz.open(file_path)
    except fitz.FileDataError as exc:
        raise CorruptedPDFError(f"File is not a valid PDF: {file_path.name}") from exc

    try:
        page_texts = []
        pages_ocred = 0
        ocr_unavailable_warned = False
        low_text_pages = []

        for page in document:
            try:
                page_text = page.get_text()
            except Exception as exc:
                raise TextExtractionError(
                    f"Failed to extract text from page {page.number + 1} of {file_path.name}"
                ) from exc

            native_text = page_text.strip() if page_text else ""

            if len(native_text) >= settings.ocr_min_chars_per_page:
                page_texts.append(page_text)
                continue

            # Below the trust threshold — either truly blank or carrying
            # only a thin native text layer. Remember the page for
            # vision QA (Phase 3), then try OCR; degrade to the thin
            # native text (if any) rather than losing it outright.
            low_text_pages.append(page.number + 1)
            if not _tesseract_available():
                if not ocr_unavailable_warned:
                    logger.warning(
                        "ocr_unavailable",
                        extra={"extra_fields": {"document_id": document_id}},
                    )
                    ocr_unavailable_warned = True
                if native_text:
                    page_texts.append(page_text)
                continue

            try:
                ocr_text = _ocr_page(page)
            except Exception as exc:
                # A single page's OCR failing (corrupt image data, etc.)
                # shouldn't take down the whole extraction — fall back to
                # this page's thin native text (if any) and move on.
                logger.warning(
                    "ocr_page_failed",
                    extra={
                        "extra_fields": {
                            "document_id": document_id,
                            "page_number": page.number + 1,
                            "error": str(exc),
                        }
                    },
                )
                if native_text:
                    page_texts.append(page_text)
                continue

            if ocr_text and ocr_text.strip():
                page_texts.append(ocr_text)
                pages_ocred += 1
            elif native_text:
                # OCR ran but found nothing usable either — the thin
                # native text is still the best material available.
                page_texts.append(page_text)

        total_pages = document.page_count
    finally:
        document.close()

    extracted_text = "\n".join(page_texts)
    extracted_characters = len(extracted_text)
    duration_seconds = time.perf_counter() - start

    logger.info(
        "pdf_text_extracted",
        extra={
            "extra_fields": {
                "document_id": document_id,
                "total_pages": total_pages,
                "extracted_characters": extracted_characters,
                "pages_ocred": pages_ocred,
                "duration_seconds": round(duration_seconds, 4),
            }
        },
    )

    return {
        "extracted_text": extracted_text,
        "total_pages": total_pages,
        "extracted_characters": extracted_characters,
        "pages_ocred": pages_ocred,
        "low_text_pages": low_text_pages,
    }


_IMAGE_MIME_BY_EXT = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "gif": "image/gif",
    "bmp": "image/bmp",
    "webp": "image/webp",
    "tiff": "image/tiff",
    "tif": "image/tiff",
    "jp2": "image/jp2",
    "jpx": "image/jp2",
}


def _mime_for_ext(ext: str) -> str:
    return _IMAGE_MIME_BY_EXT.get((ext or "").lower(), "image/png")


def _rasterize_page(page: fitz.Page) -> tuple[bytes, int, int]:
    """Render a full page to PNG bytes at Settings.ocr_dpi, returning the
    (bytes, pixel_width, pixel_height) triple. The image sent to Gemini by
    vision QA for pages that OCR poorly (see extract_images_from_pdf)."""
    pixmap = page.get_pixmap(dpi=settings.ocr_dpi)
    return pixmap.tobytes("png"), pixmap.width, pixmap.height


def extract_images_from_pdf(document_id: str, file_path: Path) -> list[dict[str, Any]]:
    """Extract the images embedded in a PDF (Phase 1 of multi-modal RAG).

    Returns a list of dicts, each describing one image record:

    - image_id: stable id, unique within the document (e.g. "<doc>_img_<xref>")
    - document_id, page_number (1-based), content_type ("figure" | "page")
    - mime_type, width, height, byte_size
    - data: the raw image bytes, ready to be persisted by the caller

    Two kinds of records are produced, both gated behind
    Settings.image_extraction_enabled by the *caller* (extract_images_
    from_pdf itself always runs when asked):

    1. Embedded images: page.get_images() + doc.extract_image(), deduped
       by xref so the same figure reused across pages yields one record
       (first occurrence's page). Images that aren't actually rendered on
       the page (get_image_rects is empty), and images whose smaller side
       falls below Settings.image_min_side_px (icons, dividers, noise),
       are skipped rather than captioned or QA'd.
    2. Full-page rasters for low-text pages (content_type="page"): the
       same ocr_min_chars_per_page decision point OCR uses — a page whose
       own text falls below the threshold is rasterized at Settings.ocr_dpi
       so vision QA can read it as an image instead of trusting a thin or
       empty text layer.

    Extraction is capped at Settings.image_max_count_per_document records
    (a pathological PDF of thousands of vector tiles must not balloon
    ingestion). A per-page or per-xref failure is logged and skipped —
    never fatal, the same degrade-don't-fail posture OCR already uses.
    """
    if not file_path.is_file():
        raise DocumentNotFoundError(f"No such file: {file_path}")

    try:
        document = fitz.open(file_path)
    except fitz.FileDataError as exc:
        raise CorruptedPDFError(f"File is not a valid PDF: {file_path.name}") from exc

    images: list[dict[str, Any]] = []
    seen_xrefs: set[int] = set()

    try:
        for page in document:
            if len(images) >= settings.image_max_count_per_document:
                break
            try:
                page_image_infos = page.get_images(full=True)
            except Exception as exc:
                logger.warning(
                    "image_extraction_page_failed",
                    extra={
                        "extra_fields": {
                            "document_id": document_id,
                            "page_number": page.number + 1,
                            "error": str(exc),
                        }
                    },
                )
                continue

            for image_info in page_image_infos:
                if len(images) >= settings.image_max_count_per_document:
                    break
                xref = image_info[0]
                if xref in seen_xrefs:
                    continue
                # A referenced-but-never-drawn image (get_image_rects
                # empty) isn't content — skip it.
                try:
                    rects = page.get_image_rects(xref)
                except Exception:
                    rects = []
                if not rects:
                    continue

                try:
                    info = document.extract_image(xref)
                except Exception as exc:
                    logger.warning(
                        "image_extract_xref_failed",
                        extra={
                            "extra_fields": {
                                "document_id": document_id,
                                "xref": xref,
                                "error": str(exc),
                            }
                        },
                    )
                    continue

                width, height = int(info.get("width") or 0), int(info.get("height") or 0)
                if width <= 0 or height <= 0:
                    continue
                if min(width, height) < settings.image_min_side_px:
                    continue

                seen_xrefs.add(xref)
                data = info.get("image", b"")
                images.append(
                    {
                        "image_id": f"{document_id}_img_{xref}",
                        "document_id": document_id,
                        "page_number": page.number + 1,
                        "content_type": "figure",
                        "mime_type": _mime_for_ext(info.get("ext", "png")),
                        "width": width,
                        "height": height,
                        "byte_size": len(data),
                        "data": data,
                    }
                )

        # Low-text pages (the same threshold OCR uses) get full-page
        # rasters tagged content_type="page", so vision QA can answer from
        # the page image directly when OCR'd text proved too thin.
        for page in document:
            if len(images) >= settings.image_max_count_per_document:
                break
            try:
                native_text = page.get_text().strip()
            except Exception:
                native_text = ""
            if len(native_text) >= settings.ocr_min_chars_per_page:
                continue
            try:
                data, width, height = _rasterize_page(page)
            except Exception as exc:
                logger.warning(
                    "page_raster_failed",
                    extra={
                        "extra_fields": {
                            "document_id": document_id,
                            "page_number": page.number + 1,
                            "error": str(exc),
                        }
                    },
                )
                continue
            images.append(
                {
                    "image_id": f"{document_id}_page_{page.number + 1}",
                    "document_id": document_id,
                    "page_number": page.number + 1,
                    "content_type": "page",
                    "mime_type": "image/png",
                    "width": width,
                    "height": height,
                    "byte_size": len(data),
                    "data": data,
                }
            )
    finally:
        document.close()

    logger.info(
        "pdf_images_extracted",
        extra={
            "extra_fields": {
                "document_id": document_id,
                "image_count": len(images),
            }
        },
    )
    return images
