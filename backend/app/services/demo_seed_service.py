"""Seeds the vector store with a bundled demo document on startup, if the
store is empty.

Exists specifically for ephemeral-filesystem deployments (e.g. Render's
free tier, which wipes local disk on every spin-down/cold start, not just
on redeploy) so the live demo always has something to query against
without requiring a real upload first. A no-op whenever the store already
has vectors — local dev with a persisted docker-compose volume, or any
platform with real persistent storage, never gets touched by this.

Mirrors scripts/ingest_corpus.py's "wrap a local file as an UploadFile"
approach (same validate_pdf_upload + DocumentProcessingService.process
pipeline the HTTP route uses) rather than importing that script directly —
it's written as a CLI entrypoint (argparse, sys.path manipulation), not a
reusable module.
"""

import io
import logging
from pathlib import Path

from fastapi import UploadFile
from starlette.datastructures import Headers

from app.core.exceptions import AppError
from app.services.document_processing_service import DocumentProcessingService
from app.services.validation_service import validate_pdf_upload
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)

DEMO_CORPUS_DIR = Path(__file__).resolve().parents[2] / "demo_corpus"


def _upload_file_from_path(path: Path) -> UploadFile:
    data = path.read_bytes()
    return UploadFile(
        file=io.BytesIO(data),
        filename=path.name,
        size=len(data),
        headers=Headers({"content-type": "application/pdf"}),
    )


async def seed_if_empty(vector_store: VectorStore) -> None:
    if vector_store.total_vectors() > 0:
        return

    pdf_paths = sorted(DEMO_CORPUS_DIR.glob("*.pdf"))
    if not pdf_paths:
        logger.warning(
            "demo_seed_skipped",
            extra={
                "extra_fields": {"reason": "no bundled PDFs found", "dir": str(DEMO_CORPUS_DIR)}
            },
        )
        return

    service = DocumentProcessingService(vector_store)
    for pdf_path in pdf_paths:
        upload_file = _upload_file_from_path(pdf_path)
        contents = await upload_file.read()
        await upload_file.seek(0)
        try:
            validate_pdf_upload(upload_file, contents)
            response = await service.process(upload_file)
        except AppError as exc:
            # One bad bundled file shouldn't crash startup — log and move
            # on, matching ingest_corpus.py's per-file error handling.
            logger.warning(
                "demo_seed_file_failed",
                extra={"extra_fields": {"file": pdf_path.name, "error": str(exc)}},
            )
            continue

        logger.info(
            "demo_seed_completed",
            extra={
                "extra_fields": {
                    "file": pdf_path.name,
                    "document_id": response.document_id,
                    "total_chunks": response.total_chunks,
                }
            },
        )
