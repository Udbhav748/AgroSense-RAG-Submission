"""Bulk-ingest a local directory of PDFs directly through the document
processing pipeline, bypassing the HTTP /upload endpoint entirely.

Why this exists: /upload runs behind the frontend's 60s axios timeout
(and whatever patience the server/proxy in front of it has). OCR alone
runs roughly a second or more per page at Settings.ocr_dpi
(document_service.py) — a single 60-page scanned document can blow past
that timeout on its own, well before a real multi-document corpus would.
This script drives DocumentProcessingService directly, one file at a
time, from a plain Python process with no HTTP layer or request timeout
in the way.

It's also the repeatable way to rebuild the index from a known set of
source PDFs — e.g. after a chunking/embedding config change, or once
documents carry shared/private metadata tags — without re-uploading
everything by hand through the UI.

Each file goes through the exact same validation
(validation_service.validate_pdf_upload) and pipeline
(DocumentProcessingService.process) the HTTP route uses — no pipeline
logic is duplicated here, just a local file wrapped as a FastAPI
UploadFile instead of one arriving over multipart/form-data. Uploaded
files are copied into backend/uploads/ under a UUID name either way (see
upload_service.py), so ingesting a large corpus roughly doubles its disk
footprint — expected, not a bug.

One file failing (corrupted PDF, oversized, etc.) is logged and skipped;
it doesn't stop the rest of the corpus from being ingested.

Usage (from backend/):
    python scripts/ingest_corpus.py path/to/corpus_dir
    python scripts/ingest_corpus.py path/to/corpus_dir --pattern "*.pdf" --no-recursive
"""

from __future__ import annotations

import argparse
import asyncio
import io
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import UploadFile  # noqa: E402
from starlette.datastructures import Headers  # noqa: E402

from app.core.exceptions import AppError, VectorStoreNotFoundError  # noqa: E402
from app.core.logging import configure_logging  # noqa: E402
from app.services.document_processing_service import DocumentProcessingService  # noqa: E402
from app.services.faiss_vector_store import FAISSVectorStore  # noqa: E402
from app.services.validation_service import validate_pdf_upload  # noqa: E402


def _load_vector_store(index_path: Path | None, metadata_path: Path | None) -> FAISSVectorStore:
    """Same load-or-empty pattern as app/api/v1/routes/query.py's
    get_vector_store() — an empty, uninitialized store is fine;
    DocumentProcessingService creates it on the first embedding batch.

    index_path/metadata_path default to None, which FAISSVectorStore
    resolves to backend/vector_store/ — the same index the running app
    uses. Pass both explicitly to build a separate index instead (a
    scratch corpus, a dry run, or a distinct index while trying out
    shared/private metadata tagging) without touching the live one.
    """
    store = FAISSVectorStore(index_path=index_path, metadata_path=metadata_path)
    try:
        store.load()
    except VectorStoreNotFoundError:
        pass
    return store


def _upload_file_from_path(path: Path) -> UploadFile:
    """Wrap a local file as a FastAPI UploadFile so it can go through
    validate_pdf_upload() and DocumentProcessingService.process()
    unmodified — the same objects the HTTP route builds from a real
    multipart request, just sourced from disk instead."""
    data = path.read_bytes()
    return UploadFile(
        file=io.BytesIO(data),
        filename=path.name,
        size=len(data),
        headers=Headers({"content-type": "application/pdf"}),
    )


async def ingest_file(service: DocumentProcessingService, path: Path) -> dict:
    upload_file = _upload_file_from_path(path)
    contents = await upload_file.read()
    await upload_file.seek(0)
    validate_pdf_upload(upload_file, contents)
    response = await service.process(upload_file)
    return response.model_dump()


async def run(
    corpus_dir: Path,
    pattern: str,
    recursive: bool,
    index_path: Path | None,
    metadata_path: Path | None,
) -> None:
    configure_logging()
    # A bulk operator wants clean per-file progress plus real warnings
    # (ocr_unavailable, ocr_page_failed, ...), not the DEBUG-level HTTP
    # wire chatter Settings.debug=true also turns on for local dev — this
    # script always runs at INFO regardless of that .env setting.
    logging.getLogger().setLevel(logging.INFO)

    paths = sorted(corpus_dir.rglob(pattern) if recursive else corpus_dir.glob(pattern))
    if not paths:
        raise SystemExit(f"No files matching {pattern!r} found under {corpus_dir}")

    vector_store = _load_vector_store(index_path, metadata_path)
    service = DocumentProcessingService(vector_store)

    scope = "recursive" if recursive else "top-level only"
    print(f"Ingesting {len(paths)} file(s) from {corpus_dir} ({scope})...")
    print(f"Index: {vector_store.index_path}\n")

    succeeded = 0
    failed = 0
    total_pages_ocred = 0
    start = time.perf_counter()

    for index, path in enumerate(paths, start=1):
        file_start = time.perf_counter()
        try:
            result = await ingest_file(service, path)
        except AppError as exc:
            failed += 1
            print(f"[{index}/{len(paths)}] FAILED  {path.name}: {exc.detail}")
            continue
        except Exception as exc:  # noqa: BLE001 - one bad file must not kill the batch
            failed += 1
            print(f"[{index}/{len(paths)}] FAILED  {path.name}: {exc}")
            continue

        succeeded += 1
        total_pages_ocred += result["pages_ocred"]
        duration = time.perf_counter() - file_start
        ocr_note = f", {result['pages_ocred']} page(s) OCR'd" if result["pages_ocred"] else ""
        print(
            f"[{index}/{len(paths)}] OK      {path.name} — "
            f"{result['total_pages']} pages, {result['total_chunks']} chunks{ocr_note} "
            f"({duration:.1f}s)"
        )

    total_duration = time.perf_counter() - start
    print(
        f"\nDone in {total_duration:.1f}s — {succeeded} succeeded, {failed} failed, "
        f"{total_pages_ocred} total page(s) recovered via OCR."
    )
    print(f"Index now holds {vector_store.total_vectors()} vectors.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("corpus_dir", type=Path, help="Directory of PDFs to ingest.")
    parser.add_argument(
        "--pattern", default="*.pdf", help="Glob pattern for files to ingest (default: *.pdf)."
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Only scan the top level of corpus_dir, not subdirectories.",
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        default=None,
        help="Write to a FAISS index at this path instead of the app's default "
        "(backend/vector_store/index.faiss) — e.g. to build a separate corpus "
        "index without touching the live one. Must be paired with --metadata-path.",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=None,
        help="Metadata JSON path to go with --index-path.",
    )
    args = parser.parse_args()

    if not args.corpus_dir.is_dir():
        raise SystemExit(f"Not a directory: {args.corpus_dir}")

    if bool(args.index_path) != bool(args.metadata_path):
        raise SystemExit("--index-path and --metadata-path must be given together.")

    asyncio.run(
        run(
            args.corpus_dir,
            args.pattern,
            recursive=not args.no_recursive,
            index_path=args.index_path,
            metadata_path=args.metadata_path,
        )
    )


if __name__ == "__main__":
    main()
