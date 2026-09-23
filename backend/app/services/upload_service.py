"""Handles saving uploaded files to disk using UUID-based filenames.

ENCRYPTION AT REST (Module 10 gap-closure): the raw uploaded file is the
user's actual document content (potentially private) persisted verbatim
to disk. It is encrypted with app.core.encryption (AES-256-GCM) before
being written — see encrypt_upload_bytes/decrypt_upload_bytes below and
their own docstrings for the on-disk marker/backward-compatibility
scheme. document_id is bound as associated data, mirroring the
session_id binding already used for ChatTurn.content/ChatSession.title.
"""

import contextlib
import logging
import tempfile
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import UploadFile

from app.core.config import settings
from app.core.encryption import decrypt_bytes, encrypt_bytes
from app.core.exceptions import UnsupportedFileTypeError
from app.services import s3_sync_service

UPLOAD_DIR = settings.data_dir(settings.upload_dir_name)

logger = logging.getLogger(__name__)

# Distinct from a real PDF's own magic bytes ('%PDF-') or any other
# supported upload format, so a legacy plaintext file (written before
# this change) is never ambiguous with an encrypted one -- the same
# "unmarked = legacy plaintext, never mistaken for ciphertext" contract
# app.core.encryption's text-field helpers already use, adapted for
# binary files (which can't use a string prefix the way a text column
# can).
_ENC_MAGIC = b"AGRIENC1"


def encrypt_upload_bytes(plaintext: bytes, *, document_id: str) -> bytes:
    """Encrypt raw uploaded-file bytes for storage, returning a single
    blob (magic marker + AES-256-GCM ciphertext) safe to write directly
    to disk."""
    ciphertext = encrypt_bytes(
        plaintext, key_b64=settings.encryption_key_b64, associated_data=document_id.encode("utf-8")
    )
    return _ENC_MAGIC + ciphertext


def decrypt_upload_bytes(stored: bytes, *, document_id: str) -> bytes:
    """Decrypt bytes produced by encrypt_upload_bytes. A blob without the
    magic marker is a legacy plaintext file (uploaded before this
    change) and is returned as-is -- no migration is required for
    already-uploaded files to remain readable. A marked blob that fails
    to decrypt (wrong key, tampered file, missing key) raises rather
    than returning anything -- fails closed."""
    if not stored.startswith(_ENC_MAGIC):
        return stored
    ciphertext = stored[len(_ENC_MAGIC) :]
    return decrypt_bytes(ciphertext, key_b64=settings.encryption_key_b64, associated_data=document_id.encode("utf-8"))


@contextlib.contextmanager
def decrypted_upload_tempfile(document_id: str, stored_path: Path) -> Iterator[Path]:
    """Decrypt an on-disk uploaded file to a short-lived plaintext
    tempfile, yielding its path for callers (PyMuPDF etc.) that need a
    real file path rather than in-memory bytes -- always deleted on
    exit, even if the caller raises. Keeps the decrypt/tempfile-lifetime
    concern in one place (upload_service) rather than duplicated across
    every extraction call site.
    """
    plaintext = decrypt_upload_bytes(stored_path.read_bytes(), document_id=document_id)
    suffix = stored_path.suffix or ".pdf"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        tmp.write(plaintext)
        tmp.close()
        yield Path(tmp.name)
    finally:
        Path(tmp.name).unlink(missing_ok=True)


async def save_uploaded_file(file: UploadFile) -> dict[str, Any]:
    """Save an uploaded file to UPLOAD_DIR under a UUID-based filename,
    encrypted at rest.

    Returns a dict of the fields needed to build a DocumentUploadResponse.
    """
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    if not file.filename:
        # UploadFile.filename is `str | None`: starlette leaves it None
        # when the multipart part's Content-Disposition carries no
        # filename parameter at all (validate_pdf_upload doesn't check
        # this — it only looks at content_type/magic bytes/size). Without
        # this guard, Path(None) below raises an unhandled TypeError
        # (-> 500) instead of the same clean 415 a bad content_type gets.
        raise UnsupportedFileTypeError("Uploaded file is missing a filename.")

    document_id = str(uuid.uuid4())
    extension = Path(file.filename).suffix
    stored_filename = f"{document_id}{extension}"
    destination = UPLOAD_DIR / stored_filename

    contents = await file.read()
    destination.write_bytes(encrypt_upload_bytes(contents, document_id=document_id))
    s3_sync_service.upload_file(destination, settings.upload_dir_name)

    logger.debug(
        "file_written",
        extra={
            "extra_fields": {
                "stored_filename": stored_filename,
                "path": str(destination),
                "file_size": len(contents),
            }
        },
    )

    return {
        "document_id": document_id,
        "original_filename": file.filename,
        "stored_filename": stored_filename,
        "file_size": len(contents),
        "upload_timestamp": datetime.now(UTC),
        "status": "uploaded",
    }
