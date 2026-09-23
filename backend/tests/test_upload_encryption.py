"""Integration tests for encryption-at-rest of uploaded document files
(app/services/upload_service.py) -- Module 10 gap-closure expanding
encryption scope beyond ChatTurn.content/ChatSession.title/feedback
comments to the raw uploaded PDF bytes themselves.

Uses a REAL PDF built with PyMuPDF (not a fake byte string) so the
round-trip test proves genuine extraction still works against the
decrypted tempfile, not just that bytes come back equal.
"""

import asyncio
import io
from pathlib import Path

import pytest

from app.core.config import settings
from app.core.encryption import EncryptionIntegrityError, EncryptionKeyMissingError, generate_key
from app.services.upload_service import (
    decrypt_upload_bytes,
    decrypted_upload_tempfile,
    encrypt_upload_bytes,
    save_uploaded_file,
)


def _make_real_pdf_bytes(text: str = "Hello from a real test PDF.") -> bytes:
    """Builds a genuine, parseable single-page PDF via PyMuPDF -- not a
    fake magic-bytes-only string -- so decrypt round-trip tests can
    prove real extraction still works."""
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


class _FakeUploadFile:
    """Minimal stand-in for fastapi.UploadFile -- only the attributes/
    methods save_uploaded_file actually uses."""

    def __init__(self, filename: str, content: bytes):
        self.filename = filename
        self._content = content

    async def read(self) -> bytes:
        return self._content


@pytest.fixture
def upload_dir(tmp_path, monkeypatch):
    import app.services.upload_service as upload_service_module

    monkeypatch.setattr(upload_service_module, "UPLOAD_DIR", tmp_path)
    monkeypatch.setattr("app.services.s3_sync_service.upload_file", lambda *a, **k: None)
    return tmp_path


@pytest.fixture
def encryption_key(monkeypatch):
    key = generate_key()
    monkeypatch.setattr(settings, "encryption_key_b64", key)
    return key


class TestEncryptUploadBytesRoundTrip:
    def test_encrypt_then_decrypt_returns_original(self, encryption_key):
        original = b"some raw file bytes, could be anything"
        encrypted = encrypt_upload_bytes(original, document_id="doc-1")
        assert encrypted != original
        assert original not in encrypted

        decrypted = decrypt_upload_bytes(encrypted, document_id="doc-1")
        assert decrypted == original

    def test_wrong_document_id_as_aad_fails_to_decrypt(self, encryption_key):
        encrypted = encrypt_upload_bytes(b"doc A's content", document_id="doc-a")
        with pytest.raises(EncryptionIntegrityError):
            decrypt_upload_bytes(encrypted, document_id="doc-b")

    def test_legacy_plaintext_bytes_pass_through_unchanged(self, encryption_key):
        """A file uploaded before this encryption change exists (real
        PDF magic bytes, no AGRIENC1 marker) must remain readable, not
        mistaken for ciphertext."""
        legacy_pdf_bytes = b"%PDF-1.4\nlegacy plaintext content"
        assert decrypt_upload_bytes(legacy_pdf_bytes, document_id="doc-legacy") == legacy_pdf_bytes


class TestSaveUploadedFileEncryptsOnDisk:
    @pytest.mark.asyncio
    async def test_persisted_file_is_not_plaintext(self, upload_dir, encryption_key):
        real_pdf = _make_real_pdf_bytes("Sensitive uploaded content.")
        result = await save_uploaded_file(_FakeUploadFile("doc.pdf", real_pdf))

        stored_path = upload_dir / result["stored_filename"]
        raw = stored_path.read_bytes()
        assert raw != real_pdf
        assert b"Sensitive uploaded content" not in raw
        assert not raw.startswith(b"%PDF-")  # real PDFs always start with this; ciphertext must not

    @pytest.mark.asyncio
    async def test_file_size_in_response_is_the_original_plaintext_size(self, upload_dir, encryption_key):
        """The user-facing file_size must reflect what they uploaded, not
        the encrypted-on-disk size (which is slightly larger)."""
        real_pdf = _make_real_pdf_bytes()
        result = await save_uploaded_file(_FakeUploadFile("doc.pdf", real_pdf))
        assert result["file_size"] == len(real_pdf)


class TestDecryptedUploadTempfileRealExtraction:
    @pytest.mark.asyncio
    async def test_real_pymupdf_extraction_works_on_the_decrypted_tempfile(self, upload_dir, encryption_key):
        """The critical end-to-end proof: a real PDF, saved encrypted,
        decrypted back to a tempfile, and genuinely re-opened/parsed by
        PyMuPDF -- not just a byte-equality check."""
        import fitz

        real_pdf = _make_real_pdf_bytes("Unique marker TEXT_9f3e7c21 for extraction proof.")
        result = await save_uploaded_file(_FakeUploadFile("doc.pdf", real_pdf))
        stored_path = upload_dir / result["stored_filename"]

        with decrypted_upload_tempfile(result["document_id"], stored_path) as tmp_path:
            assert tmp_path.exists()
            doc = fitz.open(tmp_path)
            try:
                page_text = doc[0].get_text()
            finally:
                doc.close()
            assert "TEXT_9f3e7c21" in page_text

        # tempfile must be cleaned up after the context manager exits
        assert not tmp_path.exists()

    @pytest.mark.asyncio
    async def test_tempfile_cleaned_up_even_when_caller_raises(self, upload_dir, encryption_key):
        real_pdf = _make_real_pdf_bytes()
        result = await save_uploaded_file(_FakeUploadFile("doc.pdf", real_pdf))
        stored_path = upload_dir / result["stored_filename"]

        captured_path: Path | None = None
        with pytest.raises(ValueError, match="boom"):
            with decrypted_upload_tempfile(result["document_id"], stored_path) as tmp_path:
                captured_path = tmp_path
                raise ValueError("boom")

        assert captured_path is not None
        assert not captured_path.exists()


class TestMissingKey:
    @pytest.mark.asyncio
    async def test_write_fails_closed_without_a_configured_key(self, upload_dir, monkeypatch):
        monkeypatch.setattr(settings, "encryption_key_b64", None)
        real_pdf = _make_real_pdf_bytes()

        with pytest.raises(EncryptionKeyMissingError):
            await save_uploaded_file(_FakeUploadFile("doc.pdf", real_pdf))


class TestTamperedCiphertext:
    def test_tampered_upload_fails_to_decrypt(self, encryption_key):
        encrypted = bytearray(encrypt_upload_bytes(b"original content", document_id="doc-x"))
        # Flip a byte deep in the ciphertext (past the magic marker + nonce).
        encrypted[-4] ^= 0xFF
        with pytest.raises(EncryptionIntegrityError):
            decrypt_upload_bytes(bytes(encrypted), document_id="doc-x")
