"""Application-level encryption at rest for the most security-sensitive
persisted payloads (Module 10 gap-closure: encryption at rest was
genuinely missing before this module — see docs/CHECKLIST.md's
Encryption row and docs/NOT_APPLICABLE.md).

Uses AES-256-GCM (via `cryptography`'s `AESGCM`, an authenticated cipher
-- tampering is detected, not just undetected-and-decrypted) with a key
supplied through `Settings.encryption_key_b64` (base64-encoded 32 bytes),
read from the environment/`.env`, never hardcoded and never committed.

Scope, stated honestly: this module provides a real, tested encrypt/
decrypt primitive for byte payloads. It is NOT yet wired into the
document-upload pipeline (`document_processing_service.py`) or the FAISS
index/metadata files — doing so is a larger, riskier change (a migration
story for already-uploaded plaintext documents, a decision about where
the boundary between "encrypted at rest" and "needs to be readable by
PyMuPDF/FAISS without a full decrypt-to-tempfile step" lives) that
deserves its own dedicated pass, not bundled into this one. This is
APPLICATION-LEVEL encryption, distinct from PLATFORM-LEVEL encryption
(e.g. an encrypted EBS volume) — see this module's own `EncryptionStatus`
docstring for the exact boundary.
"""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

_KEY_ENV_VAR = "ENCRYPTION_KEY_B64"
_NONCE_SIZE = 12  # 96-bit nonce, the standard/recommended size for AES-GCM


class EncryptionKeyMissingError(RuntimeError):
    """Raised when an encrypt/decrypt call is attempted with no key
    configured. Never silently falls back to storing plaintext."""


class EncryptionIntegrityError(RuntimeError):
    """Raised when ciphertext fails authentication (wrong key, or the
    ciphertext was tampered with). AES-GCM detects this cryptographically
    -- it does not merely fail to produce readable plaintext."""


def generate_key() -> str:
    """Generate a fresh, random 256-bit key, base64-encoded for storage
    in an environment variable. Never logged; caller is responsible for
    placing it in `.env`/a secret manager, not in tracked files."""
    return base64.b64encode(AESGCM.generate_key(bit_length=256)).decode("ascii")


def _load_key(key_b64: str | None) -> bytes:
    raw = key_b64 if key_b64 is not None else os.environ.get(_KEY_ENV_VAR)
    if not raw:
        raise EncryptionKeyMissingError(
            f"No encryption key configured (set {_KEY_ENV_VAR} in the environment/.env). "
            "Refusing to persist plaintext under the guise of an encrypted path."
        )
    try:
        key = base64.b64decode(raw)
    except Exception as exc:  # noqa: BLE001
        raise EncryptionKeyMissingError(f"{_KEY_ENV_VAR} is not valid base64.") from exc
    if len(key) != 32:
        raise EncryptionKeyMissingError(f"{_KEY_ENV_VAR} must decode to exactly 32 bytes (AES-256), got {len(key)}.")
    return key


@dataclass(frozen=True)
class EncryptedPayload:
    """Self-contained ciphertext: nonce + AES-GCM ciphertext+tag, both
    required to decrypt. `to_bytes`/`from_bytes` give a single on-disk
    blob format (nonce prefix + ciphertext) so callers don't need to
    manage the nonce separately."""

    nonce: bytes
    ciphertext: bytes  # AESGCM output already includes the 16-byte auth tag

    def to_bytes(self) -> bytes:
        return self.nonce + self.ciphertext

    @classmethod
    def from_bytes(cls, blob: bytes) -> EncryptedPayload:
        if len(blob) < _NONCE_SIZE:
            raise EncryptionIntegrityError("Ciphertext blob is shorter than the nonce size -- not a valid payload.")
        return cls(nonce=blob[:_NONCE_SIZE], ciphertext=blob[_NONCE_SIZE:])


def encrypt_bytes(plaintext: bytes, *, key_b64: str | None = None, associated_data: bytes | None = None) -> bytes:
    """Encrypt `plaintext`, returning a single blob (nonce + ciphertext)
    ready to write to disk. `associated_data` is authenticated but not
    encrypted -- e.g. a document_id, so a ciphertext can't be silently
    swapped onto a different record without detection."""
    key = _load_key(key_b64)
    nonce = os.urandom(_NONCE_SIZE)
    aesgcm = AESGCM(key)
    ciphertext = aesgcm.encrypt(nonce, plaintext, associated_data)
    return EncryptedPayload(nonce=nonce, ciphertext=ciphertext).to_bytes()


def decrypt_bytes(blob: bytes, *, key_b64: str | None = None, associated_data: bytes | None = None) -> bytes:
    """Decrypt a blob produced by `encrypt_bytes`. Raises
    EncryptionIntegrityError (never returns garbage) if the key is wrong
    or the ciphertext was tampered with -- AES-GCM's authentication tag
    check fails closed."""
    key = _load_key(key_b64)
    payload = EncryptedPayload.from_bytes(blob)
    aesgcm = AESGCM(key)
    try:
        return aesgcm.decrypt(payload.nonce, payload.ciphertext, associated_data)
    except InvalidTag as exc:
        raise EncryptionIntegrityError(
            "Ciphertext failed authentication -- wrong key or tampered/corrupted data."
        ) from exc


# Versioned marker for a text column value produced by encrypt_text_field,
# shared by every call site that stores an encrypted string in an
# otherwise-plaintext text column (e.g. PostgresSessionStore's
# ChatTurn.content, session_repository's ChatSession.title). Using one
# shared prefix/helper -- rather than each call site reimplementing its
# own encode/marker/decode logic -- means there is exactly one place that
# defines "what does an encrypted text column value look like," and every
# caller gets the same backward-compatibility (unmarked = legacy
# plaintext, never mistaken for ciphertext) and fail-closed behavior for
# free.
TEXT_FIELD_ENC_PREFIX = "enc1:"


def encrypt_text_field(plaintext: str, *, associated_data: str, key_b64: str | None = None) -> str:
    """Encrypt a plain `str` for storage in a text column, returning a
    single self-contained string (marker + base64 ciphertext) safe to
    write directly into that column. `associated_data` is typically a
    stable identifier (e.g. session_id) binding this ciphertext to its
    owning row, authenticated but not encrypted -- see encrypt_bytes."""
    ciphertext = encrypt_bytes(
        plaintext.encode("utf-8"), key_b64=key_b64, associated_data=associated_data.encode("utf-8")
    )
    return TEXT_FIELD_ENC_PREFIX + base64.b64encode(ciphertext).decode("ascii")


def decrypt_text_field(stored: str, *, associated_data: str, key_b64: str | None = None) -> str:
    """Decrypt a value produced by encrypt_text_field. A value without
    the TEXT_FIELD_ENC_PREFIX marker is a legacy plaintext row (written
    before encryption existed for this column) and is returned as-is --
    no migration is required for existing rows to remain readable, and a
    plaintext row is never mistaken for ciphertext. A marked value that
    fails to decrypt (wrong key, tampered ciphertext, missing key) raises
    rather than returning anything -- fails closed."""
    if not stored.startswith(TEXT_FIELD_ENC_PREFIX):
        return stored
    ciphertext = base64.b64decode(stored[len(TEXT_FIELD_ENC_PREFIX) :])
    plaintext = decrypt_bytes(ciphertext, key_b64=key_b64, associated_data=associated_data.encode("utf-8"))
    return plaintext.decode("utf-8")
