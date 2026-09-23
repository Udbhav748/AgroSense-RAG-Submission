"""Tests for app/core/encryption.py -- the Module 10 encryption-at-rest
primitive. Fully offline/deterministic, no external services.
"""

import pytest

from app.core.encryption import (
    EncryptionIntegrityError,
    EncryptionKeyMissingError,
    decrypt_bytes,
    encrypt_bytes,
    generate_key,
)


def test_generate_key_produces_32_byte_key():
    import base64

    key = generate_key()
    assert len(base64.b64decode(key)) == 32


def test_encrypt_then_decrypt_round_trip():
    key = generate_key()
    plaintext = b"this is a sensitive uploaded document's raw bytes"
    ciphertext = encrypt_bytes(plaintext, key_b64=key)
    assert ciphertext != plaintext
    assert decrypt_bytes(ciphertext, key_b64=key) == plaintext


def test_ciphertext_is_not_plaintext_and_varies_by_nonce():
    key = generate_key()
    plaintext = b"same input twice"
    a = encrypt_bytes(plaintext, key_b64=key)
    b = encrypt_bytes(plaintext, key_b64=key)
    assert a != b  # random nonce each call -- no two ciphertexts identical
    assert decrypt_bytes(a, key_b64=key) == plaintext
    assert decrypt_bytes(b, key_b64=key) == plaintext


def test_wrong_key_fails_closed():
    key_a = generate_key()
    key_b = generate_key()
    ciphertext = encrypt_bytes(b"secret", key_b64=key_a)
    with pytest.raises(EncryptionIntegrityError):
        decrypt_bytes(ciphertext, key_b64=key_b)


def test_tampered_ciphertext_fails_closed():
    key = generate_key()
    ciphertext = bytearray(encrypt_bytes(b"secret payload", key_b64=key))
    ciphertext[-1] ^= 0xFF  # flip the last byte -- inside the auth tag
    with pytest.raises(EncryptionIntegrityError):
        decrypt_bytes(bytes(ciphertext), key_b64=key)


def test_missing_key_raises_explicit_error_not_silent_plaintext(monkeypatch):
    monkeypatch.delenv("ENCRYPTION_KEY_B64", raising=False)
    with pytest.raises(EncryptionKeyMissingError):
        encrypt_bytes(b"data")
    with pytest.raises(EncryptionKeyMissingError):
        decrypt_bytes(b"\x00" * 28)


def test_malformed_key_raises_explicit_error():
    with pytest.raises(EncryptionKeyMissingError):
        encrypt_bytes(b"data", key_b64="not-valid-base64!!!")
    with pytest.raises(EncryptionKeyMissingError):
        encrypt_bytes(b"data", key_b64="dG9vc2hvcnQ=")  # valid base64, wrong length


def test_associated_data_binds_ciphertext_to_context():
    """A ciphertext encrypted with associated_data=b'doc-1' must not
    decrypt successfully under associated_data=b'doc-2' -- prevents a
    ciphertext blob from being silently swapped onto a different record."""
    key = generate_key()
    ciphertext = encrypt_bytes(b"payload", key_b64=key, associated_data=b"doc-1")
    assert decrypt_bytes(ciphertext, key_b64=key, associated_data=b"doc-1") == b"payload"
    with pytest.raises(EncryptionIntegrityError):
        decrypt_bytes(ciphertext, key_b64=key, associated_data=b"doc-2")


def test_truncated_blob_is_rejected():
    key = generate_key()
    with pytest.raises(EncryptionIntegrityError):
        decrypt_bytes(b"short", key_b64=key)
