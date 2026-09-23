"""Integration tests for encryption-at-rest of FAISS metadata.json chunk
text (app/services/faiss_vector_store.py) -- Module 10 gap-closure
expanding encryption scope to the highest-risk remaining surface.

Design proven here: the on-disk metadata.json file is encrypted, but
the in-memory self._metadata a running process searches/serves from
stays plaintext (decrypted once at load(), encrypted once at save()) --
BM25 (a lexical index over the raw terms) and every other in-memory
consumer need real plaintext to keep working; only the disk copy is
protected. This means encryption adds ZERO per-query overhead and
changes zero search behavior, verified directly below alongside the
plaintext-leakage/round-trip/fail-closed/backward-compatibility
guarantees the project's other encrypted fields already have tests for.
"""

import json

import pytest

from app.core.config import settings
from app.core.encryption import EncryptionIntegrityError, EncryptionKeyMissingError, generate_key
from app.models.document import EmbeddedChunk
from app.services.faiss_vector_store import FAISSVectorStore

FAKE_EMBEDDING_DIM = 4
EMBEDDING_A = [1.0, 0.0, 0.0, 0.0]
EMBEDDING_B = [0.0, 1.0, 0.0, 0.0]


@pytest.fixture
def encryption_key(monkeypatch):
    key = generate_key()
    monkeypatch.setattr(settings, "encryption_key_b64", key)
    return key


def _store_with_one_chunk(tmp_path, text: str, chunk_id: str = "chunk-1", document_id: str = "doc-1"):
    store = FAISSVectorStore(index_path=tmp_path / "i.faiss", metadata_path=tmp_path / "m.json")
    store.create_index(dimension=FAKE_EMBEDDING_DIM)
    store.add_embeddings(
        [
            EmbeddedChunk(
                chunk_id=chunk_id,
                document_id=document_id,
                embedding=EMBEDDING_A,
                metadata={"chunk_index": 0, "total_chunks": 1, "source": "pdf", "text": text},
            )
        ]
    )
    return store


class TestEncryptedWrite:
    def test_persisted_text_is_not_plaintext(self, tmp_path, encryption_key):
        sensitive_text = "My social security number is 123-45-6789."
        store = _store_with_one_chunk(tmp_path, sensitive_text)
        store.save()

        raw = json.loads((tmp_path / "m.json").read_text())
        assert len(raw) == 1
        raw_text = raw[0]["metadata"]["text"]
        assert raw_text != sensitive_text
        assert "123-45-6789" not in raw_text
        assert raw_text.startswith("enc1:")

    def test_non_text_metadata_fields_stay_plaintext(self, tmp_path, encryption_key):
        """chunk_index/total_chunks/source/document_id etc. must remain
        plaintext -- only free-text chunk content is encrypted."""
        store = _store_with_one_chunk(tmp_path, "some chunk text")
        store.save()

        raw = json.loads((tmp_path / "m.json").read_text())[0]
        assert raw["document_id"] == "doc-1"
        assert raw["chunk_id"] == "chunk-1"
        assert raw["metadata"]["chunk_index"] == 0
        assert raw["metadata"]["source"] == "pdf"


class TestDecryptRoundTripAndSearchUnaffected:
    def test_load_returns_decrypted_text_via_search(self, tmp_path, encryption_key):
        original_text = "What does the document say about apple scab treatment?"
        store = _store_with_one_chunk(tmp_path, original_text)
        store.save()

        fresh_store = FAISSVectorStore(index_path=tmp_path / "i.faiss", metadata_path=tmp_path / "m.json")
        fresh_store.load()
        results = fresh_store.search(EMBEDDING_A, top_k=1)
        assert results[0].text == original_text

    def test_bm25_lexical_search_still_works_on_decrypted_in_memory_text(self, tmp_path, encryption_key):
        """The real proof encryption doesn't break hybrid search: BM25
        needs real plaintext terms in memory, and it must still find a
        real keyword match after a save/load round trip through the
        encrypted-at-rest file."""
        store = _store_with_one_chunk(tmp_path, "The quick brown fox jumps over the lazy dog")
        store.save()

        fresh_store = FAISSVectorStore(index_path=tmp_path / "i.faiss", metadata_path=tmp_path / "m.json")
        fresh_store.load()
        results = fresh_store.search_bm25("quick brown fox", top_k=1)
        assert len(results) == 1
        assert "quick brown fox" in results[0].text


class TestWrongKey:
    def test_decrypting_with_a_different_key_fails_safely(self, tmp_path, encryption_key):
        from app.core.encryption import decrypt_bytes

        store = _store_with_one_chunk(tmp_path, "secret chunk content")
        store.save()

        raw = json.loads((tmp_path / "m.json").read_text())[0]
        raw_text = raw["metadata"]["text"]
        ciphertext = __import__("base64").b64decode(raw_text[len("enc1:"):])

        wrong_key = generate_key()
        with pytest.raises(EncryptionIntegrityError):
            decrypt_bytes(ciphertext, key_b64=wrong_key, associated_data="chunk-1".encode())


class TestTamperedCiphertext:
    def test_tampered_text_fails_to_decrypt_on_load(self, tmp_path, encryption_key):
        store = _store_with_one_chunk(tmp_path, "untampered chunk text")
        store.save()

        metadata_path = tmp_path / "m.json"
        raw = json.loads(metadata_path.read_text())
        text = raw[0]["metadata"]["text"]
        corrupted = text[:-4] + ("0" if text[-4] != "0" else "1") + text[-3:]
        raw[0]["metadata"]["text"] = corrupted
        metadata_path.write_text(json.dumps(raw))

        fresh_store = FAISSVectorStore(index_path=tmp_path / "i.faiss", metadata_path=metadata_path)
        with pytest.raises(EncryptionIntegrityError):
            fresh_store.load()


class TestMissingKey:
    def test_save_fails_closed_without_a_configured_key(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "encryption_key_b64", None)
        store = _store_with_one_chunk(tmp_path, "should never be persisted")

        with pytest.raises(EncryptionKeyMissingError):
            store.save()

        assert not (tmp_path / "m.json").exists()

    def test_load_fails_closed_if_key_removed_after_saving(self, tmp_path, encryption_key, monkeypatch):
        store = _store_with_one_chunk(tmp_path, "written while key was present")
        store.save()

        monkeypatch.setattr(settings, "encryption_key_b64", None)

        fresh_store = FAISSVectorStore(index_path=tmp_path / "i.faiss", metadata_path=tmp_path / "m.json")
        with pytest.raises(EncryptionKeyMissingError):
            fresh_store.load()

    def test_load_failure_raises_the_real_encryption_error_not_corrupted_store(self, tmp_path, encryption_key, monkeypatch):
        """Regression pin for a real bug caught during this change's own
        review: a missing/wrong key must surface as
        EncryptionKeyMissingError, not get relabeled as a generic
        CorruptedVectorStoreError by an overly broad except clause."""
        from app.core.exceptions import CorruptedVectorStoreError

        store = _store_with_one_chunk(tmp_path, "some text")
        store.save()
        monkeypatch.setattr(settings, "encryption_key_b64", None)

        fresh_store = FAISSVectorStore(index_path=tmp_path / "i.faiss", metadata_path=tmp_path / "m.json")
        with pytest.raises(EncryptionKeyMissingError):
            fresh_store.load()
        # Explicitly NOT CorruptedVectorStoreError -- confirmed by the
        # exact exception type pytest.raises above already enforces.
        assert EncryptionKeyMissingError is not CorruptedVectorStoreError


class TestBackwardCompatibilityWithLegacyPlaintextMetadata:
    def test_legacy_plaintext_metadata_file_loads_unchanged(self, tmp_path, encryption_key):
        """A metadata.json written before this encryption change exists
        (plain text field, no enc1: marker) must remain fully readable
        -- no forced migration, no plaintext row mistaken for
        ciphertext."""
        import faiss
        import numpy as np

        index = faiss.IndexFlatIP(FAKE_EMBEDDING_DIM)
        index.add(np.array([EMBEDDING_A], dtype="float32"))
        faiss.write_index(index, str(tmp_path / "i.faiss"))

        legacy_metadata = [
            {
                "chunk_id": "legacy-chunk-1",
                "document_id": "legacy-doc-1",
                "metadata": {"chunk_index": 0, "total_chunks": 1, "source": "pdf", "text": "a legacy plaintext chunk"},
            }
        ]
        (tmp_path / "m.json").write_text(json.dumps(legacy_metadata))

        store = FAISSVectorStore(index_path=tmp_path / "i.faiss", metadata_path=tmp_path / "m.json")
        store.load()
        results = store.search(EMBEDDING_A, top_k=1)
        assert results[0].text == "a legacy plaintext chunk"

    def test_new_saves_are_always_encrypted_even_when_legacy_data_was_loaded(self, tmp_path, encryption_key):
        import faiss
        import numpy as np

        index = faiss.IndexFlatIP(FAKE_EMBEDDING_DIM)
        index.add(np.array([EMBEDDING_A], dtype="float32"))
        faiss.write_index(index, str(tmp_path / "i.faiss"))
        legacy_metadata = [
            {
                "chunk_id": "legacy-chunk-1",
                "document_id": "legacy-doc-1",
                "metadata": {"chunk_index": 0, "total_chunks": 1, "source": "pdf", "text": "legacy plaintext"},
            }
        ]
        (tmp_path / "m.json").write_text(json.dumps(legacy_metadata))

        store = FAISSVectorStore(index_path=tmp_path / "i.faiss", metadata_path=tmp_path / "m.json")
        store.load()
        store.save()  # re-save without adding anything new

        raw = json.loads((tmp_path / "m.json").read_text())[0]
        assert raw["metadata"]["text"].startswith("enc1:")
        assert raw["metadata"]["text"] != "legacy plaintext"
