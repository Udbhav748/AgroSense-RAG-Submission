"""Integration tests for encryption-at-rest of feedback comments
(app/services/feedback_service.py) -- Module 10 gap-closure expanding
encryption scope beyond ChatTurn.content/ChatSession.title. `comment`
is real, free-text, human-typed content; `rating`/`rubric`/
`reviewer_id`/`timestamp` stay plaintext since eval/metrics_report.py
aggregates them directly and they are not free-text.

Mirrors tests/test_postgres_session_store_encryption.py's methodology:
real file I/O through the actual service functions (record_feedback/
list_feedback), not a mock of the storage layer.
"""

import base64
import json

import pytest

from app.core.config import settings
from app.core.encryption import EncryptionIntegrityError, EncryptionKeyMissingError, generate_key
from app.services import feedback_service


@pytest.fixture
def feedback_path(tmp_path, monkeypatch):
    path = tmp_path / "feedback.jsonl"
    monkeypatch.setattr(feedback_service, "FEEDBACK_DIR", tmp_path)
    monkeypatch.setattr(feedback_service, "FEEDBACK_PATH", path)
    monkeypatch.setattr("app.services.s3_sync_service.upload_file", lambda *a, **k: None)
    return path


@pytest.fixture
def encryption_key(monkeypatch):
    key = generate_key()
    monkeypatch.setattr(settings, "encryption_key_b64", key)
    return key


def _raw_events(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class TestEncryptedWrite:
    def test_persisted_comment_is_not_plaintext(self, feedback_path, encryption_key):
        sensitive = "My email is user@example.com, please don't share this."
        feedback_service.record_feedback("msg-1", "up", sensitive, reviewer_id="r1")

        raw = _raw_events(feedback_path)
        assert len(raw) == 1
        assert raw[0]["comment"] != sensitive
        assert "user@example.com" not in raw[0]["comment"]
        assert raw[0]["comment"].startswith("enc1:")

    def test_non_comment_fields_stay_plaintext(self, feedback_path, encryption_key):
        """rating/rubric/reviewer_id/message_id must remain plaintext --
        eval/metrics_report.py aggregates them directly from this file."""
        rubric = {
            "correctness": 5, "helpfulness": 4, "completeness": 5, "safety": 5,
            "tone": 4, "groundedness": 5, "citation_quality": 3,
        }
        feedback_service.record_feedback("msg-2", "up", "a comment", reviewer_id="r1", rubric=rubric)

        raw = _raw_events(feedback_path)[0]
        assert raw["rating"] == "up"
        assert raw["reviewer_id"] == "r1"
        assert raw["message_id"] == "msg-2"
        assert raw["rubric"] == rubric


class TestDecryptRoundTrip:
    def test_list_feedback_returns_decrypted_comment(self, feedback_path, encryption_key):
        original = "The answer was accurate and well cited."
        feedback_service.record_feedback("msg-3", "up", original, reviewer_id="r1")

        events = feedback_service.list_feedback()
        assert events[0]["comment"] == original

    def test_null_comment_stays_null(self, feedback_path, encryption_key):
        feedback_service.record_feedback("msg-4", "down", None, reviewer_id="r1")

        raw = _raw_events(feedback_path)[0]
        assert raw["comment"] is None
        events = feedback_service.list_feedback()
        assert events[0]["comment"] is None

    def test_multiple_events_round_trip_independently(self, feedback_path, encryption_key):
        feedback_service.record_feedback("msg-5", "up", "first comment", reviewer_id="r1")
        feedback_service.record_feedback("msg-6", "down", "second comment", reviewer_id="r1")

        events = {e["message_id"]: e["comment"] for e in feedback_service.list_feedback()}
        assert events["msg-5"] == "first comment"
        assert events["msg-6"] == "second comment"


class TestWrongKey:
    def test_decrypting_with_a_different_key_fails_safely(self, feedback_path, encryption_key):
        from app.core.encryption import decrypt_bytes

        feedback_service.record_feedback("msg-7", "up", "secret comment", reviewer_id="r1")
        raw = _raw_events(feedback_path)[0]
        ciphertext = base64.b64decode(raw["comment"][len("enc1:"):])

        wrong_key = generate_key()
        with pytest.raises(EncryptionIntegrityError):
            decrypt_bytes(ciphertext, key_b64=wrong_key, associated_data="msg-7".encode())


class TestTamperedCiphertext:
    def test_tampered_comment_fails_to_decrypt(self, feedback_path, encryption_key):
        feedback_service.record_feedback("msg-8", "up", "untampered comment", reviewer_id="r1")

        lines = feedback_path.read_text(encoding="utf-8").splitlines()
        event = json.loads(lines[0])
        corrupted = event["comment"][:-4] + ("0" if event["comment"][-4] != "0" else "1") + event["comment"][-3:]
        event["comment"] = corrupted
        feedback_path.write_text(json.dumps(event) + "\n", encoding="utf-8")

        with pytest.raises(EncryptionIntegrityError):
            feedback_service.list_feedback()


class TestMissingKey:
    def test_write_fails_closed_without_a_configured_key(self, feedback_path, monkeypatch):
        monkeypatch.setattr(settings, "encryption_key_b64", None)

        with pytest.raises(EncryptionKeyMissingError):
            feedback_service.record_feedback("msg-9", "up", "should never be persisted", reviewer_id="r1")

        assert not feedback_path.exists()

    def test_comment_free_feedback_never_requires_a_key(self, feedback_path, monkeypatch):
        """A rating with no comment must not require an encryption key at
        all -- encryption only activates when there's actually free-text
        content to protect."""
        monkeypatch.setattr(settings, "encryption_key_b64", None)

        feedback_service.record_feedback("msg-10", "up", None, reviewer_id="r1")

        raw = _raw_events(feedback_path)[0]
        assert raw["comment"] is None

    def test_read_fails_closed_if_key_removed_after_writing(self, feedback_path, encryption_key, monkeypatch):
        feedback_service.record_feedback("msg-11", "up", "written while key was present", reviewer_id="r1")

        monkeypatch.setattr(settings, "encryption_key_b64", None)

        with pytest.raises(EncryptionKeyMissingError):
            feedback_service.list_feedback()


class TestBackwardCompatibilityWithLegacyPlaintextRows:
    def test_legacy_plaintext_comment_reads_back_unchanged(self, feedback_path, encryption_key):
        legacy_event = {
            "timestamp": "2026-01-01T00:00:00+00:00",
            "message_id": "legacy-1",
            "rating": "up",
            "comment": "a legacy plaintext comment",
            "reviewer_id": "r1",
            "rubric": None,
        }
        feedback_path.write_text(json.dumps(legacy_event) + "\n", encoding="utf-8")

        events = feedback_service.list_feedback()
        assert events[0]["comment"] == "a legacy plaintext comment"

    def test_new_writes_are_always_encrypted_even_when_legacy_rows_exist(self, feedback_path, encryption_key):
        legacy_event = {
            "timestamp": "2026-01-01T00:00:00+00:00",
            "message_id": "legacy-2",
            "rating": "up",
            "comment": "legacy plaintext",
            "reviewer_id": "r1",
            "rubric": None,
        }
        feedback_path.write_text(json.dumps(legacy_event) + "\n", encoding="utf-8")

        feedback_service.record_feedback("new-1", "down", "new comment, must be encrypted", reviewer_id="r1")

        raw = _raw_events(feedback_path)
        assert raw[0]["comment"] == "legacy plaintext"
        assert raw[1]["comment"].startswith("enc1:")

        events = {e["message_id"]: e["comment"] for e in feedback_service.list_feedback()}
        assert events["legacy-2"] == "legacy plaintext"
        assert events["new-1"] == "new comment, must be encrypted"
