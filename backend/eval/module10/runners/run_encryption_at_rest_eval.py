#!/usr/bin/env python
"""Module 10 gap-closure: encryption-at-rest coverage evidence.

Runs real encrypt/decrypt/tamper/wrong-key/missing-key/plaintext-leakage
checks against the REAL application code (PostgresSessionStore,
session_repository, app.core.encryption) over a real SQLite-backed
SQLAlchemy session -- not a description of what the code is supposed to
do. Also records the real pytest pass count for the encryption test
files, so no number here is asserted without being executed in this
same run.

Usage (from backend/):
    python eval/module10/runners/run_encryption_at_rest_eval.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from eval.module10 import config  # noqa: E402

BACKEND_DIR = Path(__file__).resolve().parents[3]


def _run_encryption_checks() -> dict:
    """Real, in-process demonstration of every property claimed in the
    coverage matrix below, against the real application code (not a
    reimplementation) over a real SQLite-backed SQLAlchemy session."""
    import base64

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from app.core.config import settings
    from app.core.database import Base
    from app.core.encryption import (
        EncryptionIntegrityError,
        EncryptionKeyMissingError,
        generate_key,
    )
    from app.models.db_models import ChatSession, ChatTurn  # noqa: F401
    from app.services import postgres_session_store as pss_module
    from app.services import session_repository as repo_module
    from app.services.postgres_session_store import PostgresSessionStore
    from app.services.session_repository import list_sessions, set_session_title_if_unset

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_local = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    pss_module.SessionLocal = session_local
    repo_module.SessionLocal = session_local
    repo_module.db_enabled = lambda: True

    key = generate_key()
    settings.encryption_key_b64 = key
    results: dict[str, bool] = {}

    store = PostgresSessionStore()
    session_id = store.create_session(tenant_id=1)
    marker = "TEST_SENSITIVE_VALUE_a1b2c3d4"
    store.append_turn(session_id, "user", f"content mentioning {marker}")
    set_session_title_if_unset(session_id, f"title mentioning {marker}")

    with session_local() as db:
        raw_session = db.get(ChatSession, session_id)
        raw_content = raw_session.turns[0].content
        raw_title = raw_session.title

    results["chat_turn_content_encrypted_on_disk"] = raw_content.startswith("enc1:") and marker not in raw_content
    results["chat_session_title_encrypted_on_disk"] = raw_title.startswith("enc1:") and marker not in raw_title

    history = store.get_history(session_id)
    results["content_round_trip_correct"] = history[0]["content"] == f"content mentioning {marker}"

    sessions = list_sessions(tenant_id=1)
    results["title_round_trip_correct"] = sessions[0]["title"] == f"title mentioning {marker}"

    # Wrong key
    from app.core.encryption import decrypt_bytes

    ciphertext = base64.b64decode(raw_content[len("enc1:") :])
    try:
        decrypt_bytes(ciphertext, key_b64=generate_key(), associated_data=session_id.encode())
        results["wrong_key_rejected"] = False
    except EncryptionIntegrityError:
        results["wrong_key_rejected"] = True

    # Tampered ciphertext
    with session_local() as db:
        session = db.get(ChatSession, session_id)
        turn = session.turns[0]
        turn.content = turn.content[:-4] + ("0" if turn.content[-4] != "0" else "1") + turn.content[-3:]
        db.commit()
    try:
        store.get_history(session_id)
        results["tampered_ciphertext_rejected"] = False
    except EncryptionIntegrityError:
        results["tampered_ciphertext_rejected"] = True

    # Missing key -- write
    settings.encryption_key_b64 = None
    session_id_2 = store.create_session(tenant_id=1)
    try:
        store.append_turn(session_id_2, "user", "should never be persisted")
        results["missing_key_write_fails_closed"] = False
    except EncryptionKeyMissingError:
        results["missing_key_write_fails_closed"] = True
    settings.encryption_key_b64 = key

    # Cross-session AAD isolation
    session_a = store.create_session(tenant_id=1)
    session_b = store.create_session(tenant_id=1)
    store.append_turn(session_a, "user", "session A private content")
    with session_local() as db:
        raw_a = db.get(ChatSession, session_a).turns[0].content
    ciphertext_a = base64.b64decode(raw_a[len("enc1:") :])
    try:
        decrypt_bytes(ciphertext_a, key_b64=key, associated_data=session_b.encode())
        results["cross_session_aad_isolation"] = False
    except EncryptionIntegrityError:
        results["cross_session_aad_isolation"] = True

    # Legacy plaintext backward compatibility
    with session_local() as db:
        session = db.get(ChatSession, session_a)
        from datetime import UTC, datetime

        session.turns.append(ChatTurn(role="user", content="legacy plaintext row", created_at=datetime.now(UTC)))
        db.commit()
    history_a = store.get_history(session_a)
    results["legacy_plaintext_backward_compatible"] = "legacy plaintext row" in [t["content"] for t in history_a]

    return results


def _run_pytest_files(paths: list[str]) -> dict:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *paths],
        cwd=BACKEND_DIR,
        capture_output=True,
        text=True,
        timeout=120,
    )
    tail = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    return {"exit_code": proc.returncode, "summary_line": tail}


def main() -> None:
    checks = _run_encryption_checks()
    pytest_result = _run_pytest_files(
        [
            "tests/test_encryption.py",
            "tests/test_postgres_session_store_encryption.py",
            "tests/test_session_repository_encryption.py",
        ]
    )

    coverage_matrix = [
        {"data_category": "ChatTurn.content (chat message text)", "storage": "PostgreSQL, chat_turns.content", "at_rest_protection": "AES-256-GCM (app.core.encryption)", "status": "protected"},
        {"data_category": "ChatSession.title (derived from first user message)", "storage": "PostgreSQL, chat_sessions.title", "at_rest_protection": "AES-256-GCM (app.core.encryption)", "status": "protected"},
        {"data_category": "Tenant/User/ApiKey metadata (slug, email, key_hash, role)", "storage": "PostgreSQL", "at_rest_protection": "not application-level encrypted", "status": "unprotected_by_design", "reason": "Not free-text user content; email is queried by unique index (equality lookup on login) which transparent field encryption would break without a deterministic/blind-index scheme not currently implemented; password/API-key are already one-way hashed, not reversible plaintext to begin with."},
        {"data_category": "Uploaded document raw file (PDF bytes)", "storage": "Local filesystem, settings.upload_dir_name, UUID filename", "at_rest_protection": "not application-level encrypted", "status": "unprotected_by_design", "reason": "PyMuPDF (document_service.py) opens the file directly from disk by path for extraction; transparent encryption here requires a decrypt-to-tempfile-before-parsing step and a migration story for already-uploaded files -- a larger, riskier change explicitly deferred (see app/core/encryption.py's own module docstring, written at the time the primitive was first introduced)."},
        {"data_category": "Extracted chunk text + metadata (FAISS metadata.json)", "storage": "Local filesystem, vector_store/metadata.json", "at_rest_protection": "not application-level encrypted", "status": "unprotected_by_design", "reason": "Read directly by retrieval/reranking/prompt-building code on every request; encrypting requires decrypting every chunk on every retrieval call and touches many call sites across retrieval_service/reranking_service/prompt_builder -- deferred as a distinct, larger pass rather than bundled into this one, per this task's own proportionality constraint."},
        {"data_category": "FAISS vector index (index.faiss)", "storage": "Local filesystem, vector_store/index.faiss", "at_rest_protection": "not application-level encrypted", "status": "unprotected_by_design", "reason": "Binary embedding vectors, not reversible to exact source text; encrypting would make FAISS's own similarity search impossible without a full decrypt-to-memory step on every query, defeating the point of an on-disk index. Documented as a limitation, not solved."},
        {"data_category": "Document metadata (filename, page/chunk counts, collection tag)", "storage": "PostgreSQL, documents table", "at_rest_protection": "not application-level encrypted", "status": "unprotected_by_design", "reason": "Filenames/counts, not free-text content; low sensitivity relative to chat content."},
        {"data_category": "Feedback events (rating, optional free-text comment, rubric scores)", "storage": "Local filesystem, feedback.jsonl", "at_rest_protection": "not application-level encrypted", "status": "unprotected_by_design", "reason": "Read in plaintext by eval/metrics_report.py to compute Acceptance Rate and Inter-Annotator Agreement across many records at once -- an evaluation artifact, not primary user-content storage, and encrypting would require decrypting on every aggregation read for no corresponding sensitivity gain over the already-encrypted chat content itself."},
        {"data_category": "Application logs (structured JSON)", "storage": "stdout / captured log files", "at_rest_protection": "N/A -- content excluded by design", "status": "not_applicable", "reason": "Settings.log_prompt_content is off by default; logs record query_length, not query content, unless explicitly enabled for debugging. Already documented in docs/ARCHITECTURE.md."},
        {"data_category": "Usage logs (path, method, status, latency)", "storage": "PostgreSQL, usage_logs table", "at_rest_protection": "N/A -- no free-text user content", "status": "not_applicable", "reason": "No message/query content stored in this table by design."},
        {"data_category": "Public demo/agricultural-pathology corpus", "storage": "backend/data (or equivalent), FAISS index", "at_rest_protection": "not encrypted", "status": "not_required", "reason": "Public reference material, not private user data -- encrypting would add cost/risk for no confidentiality benefit."},
    ]

    report = {
        **config.run_metadata(sample_count=len(coverage_matrix)),
        "evaluation_name": "encryption_at_rest_final",
        "algorithm": "AES-256-GCM (cryptography.hazmat.primitives.ciphers.aead.AESGCM)",
        "primitive_reference": "backend/app/core/encryption.py",
        "shared_text_field_helpers": "encrypt_text_field/decrypt_text_field (app/core/encryption.py), used by both PostgresSessionStore (ChatTurn.content) and session_repository (ChatSession.title) -- no duplicated crypto logic across call sites.",
        "key_source": "Settings.encryption_key_b64, read from ENCRYPTION_KEY_B64 in the environment/.env -- never hardcoded, never committed. Generated via `python -c \"from app.core.encryption import generate_key; print(generate_key())\"`.",
        "nonce_strategy": "Fresh os.urandom(12) (96-bit) nonce per encryption call -- never reused.",
        "aad_strategy": "session_id bound as associated data on both ChatTurn.content and ChatSession.title -- a ciphertext from one session cannot be decrypted under a different session_id, even with the correct key.",
        "on_disk_format": "'enc1:' prefix + base64(nonce(12 bytes) + AESGCM ciphertext-with-tag). Unmarked values are treated as legacy plaintext, never mistaken for ciphertext.",
        "backward_compatibility": "New writes are always encrypted. Existing unmarked (legacy plaintext) rows remain readable as-is -- no forced migration, no silent rewrite on read.",
        "encryption_scope_expanded_this_pass": ["ChatSession.title"],
        "encryption_scope_preexisting": ["ChatTurn.content"],
        "coverage_matrix": coverage_matrix,
        "real_checks_executed_this_run": checks,
        "all_checks_passed": all(checks.values()),
        "pytest_result": pytest_result,
        "tenant_session_isolation": "Verified by cross_session_aad_isolation check above and by tests/test_session_repository_encryption.py::TestSessionIsolation / tests/test_postgres_session_store_encryption.py::TestSessionIsolation -- encryption is an additional guarantee on top of, never a replacement for, the existing tenant_id/session_id ownership checks.",
        "encryption_in_transit": "OUT OF SCOPE for this pass. HTTPS/TLS status is documented separately in docs/ARCHITECTURE.md and docs/MODULE10_FINAL_SUBMISSION.md's existing limitations -- not re-claimed or re-measured here.",
        "limitations": [
            "Uploaded raw PDF files on disk remain unencrypted (see coverage_matrix reason).",
            "FAISS metadata.json chunk text and the FAISS vector index itself remain unencrypted (see coverage_matrix reasons) -- documented as a boundary, not solved.",
            "Feedback events (feedback.jsonl) remain unencrypted -- an evaluation artifact, not primary user-content storage.",
            "No key rotation mechanism exists -- a single static key from ENCRYPTION_KEY_B64 for the process lifetime.",
            "This is application-level encryption; it does not substitute for platform-level encryption (e.g. an encrypted disk/volume) at the infrastructure layer, which is out of scope for this project's local/demo deployment.",
        ],
    }

    path = config.save_report(report, name="encryption_at_rest_final")
    print(f"Saved: {path}")
    print(f"All real checks passed: {report['all_checks_passed']}")
    print(f"Pytest: {pytest_result['summary_line']}")


if __name__ == "__main__":
    main()
