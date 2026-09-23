"""Appends chat feedback events (thumbs up/down, plus an optional detailed
rubric review) to a local JSONL file.

No database: feedback is low-volume and append-only for this project's
scale, so one file plus eval/metrics_report.py reading it back to compute
Acceptance Rate (and, for rubric-bearing events, per-criterion averages
and Inter-Annotator Agreement) is enough. Mirrors upload_service.py's
path convention (a directory name from Settings, resolved relative to
backend/).

ENCRYPTION AT REST (Module 10 gap-closure): `comment` is real,
free-text, human-typed content a reviewer can put anything in — the one
genuinely sensitive field in this file. It is encrypted with
app.core.encryption's shared encrypt_text_field/decrypt_text_field
helpers (the same primitive/format ChatTurn.content and
ChatSession.title already use), message_id bound as associated data.
`rating`/`rubric`/`reviewer_id`/`timestamp` stay plaintext — they are
structured, non-free-text fields eval/metrics_report.py aggregates
directly from this same file (Acceptance Rate, per-criterion averages,
Inter-Annotator Agreement); encrypting them would require every
existing aggregation reader to decrypt first, a broader change than
this file's own genuinely-sensitive surface justifies.
"""

import json
import logging
from datetime import UTC, datetime
from typing import Any

from app.core.config import settings
from app.core.encryption import decrypt_text_field, encrypt_text_field
from app.services import s3_sync_service

FEEDBACK_DIR = settings.data_dir(settings.feedback_dir_name)
FEEDBACK_PATH = FEEDBACK_DIR / settings.feedback_filename

logger = logging.getLogger(__name__)


def _decrypt_comment(stored: str | None, *, message_id: str) -> str | None:
    if stored is None:
        return None
    return decrypt_text_field(stored, associated_data=message_id, key_b64=settings.encryption_key_b64)


def list_feedback(limit: int = 50, reviewer_id: str | None = None) -> list[dict[str, Any]]:
    """Read back recorded feedback events, newest first (Feature #6).

    Mirrors how eval/metrics_report.py consumes the same file — this is the
    same JSONL, just surfaced as an API response for the app's own "recent
    feedback" view. Best-effort: a missing, empty, or partially-corrupt
    file yields whatever lines parsed (or []), never an exception — the
    read-back surface must not be able to break the API when the file is
    mid-write or absent on a fresh deployment.
    """
    if not FEEDBACK_PATH.exists():
        return []
    events: list[dict[str, Any]] = []
    with FEEDBACK_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                # A truncated tail line can exist if the process was
                # killed mid-write; skip it rather than failing the read.
                continue
            if isinstance(event, dict):
                if event.get("comment") is not None and event.get("message_id"):
                    event["comment"] = _decrypt_comment(event["comment"], message_id=event["message_id"])
                events.append(event)
    if reviewer_id is not None:
        events = [e for e in events if e.get("reviewer_id") == reviewer_id]
    return sorted(events, key=lambda e: e.get("timestamp", ""), reverse=True)[:limit]


def record_feedback(
    message_id: str,
    rating: str,
    comment: str | None,
    reviewer_id: str | None = None,
    rubric: dict[str, Any] | None = None,
) -> None:
    """reviewer_id identifies who submitted this judgment — resolved from
    the authenticated caller's client_name (see query.py's submit_feedback),
    never client-supplied, so one API key can't pose as multiple reviewers
    to game Inter-Annotator Agreement. None for older/non-rubric callers;
    such events are excluded from agreement (see
    eval/metrics_report.py.report_inter_annotator_agreement) since "who
    agrees with whom" is undefined without knowing who "who" is.

    rubric, when given, is RubricScores.model_dump() — the seven 1-5
    per-criterion scores (see app/models/schemas.py).
    """
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)

    encrypted_comment = (
        encrypt_text_field(comment, associated_data=message_id, key_b64=settings.encryption_key_b64)
        if comment is not None
        else None
    )

    event = {
        "timestamp": datetime.now(UTC).isoformat(),
        "message_id": message_id,
        "rating": rating,
        "comment": encrypted_comment,
        "reviewer_id": reviewer_id,
        "rubric": rubric,
    }

    with FEEDBACK_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event) + "\n")
    s3_sync_service.upload_file(FEEDBACK_PATH, settings.feedback_dir_name)

    logger.debug(
        "feedback_written",
        extra={
            "extra_fields": {
                "message_id": message_id,
                "rating": rating,
                "reviewer_id": reviewer_id,
                "has_rubric": rubric is not None,
            }
        },
    )
