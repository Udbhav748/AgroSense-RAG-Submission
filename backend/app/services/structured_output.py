"""Parse and validate the LLM's JSON-mode output.

Companion to prompt_builder.build_structured_prompt and the LLM clients'
generate_structured(). The model's raw output is expected to be a JSON
object matching StructuredAnswer; this module parses it defensively (some
providers wrap the JSON in markdown fences or trailing text) and returns a
StructuredAnswer or None on failure — the caller falls back to the
free-text path, so a malformed or non-JSON answer never breaks a request.

This is the "LLM-output validation" half of structured output: the wire
contract (ChatResponse) is already validated by Pydantic; this validates
the model's own output before it becomes the answer.
"""

from __future__ import annotations

import json
import logging
import re

from pydantic import ValidationError

from app.models.schemas import StructuredAnswer

logger = logging.getLogger(__name__)

# Some providers emit the JSON wrapped in ```json ... ``` fences or with
# stray leading/trailing prose. Strip a code fence if present, then attempt
# to extract a top-level JSON object from the remainder.
_FENCE_RE = re.compile(r"^```(?:json)?\s*", re.IGNORECASE)
_FENCE_END_RE = re.compile(r"\s*```$")


def parse_structured_answer(raw: str) -> StructuredAnswer | None:
    """Parse raw model output into a StructuredAnswer, or None.

    Never raises: any malformed/valid-but-wrong output returns None so the
    caller can degrade to the plain-text answer. A structured-output parse
    failure is logged (not warned) since it's an expected degradation path.
    """
    if not raw or not raw.strip():
        logger.info("structured_output_parse_failed", extra={"extra_fields": {"reason": "empty"}})
        return None

    cleaned = _FENCE_END_RE.sub("", _FENCE_RE.sub("", raw.strip())).strip()
    if not cleaned:
        logger.info(
            "structured_output_parse_failed", extra={"extra_fields": {"reason": "only_fence"}}
        )
        return None

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        # Fall back to extracting the first {...} block — models sometimes
        # append a trailing sentence after the JSON.
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            logger.info(
                "structured_output_parse_failed",
                extra={"extra_fields": {"reason": "no_json_block"}},
            )
            return None
        try:
            payload = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            logger.info(
                "structured_output_parse_failed",
                extra={"extra_fields": {"reason": "invalid_json"}},
            )
            return None

    try:
        answer = StructuredAnswer.model_validate(payload)
    except ValidationError as exc:
        logger.info(
            "structured_output_parse_failed",
            extra={"extra_fields": {"reason": "schema_mismatch", "detail": str(exc)}},
        )
        return None

    if not answer.answer.strip():
        logger.info(
            "structured_output_parse_failed",
            extra={"extra_fields": {"reason": "empty_answer"}},
        )
        return None

    return answer
