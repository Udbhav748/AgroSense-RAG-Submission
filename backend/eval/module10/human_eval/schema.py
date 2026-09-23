"""Module 10 gap-closure (P8): schema validation for human-reviewer
rating files (reviewer_1_ratings.json shape, and any reviewer_2 file
filled in from the blinded packet).

Deliberately dependency-free (no pydantic here) since this validates a
small, fixed, hand-authored/hand-filled JSON shape, not an API request
body -- a plain function with clear, specific error messages is easier
for a human filling in the file to act on than a stack trace.
"""

from __future__ import annotations

from eval.module10.human_eval.cases import CASES_BY_ID
from eval.module10.metrics.human import RUBRIC_DIMENSIONS


class ReviewerDataError(ValueError):
    """Raised with a clear, actionable message when a reviewer ratings
    file fails validation -- never silently coerced or ignored."""


def validate_reviewer_file(data: dict) -> list[dict]:
    """Validates the shape produced by generate_reviewer2_packet.py /
    used by reviewer_1_ratings.json. Returns the list of per-case rating
    dicts on success. Raises ReviewerDataError with a specific message on
    the FIRST problem found -- callers get one clear failure, not a
    generic exception.
    """
    if "reviewer_id" not in data or not data["reviewer_id"]:
        raise ReviewerDataError("Missing required field: reviewer_id")
    if "ratings" not in data or not isinstance(data["ratings"], list):
        raise ReviewerDataError("Missing or malformed required field: ratings (must be a list)")

    reviewer_id = data["reviewer_id"]
    rows = data["ratings"]
    seen_case_ids: set[str] = set()

    for i, row in enumerate(rows):
        if "case_id" not in row or not row["case_id"]:
            raise ReviewerDataError(f"ratings[{i}]: missing required field case_id")
        case_id = row["case_id"]

        if case_id not in CASES_BY_ID:
            raise ReviewerDataError(
                f"ratings[{i}] (case_id={case_id!r}): not a valid case_id -- must be one of "
                f"{sorted(CASES_BY_ID)[0]}..{sorted(CASES_BY_ID)[-1]}"
            )
        if case_id in seen_case_ids:
            raise ReviewerDataError(f"ratings[{i}] (case_id={case_id!r}): duplicate case_id for reviewer {reviewer_id!r}")
        seen_case_ids.add(case_id)

        if "ratings" not in row or not isinstance(row["ratings"], dict):
            raise ReviewerDataError(f"ratings[{i}] (case_id={case_id!r}): missing or malformed 'ratings' object")
        dims = row["ratings"]

        missing_dims = [d for d in RUBRIC_DIMENSIONS if d not in dims]
        if missing_dims:
            raise ReviewerDataError(
                f"ratings[{i}] (case_id={case_id!r}): missing required dimension(s): {missing_dims}"
            )

        for dim in RUBRIC_DIMENSIONS:
            value = dims[dim]
            if value is None:
                continue  # N/A is valid (e.g. Groundedness/Citation Quality on conversational cases)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ReviewerDataError(
                    f"ratings[{i}] (case_id={case_id!r}, dimension={dim!r}): must be an integer 1-5 or "
                    f"null, got {value!r}"
                )
            if not (1 <= value <= 5):
                raise ReviewerDataError(
                    f"ratings[{i}] (case_id={case_id!r}, dimension={dim!r}): must be between 1 and 5, "
                    f"got {value}"
                )

    missing_cases = set(CASES_BY_ID) - seen_case_ids
    if missing_cases:
        raise ReviewerDataError(
            f"Reviewer {reviewer_id!r} is missing ratings for {len(missing_cases)} case(s): "
            f"{sorted(missing_cases)}"
        )

    return rows


__all__ = ["ReviewerDataError", "validate_reviewer_file"]
