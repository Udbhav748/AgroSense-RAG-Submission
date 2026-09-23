"""Ground-truth PII Recall check for app.services.pii_service.detect_pii().

Builds a small synthetic document with a known, exact number of planted
PII instances per type (email, phone number, SSN-shaped string), runs it
through detect_pii() directly, and reports Recall = detected / planted,
per type and overall.

No LLM, no vector store, no network call, no precondition beyond having
the backend's dependencies installed — detect_pii() is a pure regex
function over a plain string (see its own docstring). Fast and fully
offline, deliberately separate from run_eval.py rather than folded into
it, since that harness's preconditions (a live LLM, an indexed document)
have nothing to do with what this checks.

The PLANTED_* lists below ARE the ground truth, not a description of it:
build_document() interpolates these exact values into the document text,
so "how many were planted" is always len(PLANTED_EMAILS) etc. — derived
mechanically from the same source used to build the text, never a
separate hand-count that could silently drift out of sync with it.

detect_pii() only returns per-type counts, never which specific strings
matched ("never the matched values themselves, so the log stream can't
become a second place PII leaks to" — see its own docstring). That's why
recall here is necessarily count-based (detected_count / planted_count
per type) rather than matching each planted string to a specific find —
which is also why build_document() has to stay free of any *incidental*
text that would coincidentally match one of the three regexes: an
unplanned extra match would silently inflate a type's detected count
and make the count-based recall meaningless rather than just imprecise.

Usage (from backend/):
    python eval/pii_recall_check.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.pii_service import detect_pii  # noqa: E402

# Ground truth. Fake, sequential, clearly-not-real values (555 phone
# prefix, sequential SSNs) — realistic in shape, not real people's data.
PLANTED_EMAILS = [
    "jane.doe@example.com",
    "r.smith123@testmail.org",
    "contact.us+billing@company-name.co.uk",
    "a.b@sub.domain.net",
    "info@my-startup.io",
]

PLANTED_PHONES = [
    "555-123-4567",
    "(555) 234-5678",
    "555.345.6789",
    "+1 555-456-7890",
    "5555567890",
]

PLANTED_IDS = [
    "123-45-6789",
    "234-56-7890",
    "345-67-8901",
    "456-78-9012",
    "567-89-0123",
]


def build_document() -> str:
    """A small synthetic HR onboarding memo — a plausible real-world place
    for exactly this mix of PII to appear together. Every planted value is
    interpolated from the lists above, not retyped, so the document can
    never drift out of sync with what it claims to contain."""
    paragraphs = [
        "New Employee Onboarding Records",
        (
            f"Employee: Jane Doe. Email: {PLANTED_EMAILS[0]}. Phone: {PLANTED_PHONES[0]}. "
            f"Social Security Number on file: {PLANTED_IDS[0]}. Jane's start date is next "
            "Monday, and her manager will reach out shortly with building access details."
        ),
        (
            f"Employee: Robert Smith. Contact him at {PLANTED_EMAILS[1]} for any paperwork "
            f"questions, or by phone at {PLANTED_PHONES[1]}. His Social Security Number on "
            f"record is {PLANTED_IDS[1]}. Robert previously worked in the finance department "
            "at a different company."
        ),
        (
            f"Billing and payroll inquiries for this cohort should go to {PLANTED_EMAILS[2]}, "
            f"or call the support line at {PLANTED_PHONES[2]}. Reference SSN {PLANTED_IDS[2]} "
            f"when discussing Robert's direct deposit setup, and SSN {PLANTED_IDS[3]} when "
            "discussing Jane's."
        ),
        (
            f"Employee: Aisha Bello. Reach her at {PLANTED_EMAILS[3]} or at "
            f"{PLANTED_PHONES[3]}. Her Social Security Number is {PLANTED_IDS[4]}. Aisha will "
            "be shadowing the operations team during her first two weeks."
        ),
        (
            f"For IT support during onboarding, email {PLANTED_EMAILS[4]} or call "
            f"{PLANTED_PHONES[4]}. All new hires should complete the welcome survey by Friday."
        ),
    ]
    return "\n\n".join(paragraphs)


def main() -> None:
    text = build_document()
    detected = detect_pii(text)

    planted = {
        "email": len(PLANTED_EMAILS),
        "phone_number": len(PLANTED_PHONES),
        "id_number": len(PLANTED_IDS),
    }

    print("=== PII Recall (pii_service.detect_pii) ===")
    print(f"Synthetic document: {len(text)} chars, {sum(planted.values())} PII instances planted\n")

    total_planted = 0
    total_detected_for_recall = 0
    surprises = []

    for pii_type, planted_count in planted.items():
        detected_count = detected.get(pii_type, 0)
        if detected_count > planted_count:
            # The document has an incidental match beyond what was
            # deliberately planted -- a bug in this script's test data,
            # not a real detector "over-performing". Flagged rather than
            # silently absorbed into the recall number.
            surprises.append(
                f"{pii_type}: detected {detected_count} but only {planted_count} were "
                "planted -- build_document() likely has an unintended incidental match"
            )
        recall = detected_count / planted_count if planted_count else None
        total_planted += planted_count
        total_detected_for_recall += min(detected_count, planted_count)
        recall_str = f"{recall:.4f}" if recall is not None else "n/a"
        print(f"  {pii_type:<15s} planted={planted_count:<3d} detected={detected_count:<3d} recall={recall_str}")

    overall_recall = total_detected_for_recall / total_planted if total_planted else None
    print(f"\nOverall PII Recall: {overall_recall:.4f} ({total_detected_for_recall}/{total_planted})")

    if surprises:
        print("\nWARNING -- unexpected extra matches (fix build_document(), don't trust this run's recall):")
        for surprise in surprises:
            print(f"  - {surprise}")


if __name__ == "__main__":
    main()
