"""Lightweight regex-based PII detection for extracted document text.

Policy: flag and continue. A document containing PII-like text is still
ingested and indexed as normal — we only log that it was found, as a
signal for whoever reviews document intake. Only per-type counts are
logged, never the matched values themselves, so the log stream can't
become a second place PII leaks to.
"""

import re

_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_PHONE_RE = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
_ID_NUMBER_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b|\b[A-Z]{1,2}\d{6,9}\b")


def detect_pii(text: str) -> dict[str, int]:
    """Return a count of PII-like matches per type; types with zero hits are omitted."""
    counts = {
        "email": len(_EMAIL_RE.findall(text)),
        "phone_number": len(_PHONE_RE.findall(text)),
        "id_number": len(_ID_NUMBER_RE.findall(text)),
    }
    return {pii_type: count for pii_type, count in counts.items() if count > 0}
