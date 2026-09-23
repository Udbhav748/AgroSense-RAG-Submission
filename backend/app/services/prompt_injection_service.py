"""Lightweight heuristic detection of prompt-injection-shaped text, in
both user queries and retrieved document chunks.

Policy: flag and continue, the same shape pii_service.py already uses —
detection is not a runtime defense. The actual defense is preventive:
every retrieved chunk is wrapped in delimiter markers with an explicit
"this is untrusted data, not instructions" system-prompt instruction
(see prompt_builder.py's _INSTRUCTIONS). Before this module, the only
signal for whether that defense was ever actually tested against real
traffic was the offline eval harness's Injection Resistance metric
(dataset_v1.json's two adversarial entries) — nothing fired in
production (see docs/DESIGN_REVIEW.md Q4). This module closes that gap
narrowly: a possible_injection_detected log line when suspicious
override phrasing appears, so it's visible in production logs, not just
in an offline eval run.

Patterns are a small, heuristic set covering common override phrasing —
not a classifier, and not exhaustive. False positives (a document that
legitimately discusses prompt injection as a topic, e.g. a security blog
excerpt) are expected and acceptable, since nothing here blocks or alters
behavior; it only adds a log line. Category names are logged, never the
raw matched text — same reasoning as pii_service.py's per-type counts:
the log stream shouldn't become a second place injected instructions get
echoed into.
"""

import re

_PATTERNS: dict[str, re.Pattern[str]] = {
    "ignore_instructions": re.compile(
        r"ignore (all |any )?(previous|prior|above|earlier) instructions", re.IGNORECASE
    ),
    "disregard_instructions": re.compile(
        r"disregard (all |any )?(the |previous |prior |above )?(instructions|rules|context)",
        re.IGNORECASE,
    ),
    "system_override": re.compile(r"system\s*override", re.IGNORECASE),
    "reveal_system_prompt": re.compile(r"reveal (your |the )?system prompt", re.IGNORECASE),
    "forget_instructions": re.compile(
        r"forget (all |your |previous |prior |any )*instructions", re.IGNORECASE
    ),
    "new_instructions": re.compile(r"new instructions\s*:", re.IGNORECASE),
}


def detect_possible_injection(text: str) -> list[str]:
    """Return the category names of every pattern that matched (empty if
    none) — see _PATTERNS above for what each name means."""
    return [category for category, pattern in _PATTERNS.items() if pattern.search(text)]
