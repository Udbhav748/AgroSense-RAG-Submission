"""Crop-context extraction, shared by rag_service.py.

This module used to also hold a full duplicate query-planning/routing
implementation (PlanDecision, plan_query, route_query, QueryRouter,
build_diagnosis_query, match_conversational_reply, ...) — a leftover from
an earlier "extract everything out of rag_service" pass that never
actually got wired in anywhere; rag_service.py kept its own equivalents
(_plan, _route, _build_diagnosis_query, _match_conversational_reply) as
the real, tested, production implementations throughout. That dead
duplicate was removed (see git history) once Phase 1's agent-graph audit
confirmed it had zero production callers. extract_crop_context is the one
function rag_service.py actually imports (aliased as _extract_crop_context)
and is kept here.
"""

from __future__ import annotations

import re

# Canonical crop names recognized across agricultural and LeafSense query contexts.
_CROP_KEYWORDS: dict[str, str] = {
    "tomato": "tomato",
    "tomatoes": "tomato",
    "potato": "potato",
    "potatoes": "potato",
    "corn": "corn",
    "maize": "corn",
    "apple": "apple",
    "apples": "apple",
    "grape": "grape",
    "grapes": "grape",
    "grapevine": "grape",
    "peach": "peach",
    "peaches": "peach",
    "pepper": "bell pepper",
    "peppers": "bell pepper",
    "bell pepper": "bell pepper",
    "bell peppers": "bell pepper",
    "capsicum": "bell pepper",
    "chilli": "bell pepper",
    "chili": "bell pepper",
    "cherry": "cherry",
    "cherries": "cherry",
    "strawberry": "strawberry",
    "strawberries": "strawberry",
    "blueberry": "blueberry",
    "blueberries": "blueberry",
    "raspberry": "raspberry",
    "raspberries": "raspberry",
    "soybean": "soybean",
    "soybeans": "soybean",
    "soy": "soybean",
    "squash": "squash",
    "zucchini": "squash",
    "pumpkin": "squash",
    "orange": "orange",
    "oranges": "orange",
    "citrus": "orange",
    "wheat": "wheat",
    "rice": "rice",
    "cotton": "cotton",
    "cucumber": "cucumber",
    "cucumbers": "cucumber",
    "onion": "onion",
    "onions": "onion",
    "garlic": "garlic",
    "lettuce": "lettuce",
    "coffee": "coffee",
    "banana": "banana",
    "bananas": "banana",
    "mango": "mango",
    "mangoes": "mango",
    "sugarcane": "sugarcane",
}

_CROP_PATTERN = re.compile(
    r"\b("
    + "|".join(re.escape(k) for k in sorted(_CROP_KEYWORDS.keys(), key=len, reverse=True))
    + r")\b",
    re.IGNORECASE,
)


def extract_crop_context(text: str | None) -> str | None:
    """Extract standard crop context (e.g. 'tomato', 'potato', 'peach') from text or query."""
    if not text:
        return None
    match = _CROP_PATTERN.search(text.lower())
    if match:
        matched_token = match.group(1).lower()
        return _CROP_KEYWORDS.get(matched_token, matched_token)
    return None
