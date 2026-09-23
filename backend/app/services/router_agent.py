"""LLM router agent: classifies a query's intent and hands off to a
specialist agent or tool.

This is the planner half of the multi-agent design. ChatService's
deterministic keyword planner (rag_service._plan) can only recognize
small talk, "summarize <uuid>", or a generic document question — it has
no notion of "this needs multi-step web research." The router agent
upgrades exactly that: for anything the deterministic fast paths don't
lock down, it asks the LLM (in JSON mode) whether the query should go to
retrieval, summarization, or the Research agent.

Deliberate failure-safe design, mirroring the codebase's structured-output
philosophy (structured_output.py): a router parse failure, empty response,
or malformed action degrades to the keyword planner's decision — routing
is a quality improvement, never a new failure mode. Deterministic fast
paths (small talk, "summarize <uuid>") short-circuit before any LLM call.

RouterAgent is constructed with the keyword planner injected (a
callable taking query, history and returning something with `.action` and
`.document_id`), so this module never imports rag_service — no circular
dependency, and the fast-path/fallback behavior is unit-testable with a
stub planner.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from app.core.config import settings
from app.services.agent_events import log_agent_completed, log_agent_started

if TYPE_CHECKING:
    from collections.abc import Callable

    from app.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

# Actions the router may return. "research" is the new one the keyword
# planner can't express: the query is (likely) not answerable from the
# uploaded corpus and needs the multi-step Research agent.
_KNOWN_ACTIONS = {"retrieve", "research", "summarize", "conversational"}

# Fast-path actions the injected keyword planner is trusted to decide
# without an LLM call — they're deterministic and unambiguous.
_FAST_ACTIONS = {"conversational", "summarize"}

# The JSON object the router is asked to emit. Kept in sync with
# RouterDecision below — change both together.
_ROUTER_SCHEMA = (
    '{"action": "retrieve" | "research" | "summarize" | "conversational", '
    '"document_id": "<uuid or null>", "reason": "<one short line>"}'
)

_ROUTER_PROMPT = (
    "You are the routing agent of a multi-agent document assistant. "
    "Classify the user's intent and respond with ONLY a single JSON object, "
    "no prose around it, matching exactly:\n"
    f"{_ROUTER_SCHEMA}\n\n"
    "Choose one action:\n"
    "- retrieve: a factual question the user's uploaded documents could "
    "answer, or you can't tell yet. Default to this when unsure.\n"
    "- research: a question about current, recent, or outside-the-corpus "
    "information (news, prices, events, general knowledge a document could "
    "not plausibly contain) that needs a multi-step web research pass.\n"
    "- summarize: a request to summarize a specific document named by its "
    "UUID in the query (set document_id to that UUID). Never use this "
    "without a UUID in the query.\n"
    "- conversational: greetings, thanks, small talk, or remarks about "
    "whether the system is working.\n\n"
    "Use conversation history only to disambiguate follow-ups; answer for "
    "the user's actual request."
)

# UUID pattern, same shape the upload pipeline generates (uuid4).
_UUID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE
)

# Some providers wrap the JSON in ```json fences or stray prose — strip a
# fence and pull the first {...} block, same defensive approach as
# structured_output.parse_structured_answer.
_FENCE_RE = re.compile(r"^```(?:json)?\s*", re.IGNORECASE)
_FENCE_END_RE = re.compile(r"\s*```$")


@dataclass
class RouterDecision:
    action: str
    document_id: str | None = None
    crop: str | None = None
    collection: str | None = None


def _parse_router_json(raw: str) -> dict[str, Any] | None:
    """Best-effort parse of the router's JSON output. Never raises."""
    if not raw or not raw.strip():
        return None
    cleaned = _FENCE_END_RE.sub("", _FENCE_RE.sub("", raw.strip())).strip()
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            payload = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            return None
    return payload if isinstance(payload, dict) else None


class RouterAgent:
    """LLM-based intent router with deterministic fast paths and a
    keyword-planner fallback. Logs its lifecycle via agent_events.py so
    Handoff Accuracy is measurable from the log."""

    def __init__(self, llm_client: LLMClient, fallback_planner: Callable[..., Any]):
        self._llm_client = llm_client
        self._fallback_planner = fallback_planner

    def _prompt_for(self, query: str, history: list[dict[str, Any]] | None) -> str:
        history_block = ""
        if history:
            lines = "\n".join(
                f"{turn.get('role', 'user')}: {turn.get('content', '')}" for turn in history[-4:]
            )
            history_block = f"\nConversation history:\n{lines}\n"
        return f"{_ROUTER_PROMPT}\n{history_block}\nQuery: {query.strip()}"

    def decide(self, query: str, history: list[dict[str, Any]] | None = None) -> RouterDecision:
        """Return the routed action for `query`.

        Fast paths (conversational / summarize-with-uuid) are decided by
        the injected keyword planner without an LLM call; everything else
        goes to the LLM router, which may upgrade the action to
        "research". Any router failure returns the keyword planner's
        decision unchanged.
        """
        start = time.perf_counter()
        fast = self._fallback_planner(query, history)
        fast_crop = getattr(fast, "crop", None)
        fast_collection = getattr(fast, "collection", None)
        log_agent_started("router", query, decision=fast.action)

        if fast.action in _FAST_ACTIONS:
            log_agent_completed("router", query, start, outcome="fast_path", action=fast.action)
            return RouterDecision(
                action=fast.action,
                document_id=fast.document_id,
                crop=fast_crop,
                collection=fast_collection,
            )

        decision = self._classify(query, history)
        if decision is None:
            # Router LLM call failed or returned something unusable —
            # degrade to the keyword planner's decision.
            logger.warning(
                "router_fallback",
                extra={
                    "extra_fields": {"fallback_action": fast.action, "query_length": len(query)}
                },
            )
            log_agent_completed("router", query, start, outcome="fallback", action=fast.action)
            return RouterDecision(
                action=fast.action,
                document_id=fast.document_id,
                crop=fast_crop,
                collection=fast_collection,
            )

        # Preserve extracted crop context if classification didn't provide one
        if decision.crop is None:
            decision.crop = fast_crop
        if decision.collection is None:
            decision.collection = fast_collection or fast_crop

        log_agent_completed("router", query, start, outcome="llm", action=decision.action)
        return decision

    def _classify(self, query: str, history: list[dict[str, Any]] | None) -> RouterDecision | None:
        """One JSON-mode LLM call; None on any parse/validation failure."""
        try:
            raw = self._llm_client.generate_structured(self._prompt_for(query, history))
        except Exception as exc:
            logger.warning("router_call_failed", extra={"extra_fields": {"error": str(exc)}})
            return None

        payload = _parse_router_json(raw)
        if payload is None:
            logger.info("router_parse_failed", extra={"extra_fields": {"query_length": len(query)}})
            return None

        action = str(payload.get("action", "")).strip().lower()
        if action not in _KNOWN_ACTIONS:
            logger.info(
                "router_unknown_action",
                extra={"extra_fields": {"action": action, "query_length": len(query)}},
            )
            return None

        document_id = payload.get("document_id")
        document_id = str(document_id).strip() if document_id else None
        if document_id and not _UUID_RE.fullmatch(document_id):
            document_id = None

        # "summarize" without a usable UUID is meaningless — treat it as a
        # document query rather than hand off a broken task.
        if action == "summarize" and not document_id:
            logger.info(
                "router_summarize_without_id",
                extra={"extra_fields": {"query_length": len(query)}},
            )
            action = "retrieve"

        # "research" requires the research agent's prerequisite (web search
        # enabled); otherwise degrade to plain retrieval so the request
        # stays answerable.
        if action == "research" and not settings.web_search_enabled:
            logger.info(
                "router_research_degraded_web_disabled",
                extra={"extra_fields": {"query_length": len(query)}},
            )
            action = "retrieve"

        return RouterDecision(action=action, document_id=document_id)
