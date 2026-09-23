"""Shared sub-query planning, used by both the web ResearchAgent and
the local-document LocalResearchAgent — one JSON-mode LLM call that
breaks a complex query into up to max_subqueries standalone
sub-queries. Degrades to [query] itself on any failure."""

import json
import logging

from app.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

_PLAN_PROMPT = (
    "Break this question into up to {max_subqueries} standalone "
    "search queries that together would answer it. If the question "
    "is already simple, return just one query (the original "
    'question). Return ONLY a JSON object like {{"queries": ["...", "..."]}}.\n\n'
    "Question: {query}"
)


def plan_subqueries(llm_client: LLMClient, query: str, max_subqueries: int) -> list[str]:
    try:
        raw = llm_client.generate(
            _PLAN_PROMPT.format(query=query, max_subqueries=max(1, max_subqueries))
        ).strip()
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = json.loads(raw)
        queries = parsed.get("queries") if isinstance(parsed, dict) else None
        if isinstance(queries, list) and queries:
            return [str(q) for q in queries][: max(1, max_subqueries)]
    except Exception:
        logger.warning("subquery_planning_failed", extra={"extra_fields": {"query": query}})
    return [query]
