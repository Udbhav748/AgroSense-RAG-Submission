"""Web search fallback tool: fetches top web results when document
retrieval doesn't confidently answer a query (see rag_service.ChatService's
corrective RAG loop).

Provider-switchable (Feature #7) via Settings.web_search_provider:
- "duckduckgo" (default) — keyless, queries DDG's public search directly
  via the duckduckgo-search SDK.
- "brave" / "bing" — keyed providers, dispatched over httpx using
  Settings.web_search_api_key.

Isolated wrapper: nothing else in the codebase imports duckduckgo_search
directly, the same pattern gemini_client.py/groq_client.py use to isolate
their own SDKs.

Force-disable without key: if settings.web_search_enabled is True but the
configured provider needs an API key that isn't set, the tool is treated as
disabled — search_web() returns [] with a distinct "web_search_unavailable"
log event rather than failing (or, worse-looking, returning zero results as
if "no results" was a real answer). web_search_ready() exposes the same
check for callers/monitoring.

Note: DuckDuckGo's unofficial search endpoints are known to rate-limit or
silently return zero results from data-center/cloud IPs (bot detection),
even with no error raised. search_web() doesn't try to distinguish that
from "genuinely no results" — callers already treat an empty list as "no
web context available" either way, so there's nothing more useful to do
with the distinction than log it.
"""

import logging
import time
from typing import Any

try:
    from duckduckgo_search import DDGS
    from duckduckgo_search.exceptions import DuckDuckGoSearchException
except ImportError:
    DDGS = None  # type: ignore[assignment,misc]
    DuckDuckGoSearchException = Exception  # type: ignore[assignment,misc]
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_fixed,
)

from app.core.config import settings
from app.core.exceptions import WebSearchError
from app.models.document import WebSearchResult
from app.services.tool_registry import track_tool

logger = logging.getLogger(__name__)

KEYED_PROVIDERS = frozenset({"brave", "bing"})
SUPPORTED_PROVIDERS = KEYED_PROVIDERS | frozenset({"duckduckgo"})


def web_search_ready() -> bool:
    """True if the configured provider can actually search right now.

    False when provider is unknown or is a keyed provider without a key —
    the "force-disable without key" behavior (Feature #7). Distinct from
    Settings.web_search_enabled (the master on/off switch): a deployment
    that sets web_search_enabled=True but forgets WEB_SEARCH_API_KEY for a
    keyed provider gets a silently-disabled tool, not a crash.
    """
    provider = settings.web_search_provider
    if provider not in SUPPORTED_PROVIDERS:
        logger.warning(
            "web_search_unavailable",
            extra={"extra_fields": {"reason": "unknown_provider", "provider": provider}},
        )
        return False
    if provider in KEYED_PROVIDERS and not settings.web_search_api_key:
        logger.warning(
            "web_search_unavailable",
            extra={
                "extra_fields": {
                    "reason": "missing_api_key",
                    "provider": provider,
                }
            },
        )
        return False
    return not (provider == "duckduckgo" and DDGS is None)


def _log_retry(retry_state: RetryCallState) -> None:
    # .outcome is a property, not a plain attribute -- assign it to a local
    # so mypy can narrow the None check (it won't narrow across repeated
    # property reads, since a property isn't guaranteed to return the same
    # value each access).
    outcome = retry_state.outcome
    exception = outcome.exception() if outcome is not None else None
    logger.warning(
        "web_search_retrying",
        extra={
            "extra_fields": {
                "attempt": retry_state.attempt_number,
                "exception": str(exception),
            }
        },
    )


def _search_duckduckgo(query: str, max_results: int) -> list[WebSearchResult]:
    with DDGS(timeout=settings.web_search_timeout_seconds) as ddgs:
        raw_results = list(ddgs.text(query, max_results=max_results))
    return [
        WebSearchResult(
            title=raw.get("title", ""),
            url=raw.get("href", ""),
            snippet=raw.get("body", ""),
        )
        for raw in raw_results
    ]


def _search_keys(query: str, max_results: int) -> list[WebSearchResult]:
    """Shared JSON-in / JSON-out path for the keyed httpx providers
    (brave, bing). Each provider implements `_parse_<provider>` on the
    raw response body; the request mechanics are identical."""
    import httpx

    provider = settings.web_search_provider
    if not settings.web_search_api_key:
        raise WebSearchError(f"{provider} requires an API key (WEB_SEARCH_API_KEY)")

    headers = {"Accept": "application/json"}
    if provider == "brave":
        url = f"https://api.search.brave.com/res/v1/web/search?q={query}&count={max_results}"
        headers["X-Subscription-Token"] = settings.web_search_api_key
        parse = _parse_brave
    elif provider == "bing":
        url = f"https://api.bing.microsoft.com/v7.0/search?q={query}&count={max_results}"
        headers["Ocp-Apim-Subscription-Key"] = settings.web_search_api_key
        parse = _parse_bing
    else:
        raise WebSearchError(f"Unknown web search provider: {provider}")

    try:
        response = httpx.get(url, headers=headers, timeout=settings.web_search_timeout_seconds)
        response.raise_for_status()
        return parse(response.json())
    except WebSearchError:
        raise
    except Exception as exc:
        raise WebSearchError(f"{provider} web search failed: {exc}") from exc


def _parse_brave(data: dict[str, Any]) -> list[WebSearchResult]:
    results = []
    for item in data.get("web", {}).get("results", []):
        results.append(
            WebSearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("description", ""),
            )
        )
    return results


def _parse_bing(data: dict[str, Any]) -> list[WebSearchResult]:
    results = []
    for item in data.get("webPages", {}).get("value", []):
        results.append(
            WebSearchResult(
                title=item.get("name", ""),
                url=item.get("url", ""),
                snippet=item.get("snippet", ""),
            )
        )
    return results


def _search_web_core(query: str, max_results: int | None = None) -> list[WebSearchResult]:
    """Core web-search logic (see search_web() below). Split out of the
    @track_tool/@retry-wrapped public function so the dynamic tool
    registry (services/tools/) can invoke web search without double-
    instrumenting it: the registry is the single instrumentation point on
    the agent path, @track_tool is the single one on the inline path.

    Raises WebSearchError on failure — callers (ChatService, the tool
    registry) are expected to catch it and degrade gracefully (treat it
    like zero results) rather than fail the whole chat request over a
    best-effort fallback tool.

    Returns [] (never raises) when the configured provider isn't actually
    usable (unknown provider, or keyed provider without its key) — the
    "force-disable without key" behavior of Feature #7. The distinct
    "web_search_unavailable" event is logged by web_search_ready() so the
    operator can tell "disabled" from "searched and found nothing".
    """
    resolved_max_results = (
        max_results if max_results is not None else settings.web_search_result_count
    )

    if not web_search_ready():
        return []

    start = time.perf_counter()
    provider = settings.web_search_provider
    try:
        if provider == "duckduckgo":
            results = _search_duckduckgo(query, resolved_max_results)
        else:
            results = _search_keys(query, resolved_max_results)
    except DuckDuckGoSearchException as exc:
        raise WebSearchError(f"Web search failed: {exc}") from exc
    except Exception as exc:
        raise WebSearchError(f"Unexpected error during web search: {exc}") from exc

    processing_duration = time.perf_counter() - start

    logger.info(
        "web_search_completed",
        extra={
            "extra_fields": {
                "query_length": len(query),
                "result_count": len(results),
                "provider": provider,
                "processing_duration": round(processing_duration, 4),
            }
        },
    )

    return results


@track_tool("web_search")
@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(1),
    retry=retry_if_exception_type(WebSearchError),
    reraise=True,
    before_sleep=_log_retry,
)
def search_web(query: str, max_results: int | None = None) -> list[WebSearchResult]:
    """Public, instrumented web-search entry point — the inline
    ChatService path. Validates args, runs _search_web_core (with retry),
    and logs one tool_invocation event. The dynamic tool registry calls
    _search_web_core directly instead, so a single web-search call is
    never counted twice.

    Raises WebSearchError on failure — callers are expected to catch it
    and degrade gracefully (treat it like zero results) rather than fail
    the whole chat request over a best-effort fallback tool.

    Returns [] (never raises) when the configured provider isn't actually
    usable (unknown provider, or keyed provider without its key) — the
    "force-disable without key" behavior of Feature #7. The distinct
    "web_search_unavailable" event is logged by web_search_ready() so the
    operator can tell "disabled" from "searched and found nothing".
    """
    return _search_web_core(query, max_results=max_results)
