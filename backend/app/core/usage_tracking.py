"""Per-request LLM usage/cost accumulator, keyed by a context variable.

GeminiClient/GroqClient call record_usage() on every generation. When a
request scope is active — ChatService calls reset_usage() at the start of
each request and reads current_usage() when building the chat_query_handled
log — those calls accumulate into a request-scoped total, the source for
the per-request token/cost rollup (Cost per Successful Task).

Outside a request scope record_usage() is a no-op, so direct client usage
in tests or scripts is unaffected: the accumulator is opt-in by resetting
the context variable, never required.

Contextvar (not a module global) so concurrent requests in the same
process don't clobber each other's totals.
"""

from __future__ import annotations

import contextvars
from typing import TypedDict


class Usage(TypedDict):
    llm_calls: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float


_scope: contextvars.ContextVar[Usage | None] = contextvars.ContextVar(
    "llm_usage_scope", default=None
)


def reset_usage() -> None:
    """Start a fresh per-request accumulator (call once per request)."""
    _scope.set(
        {
            "llm_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "estimated_cost_usd": 0.0,
        }
    )


def record_usage(
    *,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    estimated_cost_usd: float,
) -> None:
    """Accumulate one generation's usage into the active request scope, if any."""
    usage = _scope.get()
    if usage is None:
        return
    usage["llm_calls"] += 1
    usage["prompt_tokens"] += prompt_tokens
    usage["completion_tokens"] += completion_tokens
    usage["total_tokens"] += total_tokens
    usage["estimated_cost_usd"] = round(usage["estimated_cost_usd"] + estimated_cost_usd, 6)


def current_usage() -> Usage:
    """Request-scoped totals so far (all zeros when no scope is active)."""
    usage = _scope.get()
    if usage is None:
        return {
            "llm_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "estimated_cost_usd": 0.0,
        }
    # A copy, not the live accumulator itself — callers must not be able to
    # mutate the request-scoped total by mutating what this returns.
    # dict(usage) makes that copy but degrades TypedDict-ness to
    # dict[str, object] (heterogeneous field types), which mypy then
    # rejects against the declared Usage return type; constructing a fresh
    # Usage from the same keys keeps both the copy and the precise type.
    return Usage(**usage)
