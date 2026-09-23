"""A deterministic, zero-cost, zero-network LLMClient implementation --
Module 10 gap-closure (P7), added specifically so the real HTTP/uvicorn
request path can be load/concurrency-tested without depending on a live
Gemini/Groq call (rate limits, cost, quota) or on in-process
unittest.mock (which cannot reach a genuinely separate subprocess).

Reachable ONLY via Settings.llm_provider == "mock" (see
app/services/llm_provider.py's _PROVIDERS registry) -- never the
default, never activated by any other configuration. This is a
benchmarking/testing aid, not a production LLM backend: it never calls
a real model and always returns the same grounded-looking, cited answer
text so downstream grounding/citation-adjacent logic sees realistic
shapes without depending on live generation quality.
"""

from __future__ import annotations

from collections.abc import Iterator

from app.services.llm_client import LLMClient

_ANSWER_TEXT = (
    "This is a deterministic mock answer citing the retrieved context [1]. "
    "No live LLM provider was called to produce this response."
)


class MockLLMClient(LLMClient):
    """Instant, deterministic, no-network stand-in for a real provider.
    Every call is O(1) and makes no I/O -- exists purely so
    application/HTTP/retrieval overhead can be measured under
    concurrency without conflating it with live provider latency, rate
    limits, or cost (see eval/module10/runners/run_load_concurrency_final_eval.py).
    """

    def generate(self, prompt: str) -> str:
        return _ANSWER_TEXT

    def generate_stream(self, prompt: str) -> Iterator[str]:
        yield _ANSWER_TEXT

    def generate_structured(self, prompt: str) -> str:
        return '{"answer": "%s", "sources": []}' % _ANSWER_TEXT
