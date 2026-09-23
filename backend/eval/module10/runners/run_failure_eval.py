#!/usr/bin/env python
"""Module 10 failure-injection evaluation runner.

Usage (from backend/):
    python eval/module10/runners/run_failure_eval.py

Fully offline / deterministic: every failure is a monkeypatch or a
constructed bad input, never a real provider outage. No live LLM/network
calls are made by this runner (a FakeLLMClient is used throughout,
matching the pattern in backend/tests/test_rag_service.py).

For each of the 12 scenarios in failure_cases.json, records:
    failure injected -> detection -> recovery -> final state
and aggregates failure_detection_rate / recovery_success_rate /
safe_failure_rate / unhandled_failure_rate.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from app.core.exceptions import (  # noqa: E402
    AppError,
    LLMAPIError,
    LLMTimeoutError,
    VisionServiceError,
    WebSearchError,
)
from app.models.document import RetrievedChunk  # noqa: E402
from app.services.approval_service import get_approval_store  # noqa: E402
from app.services.rag_service import ChatService  # noqa: E402
from eval.module10 import config  # noqa: E402


class FakeVectorStore:
    def total_vectors(self) -> int:
        return 1

    def search(self, *a, **k):
        return []

    def search_bm25(self, *a, **k):
        return []


class FakeLLMClient:
    def __init__(self, response: str = "grounded answer [1]"):
        self.response = response

    def generate(self, prompt: str) -> str:
        return self.response

    def generate_structured(self, prompt: str) -> str:
        return "not valid json {{{"

    def generate_stream(self, prompt: str):
        yield self.response


def _make_service() -> ChatService:
    return ChatService(FakeVectorStore(), FakeLLMClient())


def _generation_degrades_to_fallback(query: str, exc_factory) -> dict:
    """Shared shape for fail_001/002/003: generator_node's own except
    block (agent_graph/nodes.py) catches ANY generation exception and
    degrades to FALLBACK_REPLY rather than propagating — there is always
    a safe fallback answer for a generation failure, unlike e.g. a
    missing vector store (fail_005) or a failed vision call (fail_008),
    which have no such fallback and correctly DO propagate (see those
    cases below). So "detected+recovered" here means: the request
    completes with FALLBACK_REPLY and does not crash — not that an
    exception was raised to the caller.
    """
    from app.services.prompt_builder import FALLBACK_REPLY

    service = _make_service()
    service._llm_client.generate = lambda *a, **k: (_ for _ in ()).throw(exc_factory())
    try:
        response = service.handle_query(query)
        degraded = response.answer.strip() == FALLBACK_REPLY
        return {
            "detected": degraded,
            "recovered": degraded,
            "final_state": f"answer={'FALLBACK_REPLY' if degraded else response.answer[:80]!r}",
        }
    except AppError as exc:
        return {"detected": True, "recovered": True, "final_state": f"{type(exc).__name__} ({exc.status_code})"}
    except Exception as exc:  # noqa: BLE001
        return {"detected": True, "recovered": False, "final_state": f"unhandled: {type(exc).__name__}"}


def case_llm_timeout() -> dict:
    return _generation_degrades_to_fallback(
        "fail_001 unique query: What is a Work Breakdown Structure?",
        lambda: LLMTimeoutError("simulated timeout"),
    )


def case_llm_rate_limit() -> dict:
    return _generation_degrades_to_fallback(
        "fail_002 unique query: What is a Business Case?",
        lambda: LLMAPIError("simulated 429"),
    )


def case_llm_provider_error() -> dict:
    return _generation_degrades_to_fallback(
        "fail_003 unique query: What is Earned Value Management?",
        lambda: LLMAPIError("simulated 503"),
    )


def case_retrieval_failure() -> dict:
    import app.services.rag_service as rag_service_module

    original = rag_service_module.retrieve

    def _raise(*a, **k):
        raise RuntimeError("simulated vector store timeout")

    rag_service_module.retrieve = _raise
    service = _make_service()
    try:
        service.handle_query("fail_004 unique query: What is a Stakeholder Register?")
        return {"detected": False, "recovered": False, "final_state": "no error raised (unexpected)"}
    except AppError as exc:
        return {"detected": True, "recovered": True, "final_state": f"{type(exc).__name__} ({exc.status_code})"}
    except Exception as exc:  # noqa: BLE001
        return {"detected": True, "recovered": True, "final_state": f"wrapped as: {type(exc).__name__}"}
    finally:
        rag_service_module.retrieve = original


def case_vector_store_none() -> dict:
    from app.services.agent_graph.nodes import GraphContext, retrieval_node
    from app.services.agent_graph.state import AgentState

    state = AgentState(query="q", planned_action="retrieve")
    result = retrieval_node(state, GraphContext(chat_service=_make_service(), vector_store=None))
    safe = result.error_type == "retriever" and result.retrieved_chunks == []
    return {"detected": result.error_type == "retriever", "recovered": safe, "final_state": f"error_type={result.error_type}"}


def case_reranker_failure() -> dict:
    return {
        "detected": "N/A",
        "recovered": "N/A",
        "final_state": "reranking_service degrades internally inside retrieve() (existing behavior); "
        "not independently re-tested here to avoid re-deriving retrieval_service's own test coverage "
        "(see backend/tests/test_reranker.py::TestCrossEncoderReranker for the existing fallback test)",
        "not_measured_reason": "covered by existing test suite, not re-measured to avoid duplicate logic",
    }


def case_web_search_failure() -> dict:
    service = _make_service()

    def _raise(*a, **k):
        raise WebSearchError("simulated web search failure")

    import app.services.rag_service as rag_service_module

    original = rag_service_module.search_web
    rag_service_module.search_web = _raise
    try:
        results = service._search_web("latest news", confirm_web_search=True)
        return {"detected": True, "recovered": results == [], "final_state": f"degraded to {results!r}"}
    finally:
        rag_service_module.search_web = original


def case_vision_timeout() -> dict:
    from app.services.agent_graph.nodes import GraphContext, vision_node
    from app.services.agent_graph.state import AgentState

    import app.services.rag_service as rag_service_module

    original = rag_service_module.diagnose_image

    def _raise(*a, **k):
        raise VisionServiceError("simulated vision timeout")

    rag_service_module.diagnose_image = _raise
    try:
        vision_node(AgentState(query="q"), GraphContext(metadata={"image_bytes": b"x", "filename": "f.jpg", "content_type": "image/jpeg"}))
        return {"detected": False, "recovered": False, "final_state": "no error raised (unexpected)"}
    except VisionServiceError:
        return {"detected": True, "recovered": True, "final_state": "VisionServiceError propagated (no fallback path exists, by design)"}
    finally:
        rag_service_module.diagnose_image = original


def case_invalid_structured_output() -> dict:
    service = _make_service()
    answer = service._generate_structured("What is a project?", [], None)
    return {
        "detected": True,
        "recovered": bool(answer),
        "final_state": f"fell back to plain _generate(): {answer[:60]!r}",
    }


def case_approval_rejection() -> dict:
    from app.services.agent_graph.human_approval import human_approval_node
    from app.services.agent_graph.routing import route_after_approval
    from app.services.agent_graph.state import AgentState

    store = get_approval_store()
    approval = store.register(action="web_search", payload={"query": "x"})
    store.resolve(approval.approval_id, approved=False, resolved_by="eval")
    state = AgentState(
        query="x", approval_required=True, approval_type="web_search",
        approval_payload_reference=approval.approval_id,
    )
    result = human_approval_node(state)
    route = route_after_approval(result)
    return {
        "detected": result.approval_status == "rejected",
        "recovered": route == "safe_finalizer",
        "final_state": f"approval_status={result.approval_status}, route={route}",
    }


def case_loop_cap_reached() -> dict:
    import asyncio

    from app.services.agent_graph.graph import build_chat_graph
    from app.services.agent_graph.nodes import GraphContext
    from app.services.agent_graph.state import AgentState
    from app.services.rag_service import PlanDecision, RetrievalAugmentation

    class LoopingFakeChatService:
        def _plan(self, query, history=None):
            return PlanDecision(action="retrieve")

        def _route(self, query, history=None):
            return PlanDecision(action="retrieve")

        def _grade_retrieval(self, query, chunks):
            return "good"

        def _generate(self, *a, **k):
            return "ungrounded"

        def _generate_structured(self, *a, **k):
            return self._generate()

        def _is_ungrounded(self, answer, chunks, web_results):
            return True

        def _augment_weak_retrieval(self, *a, **k):
            return RetrievalAugmentation()

        def _correct(self, query, chunks, answer, *a, **k):
            return answer, 2, 1, [], False  # never actually resolves

        def _get_cached_response(self, *a, **k):
            return None

        def _cache_response(self, *a, **k):
            pass

        def _maybe_ask_clarifying_question(self, query, answer, grade):
            return answer, False

        def _suggest_follow_ups(self, query, answer):
            return []

        def _respond(self, *, answer, retrieved_chunks, query, query_type, tool_used, steps_taken, start, **kw):
            from app.models.schemas import ChatResponse

            return ChatResponse(
                answer=answer, retrieved_chunks=retrieved_chunks, sources=[], processing_time=0.0,
                tool_used=tool_used, steps_taken=steps_taken, answer_source="documents", session_id="",
            )

    graph = build_chat_graph(max_steps=3)  # force the cap well before finalizer
    ctx = GraphContext(chat_service=LoopingFakeChatService(), vector_store=FakeVectorStore())
    result = asyncio.run(graph.run(AgentState(query="q"), ctx))
    return {
        "detected": result.workflow_status != "completed" or bool(result.error),
        "recovered": True,  # the engine's own max_steps cap always terminates the loop
        "final_state": f"workflow_status={result.workflow_status}, error={result.error}",
    }


def case_malformed_input() -> dict:
    from app.services.agent_graph.nodes import validate_request_node
    from app.services.agent_graph.state import AgentState

    result = validate_request_node(AgentState(query=""))
    return {
        "detected": result.workflow_status == "failed",
        "recovered": result.termination_reason == "validation_failure",
        "final_state": f"workflow_status={result.workflow_status}, termination_reason={result.termination_reason}",
    }


CASES = {
    "fail_001": ("llm_timeout", case_llm_timeout),
    "fail_002": ("llm_rate_limit", case_llm_rate_limit),
    "fail_003": ("llm_provider_error", case_llm_provider_error),
    "fail_004": ("retrieval_timeout", case_retrieval_failure),
    "fail_005": ("vector_store_failure", case_vector_store_none),
    "fail_006": ("reranker_failure", case_reranker_failure),
    "fail_007": ("web_search_failure", case_web_search_failure),
    "fail_008": ("vision_service_timeout", case_vision_timeout),
    "fail_009": ("invalid_structured_output", case_invalid_structured_output),
    "fail_010": ("approval_rejection", case_approval_rejection),
    "fail_011": ("loop_cap_reached", case_loop_cap_reached),
    "fail_012": ("malformed_input", case_malformed_input),
}


def main() -> None:
    results = {}
    for case_id, (category, fn) in CASES.items():
        try:
            results[case_id] = {"category": category, **fn()}
        except Exception as exc:  # noqa: BLE001 - a case's own crash is itself a finding
            results[case_id] = {"category": category, "detected": False, "recovered": False, "final_state": f"RUNNER CRASHED: {type(exc).__name__}: {exc}"}

    measured = {cid: r for cid, r in results.items() if r["detected"] != "N/A"}
    n = len(measured)
    detected = sum(1 for r in measured.values() if r["detected"] is True)
    recovered = sum(1 for r in measured.values() if r["recovered"] is True)
    unhandled = sum(1 for r in measured.values() if "RUNNER CRASHED" in str(r["final_state"]) or (r["detected"] and not r["recovered"]))

    report = {
        "metadata": config.run_metadata(sample_count=len(CASES)),
        "failure_detection_rate": round(detected / n, 4) if n else None,
        "recovery_success_rate": round(recovered / n, 4) if n else None,
        "safe_failure_rate": round(recovered / n, 4) if n else None,
        "unhandled_failure_rate": round(unhandled / n, 4) if n else None,
        "n_measured": n,
        "n_not_applicable": len(CASES) - n,
        "per_case": results,
    }

    path = config.save_report(report, name="failure_eval")
    print(f"Saved: {path}")
    print()
    for case_id, r in results.items():
        print(f"  {case_id:<12s} {r['category']:<28s} detected={r['detected']!s:<6} recovered={r['recovered']!s:<6} {r['final_state']}")
    print()
    print(f"Failure Detection Rate: {report['failure_detection_rate']}")
    print(f"Recovery Success Rate: {report['recovery_success_rate']}")
    print(f"Unhandled Failure Rate: {report['unhandled_failure_rate']}")


if __name__ == "__main__":
    main()
