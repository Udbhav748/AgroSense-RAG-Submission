#!/usr/bin/env python
"""Module 10 memory evaluation runner.

Usage (from backend/):
    python eval/module10/runners/run_memory_eval.py

retention_cases / irrelevance_cases: live ChatService.handle_query calls
with explicit history (real LLM call, measures whether memory actually
changes generation). session_boundary_cases / cross_session_leakage_cases:
deterministic, no LLM — exercise InMemorySessionStore / AgentMemory
directly, reusing the same fixture style as
backend/tests/test_agent_memory.py / test_memory_bounds.py rather than
re-deriving new assertions.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from app.services.agent_memory import AgentMemory  # noqa: E402
from app.services.faiss_vector_store import FAISSVectorStore  # noqa: E402
from app.services.llm_provider import build_llm_client  # noqa: E402
from app.services.rag_service import ChatService  # noqa: E402
from app.services.session_store import InMemorySessionStore  # noqa: E402
from eval.module10 import config  # noqa: E402


def run_retention_and_irrelevance(chat_service: ChatService, cases: list[dict]) -> dict:
    results = []
    for case in cases:
        response = chat_service.handle_query(case["query"], history=case["history"])
        answer_lower = response.answer.lower()
        recalled = all(kw.lower() in answer_lower for kw in case.get("expected_answer_contains", []))
        leaked_irrelevant = any(
            kw.lower() in answer_lower for kw in case.get("expected_answer_not_contains", [])
        )
        results.append(
            {
                "case_id": case["id"],
                "query": case["query"],
                "answer": response.answer,
                "recalled_expected": recalled,
                "leaked_irrelevant_content": leaked_irrelevant,
                "pass": recalled and not leaked_irrelevant,
            }
        )
    n = len(results)
    recall_rate = sum(1 for r in results if r["recalled_expected"]) / n if n else None
    irrelevance_pass_rate = (
        sum(1 for r in results if not r["leaked_irrelevant_content"]) / n if n else None
    )
    return {
        "memory_recall_rate": round(recall_rate, 4) if recall_rate is not None else None,
        "irrelevance_pass_rate": round(irrelevance_pass_rate, 4) if irrelevance_pass_rate is not None else None,
        "n": n,
        "per_case": results,
    }


def run_session_boundary_cases() -> dict:
    store = InMemorySessionStore(max_sessions=10, max_turns_per_session=6)
    session_a = store.get_or_create_session(None)
    for i in range(8):
        store.append_turn(session_a, "user", f"fact-{i}: the value is {i}")
        store.append_turn(session_a, "assistant", f"acknowledged {i}")
    history_a = store.get_history(session_a) or []

    session_b = store.get_or_create_session(None)
    history_b = store.get_history(session_b)

    return {
        "last_n_turns": {
            "turns_retained": len(history_a),
            "max_turns_per_session": 6,
            "pass": len(history_a) <= 6,
        },
        "session_separation": {
            "session_b_sees_session_a_history": bool(history_b),
            "pass": not history_b,
        },
    }


def run_cross_session_leakage_case() -> dict:
    memory = AgentMemory(max_sessions=10)
    session_a = "eval-session-a"
    session_b = "eval-session-b"
    memory.upsert_fact(session_a, "planted_secret", "ZX-4471-Q", source_chunk_ids=[])

    context_b = memory.build_context(session_b)
    leaked = "ZX-4471-Q" in (context_b or "")

    return {
        "cross_session_leak_rate": 1.0 if leaked else 0.0,
        "leaked": leaked,
        "expected": 0,
        "pass": not leaked,
    }


def main() -> None:
    dataset = config.load_dataset("memory_eval.json")

    vector_store = FAISSVectorStore()
    vector_store.load()
    chat_service = ChatService(vector_store, build_llm_client())

    retention_result = run_retention_and_irrelevance(chat_service, dataset["retention_cases"])
    irrelevance_result = run_retention_and_irrelevance(chat_service, dataset["irrelevance_cases"])
    session_boundary_result = run_session_boundary_cases()
    cross_session_result = run_cross_session_leakage_case()

    total = len(dataset["retention_cases"]) + len(dataset["irrelevance_cases"])
    report = {
        "metadata": config.run_metadata(sample_count=total, dataset_version=dataset["dataset_version"]),
        "retention": retention_result,
        "irrelevance": irrelevance_result,
        "session_boundary": session_boundary_result,
        "cross_session_leakage": cross_session_result,
    }

    path = config.save_report(report, name="memory_eval")
    print(f"Saved: {path}")
    print()
    print(f"Memory Recall Rate: {retention_result['memory_recall_rate']}")
    print(f"Irrelevance Pass Rate: {irrelevance_result['irrelevance_pass_rate']}")
    print(f"Session boundary: {session_boundary_result}")
    print(f"Cross-session leak rate: {cross_session_result['cross_session_leak_rate']}")


if __name__ == "__main__":
    main()
