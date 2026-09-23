#!/usr/bin/env python
"""Phase 5, Step 3 (mandatory): live post-fix Faithfulness measurement.

Re-runs the exact two human-evaluation queries (rows 17, 19 in
docs/HUMAN_EVAL.md) that were the strongest evidence of the Faithfulness
regression Phase 3 fixed (generator_node laundering LLM provider failures
into FALLBACK_REPLY). Captures full evidence for each: query, retrieved
chunks, retrieval grade, final answer, whether the answer is the new
GENERATION_ERROR_REPLY sentinel or a real generated answer, latency, and
token/cost telemetry.

SCOPE, STATED HONESTLY: only these two cases are re-run live, not the full
20-case RAG benchmark or the full 24-case human evaluation. This is a
deliberate choice to conserve the Groq API key's daily token quota (fully
exhausted once already during Phase 3's investigation) for future work,
per this phase's own instruction: "If complete RAG re-run is too
expensive, explicitly state the limited scope and do not extrapolate."
No claim is made here about the full 20/24-case Faithfulness number.

Usage (from backend/):
    python eval/module10/runners/run_faithfulness_post_phase3.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from app.core.config import settings  # noqa: E402
from app.services.faiss_vector_store import FAISSVectorStore  # noqa: E402
from app.services.llm_provider import build_llm_client  # noqa: E402
from app.services.prompt_builder import FALLBACK_REPLY, GENERATION_ERROR_REPLY, PROMPT_VERSION  # noqa: E402
from app.services.rag_service import ChatService  # noqa: E402
from eval.module10 import config  # noqa: E402

CASES = [
    {
        "row": 17,
        "id": "faithfulness_post_phase3_row17",
        "query": "What's the treatment for scab?",
        "case_type": "hard/ambiguous",
        "note": "Retrieval previously confirmed excellent (5 apple-scab-specific chunks, "
        "including exact dosage/active-ingredient text) via prompt capture in Phase 3's "
        "investigation. Pre-fix behavior: FALLBACK_REPLY ('I couldn't find that information...') "
        "despite the answer being present in context -- the core regression symptom.",
    },
    {
        "row": 19,
        "id": "faithfulness_post_phase3_row19",
        "query": "What is the treatment for citrus greening disease (HLB)?",
        "case_type": "hard/out_of_corpus (stale label -- see docs/HUMAN_EVAL.md row 19)",
        "note": "Retrieval previously confirmed excellent (full HLB diagnostic guide + dosage "
        "reference) despite hard_cases.json's stale 'out of corpus' label. Pre-fix behavior: "
        "FALLBACK_REPLY despite on-topic retrieval.",
    },
]


def run_case(chat_service: ChatService, case: dict) -> dict:
    start = time.perf_counter()
    response = chat_service.handle_query(case["query"])
    latency_s = time.perf_counter() - start

    is_generation_error = response.answer.strip() == GENERATION_ERROR_REPLY
    is_fallback = response.answer.strip() == FALLBACK_REPLY
    is_real_answer = not is_generation_error and not is_fallback

    return {
        "row": case["row"],
        "case_id": case["id"],
        "case_type": case["case_type"],
        "note": case["note"],
        "query": case["query"],
        "retrieved_chunk_count": len(response.retrieved_chunks),
        "retrieval_confidence": response.retrieval_confidence,
        "sources": [
            {"document_id": s.document_id, "excerpt": (s.excerpt or "")[:200]}
            for s in (response.sources or [])
        ],
        "answer": response.answer,
        "answer_classification": (
            "generation_error_replay" if is_generation_error
            else "fallback_not_in_documents" if is_fallback
            else "real_generated_answer"
        ),
        "is_real_answer": is_real_answer,
        "hallucination_detected": response.hallucination_detected,
        "grounding_score": response.grounding_score,
        "tool_used": response.tool_used,
        "steps_taken": response.steps_taken,
        "answer_source": response.answer_source,
        "latency_s": round(latency_s, 4),
        "processing_time_reported_s": response.processing_time,
    }


def main() -> None:
    vector_store = FAISSVectorStore()
    vector_store.load()
    chat_service = ChatService(vector_store, build_llm_client())

    results = [run_case(chat_service, case) for case in CASES]

    n_real = sum(1 for r in results if r["is_real_answer"])
    report = {
        "metadata": {
            **config.run_metadata(sample_count=len(CASES), dataset_version="human_eval_rows_17_19"),
            "prompt_version": PROMPT_VERSION,
            "llm_provider": settings.llm_provider,
            "groq_model_name": getattr(settings, "groq_model_name", None),
            "retrieval_config": {
                "hybrid_search_enabled": settings.hybrid_search_enabled,
                "reranking_enabled": settings.reranking_enabled,
                "retrieval_top_k": settings.retrieval_top_k,
                "retrieval_grade_threshold": settings.retrieval_grade_threshold,
            },
        },
        "scope_note": "LIMITED SCOPE, STATED HONESTLY: only the 2 previously-failing human-eval "
        "cases (rows 17, 19) were re-run live in this pass, to conserve the Groq daily token "
        "quota after it was fully exhausted once already during Phase 3's investigation. This is "
        "NOT a full 20-case RAG benchmark or 24-case human-evaluation re-run, and the result below "
        "must not be extrapolated as the new full-dataset Faithfulness score.",
        "pre_fix_reference": "Both cases previously returned FALLBACK_REPLY ('I couldn't find that "
        "information in the uploaded documents.') despite retrieval confirmed correct -- see "
        "docs/PHASE3_PRODUCTION_HARDENING_REPORT.md Section D and docs/HUMAN_EVAL.md rows 17/19.",
        "summary": {
            "n_cases": len(results),
            "n_real_answers_post_fix": n_real,
            "n_generation_error_replies": sum(1 for r in results if r["answer_classification"] == "generation_error_replay"),
            "n_fallback_not_in_documents": sum(1 for r in results if r["answer_classification"] == "fallback_not_in_documents"),
        },
        "per_case": results,
    }

    path = config.save_report(report, name="faithfulness_post_phase3")
    print(f"Saved: {path}")
    print()
    for r in results:
        print(f"Row {r['row']} ({r['case_id']}): {r['answer_classification']}")
        print(f"  answer: {r['answer'][:200]}")
        print(f"  chunks={r['retrieved_chunk_count']} confidence={r['retrieval_confidence']} latency={r['latency_s']}s")
        print()
    print(f"Summary: {n_real}/{len(results)} real generated answers post-fix.")


if __name__ == "__main__":
    main()
