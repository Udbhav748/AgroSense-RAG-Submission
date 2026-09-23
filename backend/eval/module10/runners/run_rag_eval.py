#!/usr/bin/env python
"""Module 10 RAG evaluation runner.

Usage (from backend/):
    python eval/module10/runners/run_rag_eval.py
    python eval/module10/runners/run_rag_eval.py --ablation-only
    python eval/module10/runners/run_rag_eval.py --case rag_001

Runs rag_eval.json's cases through the real ChatService against the live
indexed vector store, computing:
- Precision@5 / Recall@5 / Hit Rate@5 / MRR (module10/metrics/retrieval.py,
  which re-exports run_eval.py's existing implementations)
- lexical + claim-decomposition groundedness (module10/metrics/groundedness.py)
- citation accuracy (run_eval.py's citation_supported, re-exported)
- a 3-way retrieval ablation: semantic-only / hybrid / hybrid+rerank, by
  toggling Settings.hybrid_search_enabled/reranking_enabled for the SAME
  dataset in the SAME process

This is a LIVE evaluation: it calls the real embedding model and the real
configured LLM provider (Settings.llm_provider). No live outage is caused,
but real API quota is used for generation/groundedness-judge calls.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # backend/

from app.core.config import settings  # noqa: E402
from app.services.faiss_vector_store import FAISSVectorStore  # noqa: E402
from app.services.llm_provider import build_llm_client  # noqa: E402
from app.services.rag_service import ChatService  # noqa: E402
from eval.module10 import config  # noqa: E402
from eval.module10.metrics import groundedness as gmetrics  # noqa: E402
from eval.module10.metrics import retrieval as rmetrics  # noqa: E402
from eval.run_eval import citation_supported  # noqa: E402


def _run_one_configuration(chat_service: ChatService, cases: list[dict], *, config_label: str) -> dict:
    # The response cache's key (query+crop+disease+tenant+document_ids) does
    # NOT include hybrid_search_enabled/reranking_enabled — by design, the
    # cache is meant to survive config-invariant repeats, not to distinguish
    # ablation configurations. Left enabled, the second and third configs in
    # a 3-way ablation over the SAME query set would silently replay the
    # first config's cached answer instead of actually re-running retrieval,
    # making the ablation numbers meaningless. Disabled for the lifetime of
    # this evaluation run only (an eval-mode choice, not a production change).
    chat_service._get_cached_response = lambda *a, **k: None  # noqa: SLF001
    retrieval_results = []
    groundedness_results = []
    citation_flags = []
    per_case = []

    for case in cases:
        case_id = case["id"]
        query = case["query"]
        keywords = case.get("expected_chunk_keywords", [])

        if not query:
            per_case.append({"case_id": case_id, "skipped": "empty query"})
            continue

        try:
            response = chat_service.handle_query(query)
        except Exception as exc:  # noqa: BLE001 - eval must not crash on one bad case
            per_case.append({"case_id": case_id, "error": f"{type(exc).__name__}: {exc}"})
            continue

        chunks = response.retrieved_chunks
        retrieval_result = rmetrics.score_case(case_id, chunks, keywords)
        retrieval_results.append(retrieval_result)

        evidence_texts = [c.text for c in chunks] + case.get("expected_facts", [])
        claim_result = gmetrics.claim_decomposition_groundedness(response.answer, evidence_texts)
        lexical = gmetrics.lexical_groundedness(response.answer, chunks)

        cite_ok = citation_supported(response.sources, keywords) if keywords else None
        if cite_ok is not None:
            citation_flags.append(cite_ok)

        groundedness_results.append(
            {
                "case_id": case_id,
                "lexical_grounded": lexical,
                "claim_decomposition": {
                    "total_claims": claim_result.total_claims,
                    "supported_claims": claim_result.supported_claims,
                    "groundedness_rate": claim_result.groundedness_rate,
                    "unsupported_claims": claim_result.unsupported,
                },
            }
        )

        per_case.append(
            {
                "case_id": case_id,
                "type": case.get("type"),
                "query": query,
                "answer": response.answer,
                "answer_source": response.answer_source,
                "retrieval_confidence": response.retrieval_confidence,
                "precision_at_5": retrieval_result.precision_at_5,
                "recall_at_5": retrieval_result.recall_at_5,
                "hit_at_5": retrieval_result.hit_at_5,
                "reciprocal_rank": retrieval_result.reciprocal_rank,
                "lexical_grounded": lexical,
                "claim_groundedness_rate": claim_result.groundedness_rate,
                "citation_supported": cite_ok,
                "expected_fallback": case.get("expected_fallback", False),
            }
        )

    retrieval_agg = rmetrics.aggregate(retrieval_results)
    claim_rates = [
        r["claim_decomposition"]["groundedness_rate"]
        for r in groundedness_results
        if r["claim_decomposition"]["groundedness_rate"] is not None
    ]
    lexical_rate = (
        sum(1 for r in groundedness_results if r["lexical_grounded"]) / len(groundedness_results)
        if groundedness_results
        else None
    )

    return {
        "configuration": config_label,
        "retrieval": retrieval_agg,
        "groundedness": {
            "lexical_groundedness_rate": round(lexical_rate, 4) if lexical_rate is not None else None,
            "mean_claim_groundedness_rate": round(sum(claim_rates) / len(claim_rates), 4)
            if claim_rates
            else None,
            "n": len(groundedness_results),
        },
        "citation_accuracy": round(sum(citation_flags) / len(citation_flags), 4) if citation_flags else None,
        "n_cases": len(cases),
        "per_case": per_case,
    }


def run_ablation(cases: list[dict]) -> dict:
    """Semantic-only / hybrid / hybrid+rerank, same dataset, same process."""
    vector_store = FAISSVectorStore()
    vector_store.load()
    llm_client = build_llm_client()

    original_hybrid = settings.hybrid_search_enabled
    original_rerank = settings.reranking_enabled
    results = {}
    try:
        for label, hybrid, rerank in [
            ("semantic_only", False, False),
            ("hybrid", True, False),
            ("hybrid_plus_rerank", True, True),
        ]:
            settings.hybrid_search_enabled = hybrid
            settings.reranking_enabled = rerank
            chat_service = ChatService(vector_store, llm_client)
            results[label] = _run_one_configuration(chat_service, cases, config_label=label)
    finally:
        settings.hybrid_search_enabled = original_hybrid
        settings.reranking_enabled = original_rerank

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation-only", action="store_true", help="Skip the single full-config run, only run the 3-way ablation")
    parser.add_argument("--case", help="Run only this case id")
    args = parser.parse_args()

    dataset = config.load_dataset("rag_eval.json")
    cases = dataset["cases"]
    if args.case:
        cases = [c for c in cases if c["id"] == args.case]
        if not cases:
            raise SystemExit(f"No case with id {args.case!r} in rag_eval.json")

    ablation = run_ablation(cases)

    report = {
        "metadata": config.run_metadata(sample_count=len(cases), dataset_version=dataset["dataset_version"]),
        "ablation": ablation,
    }

    path = config.save_report(report, name="rag_eval")
    print(f"Saved: {path}")
    print()
    print(f"{'Configuration':<20} {'P@5':>8} {'Recall@5':>10} {'Hit@5':>8} {'MRR':>8} {'Groundedness(lex)':>18} {'Citation':>10}")
    for label, result in ablation.items():
        r = result["retrieval"]
        g = result["groundedness"]
        p5 = f"{r['mean_precision_at_5']:.4f}" if r["mean_precision_at_5"] is not None else "n/a"
        rec5 = f"{r['mean_recall_at_5']:.4f}" if r["mean_recall_at_5"] is not None else "n/a"
        hit5 = f"{r['hit_rate_at_5']:.4f}" if r["hit_rate_at_5"] is not None else "n/a"
        mrr = f"{r['mrr']:.4f}" if r["mrr"] is not None else "n/a"
        ground = f"{g['lexical_groundedness_rate']:.4f}" if g["lexical_groundedness_rate"] is not None else "n/a"
        cite = f"{result['citation_accuracy']:.4f}" if result["citation_accuracy"] is not None else "n/a"
        print(f"{label:<20} {p5:>8} {rec5:>10} {hit5:>8} {mrr:>8} {ground:>18} {cite:>10}")


if __name__ == "__main__":
    main()
