#!/usr/bin/env python
"""Module 10 gap-closure: real NLI-based (entailment) groundedness
scoring, alongside the existing lexical-overlap compute_faithfulness
proxy, on the same 20-case golden dataset -- computed side by side so
the upgrade can be judged directly against the metric it improves on.

No model is trained here -- a pretrained cross-encoder NLI model
(cross-encoder/nli-MiniLM2-L6-H768) is used purely for inference, the
same posture as this project's existing reranker
(cross-encoder/ms-marco-MiniLM-L-6-v2). See
eval/module10/metrics/nli_faithfulness.py's module docstring for the
full rationale.

Reuses scripts/run_rag_eval.py's existing, already-fixed retrieval and
generation functions (execute_retrieval, execute_answer_generation) --
does not reimplement retrieval or generation, and does not modify
run_rag_eval.py's own dataclasses/report shape, so none of its existing
pinned tests are touched by this addition.

Usage (from backend/):
    python eval/module10/runners/run_nli_faithfulness_eval.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from app.services.llm_provider import build_llm_client  # noqa: E402
from app.services.rag_service import ChatService  # noqa: E402
from eval.module10 import config  # noqa: E402
from eval.module10.metrics.nli_faithfulness import (  # noqa: E402
    NLI_MODEL_NAME,
    compute_nli_faithfulness,
)
from scripts.run_rag_eval import (  # noqa: E402
    GOLDEN_DATASET,
    _load_vector_store,
    compute_faithfulness,
    execute_answer_generation,
    execute_retrieval,
)


def main() -> None:
    vector_store = _load_vector_store()
    chat_service = ChatService(vector_store=vector_store, llm_client=build_llm_client())

    per_case = []
    for case in GOLDEN_DATASET:
        start = time.perf_counter()
        chunks = execute_retrieval(
            query=case["query"], top_k=8, crop=case.get("crop"), vector_store=vector_store,
            hybrid=True, rerank_flag=True,
        )
        answer = execute_answer_generation(case["query"], chunks, chat_service=chat_service)
        latency_s = time.perf_counter() - start

        lexical_score, lexical_supported, lexical_total = compute_faithfulness(answer, chunks)
        nli_result = compute_nli_faithfulness(answer, chunks)

        per_case.append(
            {
                "id": case["id"],
                "crop": case.get("crop"),
                "disease": case.get("disease"),
                "query": case["query"],
                "retrieved_chunks_count": len(chunks),
                "lexical_faithfulness": round(lexical_score, 4),
                "lexical_supported_claims": lexical_supported,
                "lexical_total_claims": lexical_total,
                "nli_faithfulness": nli_result["nli_faithfulness"],
                "nli_supported_claims": nli_result["supported_claims"],
                "nli_total_claims": nli_result["total_claims"],
                "delta_nli_minus_lexical": round(nli_result["nli_faithfulness"] - lexical_score, 4),
                "nli_per_claim": nli_result["per_claim"],
                "latency_s": round(latency_s, 4),
                "generated_answer": answer,
            }
        )
        print(
            f"{case['id']}: lexical={lexical_score:.4f} nli={nli_result['nli_faithfulness']:.4f} "
            f"delta={nli_result['nli_faithfulness'] - lexical_score:+.4f}"
        )

    n = len(per_case)
    mean_lexical = round(sum(c["lexical_faithfulness"] for c in per_case) / n, 4)
    mean_nli = round(sum(c["nli_faithfulness"] for c in per_case) / n, 4)

    report = {
        **config.run_metadata(sample_count=n, dataset_version="run_rag_eval_golden_v1_20cases"),
        "evaluation_name": "nli_faithfulness_upgrade",
        "nli_model": NLI_MODEL_NAME,
        "nli_model_trained_by_this_project": False,
        "methodology": (
            "Real premise->hypothesis entailment inference via a pretrained cross-encoder NLI "
            "model, scored per-claim against EACH retrieved chunk individually (max entailment "
            "probability kept per claim, not one concatenated context) -- the standard "
            "'is this claim supported by ANY retrieved chunk' RAG-groundedness definition. "
            "Compared directly against the existing lexical-overlap compute_faithfulness proxy "
            "on the identical answer/retrieved-chunks pair per case."
        ),
        "mean_lexical_faithfulness": mean_lexical,
        "mean_nli_faithfulness": mean_nli,
        "mean_delta_nli_minus_lexical": round(mean_nli - mean_lexical, 4),
        "per_case": per_case,
        "honest_finding_domain_mismatch": {
            "summary": (
                "Real experimentation found that generic pretrained NLI models are poorly "
                "calibrated for this project's structured, pipe-delimited retrieval-chunk "
                "format -- they were trained on clean, single-topic natural-sentence pairs "
                "(SNLI/MultiNLI), not 'structured database record -> derived claim' pairs. "
                "The mean_nli_faithfulness figure above is measured honestly, but it is NOT "
                "a more trustworthy groundedness signal than the lexical proxy for this "
                "corpus -- it is a differently-flawed, systematically-too-harsh score, not a "
                "clean upgrade."
            ),
            "concrete_example": (
                "eval-corn-01, claim 'The recommended fungicides for northern corn leaf "
                "blight are Propiconazole (25% EC) and Azoxystrobin (23% SC)' against the "
                "real retrieved chunk containing 'Chemical Active Ingredient: Propiconazole "
                "25% EC or Azoxystrobin 23% SC' -- a claim a human would clearly judge as "
                "grounded. Verified no truncation (164/512 tokens) and the needed text is "
                "present in what the model sees."
            ),
            "models_and_premise_formats_tested": [
                {"model": "cross-encoder/nli-MiniLM2-L6-H768 (used for this report's scores)", "premise_format": "raw chunk text", "entailment_probability": 0.0607},
                {"model": "cross-encoder/nli-deberta-v3-base (larger, more capable)", "premise_format": "raw chunk text", "entailment_probability": 0.0028, "note": "classified as neutral, not even contradiction -- a bigger pretrained model does not fix this"},
                {"model": "cross-encoder/nli-MiniLM2-L6-H768", "premise_format": "naturalized ('Key is Value.' sentences instead of pipe-delimited)", "entailment_probability": 0.0225, "note": "made it worse, not better"},
                {"model": "cross-encoder/nli-MiniLM2-L6-H768", "premise_format": "single isolated relevant field only", "entailment_probability": 0.0032, "note": "worse -- the model needs some surrounding context (e.g. disease name) to verify subject-matching"},
                {"model": "cross-encoder/nli-MiniLM2-L6-H768", "premise_format": "hand-picked 2-field subset (disease + chemical active ingredient only, other fields removed)", "entailment_probability": 0.4892, "note": "closest to the 0.5 threshold, but required manually choosing which fields to keep per claim -- a fragile, corpus-specific heuristic, not a general, principled fix"},
            ],
            "conclusion": (
                "No training was attempted (per this project's explicit scope decision -- a "
                "generic pretrained model was always the intended approach, not a custom-"
                "trained one). No simple, general preprocessing fix was found either. This is "
                "reported as a genuine, disclosed negative result: the lexical-overlap proxy "
                "remains the project's PRIMARY faithfulness metric; nli_faithfulness above is "
                "an additive, honestly-measured, but NOT more trustworthy signal for this "
                "corpus's structured-data retrieval format. A reliable NLI-based upgrade would "
                "likely need either a model fine-tuned for this exact premise style, or a much "
                "more sophisticated (non-fragile) premise-construction step than tried here -- "
                "both out of scope for this pass."
            ),
        },
        "limitations": [
            "n=20 cases, one domain/corpus -- not a claim that generalizes beyond this "
            "specific evaluation set.",
            "See honest_finding_domain_mismatch above: the NLI score is NOT recommended as a "
            "replacement for the lexical faithfulness proxy on this project's corpus -- both "
            "are reported, neither is claimed superior.",
            "Entailment threshold (0.5) is the natural softmax decision boundary, not tuned "
            "against this project's own dataset.",
            "Per-claim sentence splitting reuses compute_faithfulness's own regex-based "
            "splitter -- inherits whatever edge cases that has (e.g. abbreviations, decimal "
            "numbers) for both metrics identically.",
            "This is an additive metric alongside the existing lexical proxy, not a "
            "replacement -- scripts/run_rag_eval.py's own compute_faithfulness and its "
            "extensive existing test coverage are unchanged.",
        ],
    }

    path = config.save_report(report, name="nli_faithfulness_upgrade")
    print(f"\nSaved: {path}")
    print(f"Mean lexical faithfulness: {mean_lexical}")
    print(f"Mean NLI faithfulness:     {mean_nli}")
    print(f"Mean delta (NLI - lexical): {mean_nli - mean_lexical:+.4f}")


if __name__ == "__main__":
    main()
