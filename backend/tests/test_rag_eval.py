"""Unit tests for the Quantitative RAG Evaluation & Benchmarking suite.

Tests metrics calculation engines (Context Recall, Context Precision,
Faithfulness, Answer Relevance, Harmonic Composite), golden dataset validity,
ASCII scorecard formatting, report persistence, and CLI execution.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Ensure backend root is on sys.path
_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

import pytest

from scripts.run_rag_eval import (
    GOLDEN_DATASET,
    BenchmarkReport,
    ItemEvalResult,
    compute_answer_relevance,
    compute_context_precision,
    compute_context_recall,
    compute_faithfulness,
    compute_harmonic_composite,
    format_ascii_scorecard,
    main,
    parse_args,
    run_evaluation,
    save_eval_report,
)


# ===========================================================================
# 1. Dataset Integrity Tests
# ===========================================================================

def test_golden_dataset_count_and_keys():
    """Verify golden dataset contains 20 comprehensive plant pathology items."""
    assert len(GOLDEN_DATASET) == 20

    required_keys = {
        "id",
        "crop",
        "disease",
        "query",
        "ground_truth_answer",
        "expected_active_ingredients",
        "expected_organic_remedies",
    }

    for item in GOLDEN_DATASET:
        missing = required_keys - set(item.keys())
        assert not missing, f"Item {item.get('id')} missing keys: {missing}"
        assert len(item["query"].strip()) > 10
        assert len(item["ground_truth_answer"].strip()) > 20
        assert isinstance(item["expected_active_ingredients"], list)
        assert isinstance(item["expected_organic_remedies"], list)


def test_golden_dataset_crops_coverage():
    """Verify all 7 major crop categories are covered."""
    crops = {item["crop"].lower() for item in GOLDEN_DATASET}
    expected_crops = {"tomato", "potato", "apple", "corn", "grape", "orange", "bell pepper"}
    for crop in expected_crops:
        assert crop in crops, f"Expected crop '{crop}' not found in dataset"


# ===========================================================================
# 2. Metric Engine: Context Recall Tests
# ===========================================================================

def test_context_recall_perfect():
    """Verify 1.0 recall when all ingredients and remedies are present."""
    expected_active = ["Chlorothalonil", "Azoxystrobin"]
    expected_organic = ["Copper octanoate", "Bacillus subtilis"]
    chunks = [
        "Chlorothalonil 75% WP and Azoxystrobin 23% SC provide control.",
        "Organic sprays include Copper octanoate and Bacillus subtilis bio-fungicide.",
    ]

    recall, matched, missing = compute_context_recall(expected_active, expected_organic, chunks)
    assert recall == 1.0
    assert len(matched) == 4
    assert len(missing) == 0


def test_context_recall_partial_and_zero():
    """Verify partial and zero recall computations."""
    expected_active = ["Mandipropamid", "Cyazofamid", "Cymoxanil"]
    expected_organic = ["Copper hydroxide"]
    # Only Mandipropamid and Copper hydroxide present (2 out of 4)
    chunks = ["Mandipropamid is used for downy mildew and late blight with Copper hydroxide."]

    recall, matched, missing = compute_context_recall(expected_active, expected_organic, chunks)
    assert recall == 0.50
    assert "Mandipropamid" in matched
    assert "Copper hydroxide" in matched
    assert "Cyazofamid" in missing
    assert "Cymoxanil" in missing

    # Zero recall
    zero_chunks = ["No relevant information about apples."]
    zero_recall, _, _ = compute_context_recall(expected_active, expected_organic, zero_chunks)
    assert zero_recall == 0.0


def test_context_recall_empty_cases():
    """Verify graceful handling of empty chunks or 'healthy/None' expectations."""
    # Empty chunks -> 0.0
    recall, _, missing = compute_context_recall(["Captan"], [], [])
    assert recall == 0.0
    assert "Captan" in missing

    # None / healthy expectation -> 1.0
    recall_healthy, _, _ = compute_context_recall(["None"], ["Compost tea"], ["Apply compost tea regularly."])
    assert recall_healthy == 1.0


# ===========================================================================
# 3. Metric Engine: Context Precision Tests
# ===========================================================================

def test_context_precision_all_relevant():
    """Verify precision is 1.0 when all retrieved chunks are relevant."""
    active = ["Difenoconazole"]
    organic = ["Potassium bicarbonate"]
    chunks = [
        "Difenoconazole 25% EC is effective against leaf mold.",
        "Potassium bicarbonate spray controls fungal hyphae.",
    ]
    precision = compute_context_precision(active, organic, "leaf mold", chunks)
    assert precision == 1.0


def test_context_precision_rank_weighted():
    """Verify rank weighting (Average Precision at K)."""
    active = ["Myclobutanil"]
    organic = ["Sulfur"]
    disease = "apple scab"

    # Chunk 1 is relevant (Myclobutanil), Chunk 2 is irrelevant
    # Ranks: [rel, non-rel] -> Precision@1 = 1/1 = 1.0. AP = 1.0 / 1 = 1.0
    chunks_top_rel = [
        "Myclobutanil 20% WP controls cedar apple rust.",
        "General soil irrigation guidelines for orchards.",
    ]
    prec_top = compute_context_precision(active, organic, disease, chunks_top_rel)
    assert prec_top == 1.0

    # Chunk 1 is irrelevant, Chunk 2 is relevant (Myclobutanil)
    # Ranks: [non-rel, rel] -> Precision@2 = 1/2 = 0.5. AP = 0.5 / 1 = 0.5
    chunks_bottom_rel = [
        "General soil irrigation guidelines for orchards.",
        "Myclobutanil 20% WP controls cedar apple rust.",
    ]
    prec_bottom = compute_context_precision(active, organic, disease, chunks_bottom_rel)
    assert prec_bottom == 0.5


def test_context_precision_no_relevant_or_empty():
    """Verify precision is 0.0 when no chunks are relevant or list is empty."""
    precision_empty = compute_context_precision(["Captan"], [], "scab", [])
    assert precision_empty == 0.0

    chunks_irrelevant = ["General tractor maintenance tips.", "Weather forecast for tomorrow."]
    precision_irrel = compute_context_precision(["Captan"], ["Sulfur"], "scab", chunks_irrelevant)
    assert precision_irrel == 0.0


# ===========================================================================
# 4. Metric Engine: Faithfulness Tests
# ===========================================================================

def test_faithfulness_grounded():
    """Verify high faithfulness score when answer is grounded in retrieved chunks."""
    chunks = [
        "Tomato early blight is treated with Chlorothalonil 75% WP and Azoxystrobin.",
        "Organic remedies include copper octanoate and Bacillus subtilis.",
    ]
    answer = (
        "Early blight on tomatoes is treated with Chlorothalonil 75% WP and Azoxystrobin. "
        "Organic controls include copper octanoate and Bacillus subtilis."
    )
    score, supported, total = compute_faithfulness(answer, chunks)
    assert score == 1.0
    assert supported == total
    assert total > 0


def test_faithfulness_hallucinated():
    """Verify lower faithfulness when answer contains ungrounded claims."""
    chunks = ["Tomato bacterial spot is managed with fixed copper sprays."]
    answer = (
        "Spray synthetic antibiotic streptomycin at high doses. "
        "Also inject penicillin directly into the vascular tissue."
    )
    score, supported, total = compute_faithfulness(answer, chunks)
    assert score < 0.5


def test_faithfulness_empty():
    """Verify edge cases for faithfulness."""
    score, _, _ = compute_faithfulness("", ["Some chunk"])
    assert score == 0.0

    score_no_chunks, _, _ = compute_faithfulness("A valid answer.", [])
    assert score_no_chunks == 0.0


# ===========================================================================
# 5. Metric Engine: Answer Relevance Tests
# ===========================================================================

def test_answer_relevance_matching():
    """Verify high relevance for semantically aligned query and answer."""
    query = "What fungicides manage apple scab?"
    answer = "Apple scab is managed using Captan 50% WP or Difenoconazole fungicides."
    score = compute_answer_relevance(query, answer)
    assert 0.5 <= score <= 1.0


def test_answer_relevance_unrelated():
    """Verify low relevance score for completely unrelated answer."""
    query = "What fungicides manage apple scab?"
    answer = "The boiling point of water at sea level is 100 degrees Celsius."
    score = compute_answer_relevance(query, answer)
    assert score < 0.55


def test_answer_relevance_custom_embed_fn():
    """Verify answer relevance with mock embedding function."""
    def mock_embed(text: str) -> list[float]:
        if "apple" in text.lower():
            return [1.0, 0.0, 0.0]
        return [0.0, 1.0, 0.0]

    q = "apple disease"
    a = "apple scab treatment"
    score = compute_answer_relevance(q, a, embed_fn=mock_embed)
    assert score == 1.0


# ===========================================================================
# 6. Metric Engine: Harmonic Composite Tests
# ===========================================================================

def test_harmonic_composite_full_mode():
    """Verify 4-dimension composite calculation in full mode."""
    composite = compute_harmonic_composite(
        faithfulness=0.90,
        context_recall=0.85,
        context_precision=0.80,
        answer_relevance=0.95,
    )
    assert 0.0 <= composite <= 1.0
    assert composite > 0.80


def test_harmonic_composite_retrieval_only_mode():
    """Verify retrieval-only composite calculation when LLM metrics are None."""
    composite = compute_harmonic_composite(
        faithfulness=None,
        context_recall=1.0,
        context_precision=1.0,
        answer_relevance=None,
    )
    assert composite == 1.0

    comp_mixed = compute_harmonic_composite(
        faithfulness=None,
        context_recall=0.8,
        context_precision=0.6,
        answer_relevance=None,
    )
    assert 0.0 <= comp_mixed <= 1.0


# ===========================================================================
# 7. Reporting & ASCII Scorecard Tests
# ===========================================================================

def test_ascii_scorecard_formatting():
    """Verify ASCII scorecard table renders required sections and headers."""
    sample_report = BenchmarkReport(
        timestamp="2026-08-15T00:00:00Z",
        total_queries=2,
        crops_evaluated=["apple", "tomato"],
        retrieval_mode="Hybrid RRF + Cross-Encoder",
        llm_enabled=True,
        mean_context_recall=0.90,
        mean_context_precision=0.85,
        mean_faithfulness=0.95,
        mean_answer_relevance=0.88,
        mean_composite_score=0.89,
        mean_latency_sec=0.15,
        quality_gate_passed=True,
        item_results=[
            {
                "id": "eval-tomato-01",
                "crop": "tomato",
                "disease": "early blight",
                "context_recall": 1.0,
                "context_precision": 1.0,
                "faithfulness": 1.0,
                "answer_relevance": 0.90,
                "composite_score": 0.97,
                "latency_sec": 0.12,
            },
            {
                "id": "eval-apple-01",
                "crop": "apple",
                "disease": "apple scab",
                "context_recall": 0.80,
                "context_precision": 0.70,
                "faithfulness": 0.90,
                "answer_relevance": 0.85,
                "composite_score": 0.81,
                "latency_sec": 0.18,
            },
        ],
    )

    table_str = format_ascii_scorecard(sample_report)
    assert "AGROSENSE-RAG QUANTITATIVE BENCHMARK SCORECARD" in table_str
    assert "Hybrid RRF + Cross-Encoder" in table_str
    assert "eval-tomato-01" in table_str
    assert "eval-apple-01" in table_str
    assert "EXECUTIVE AGGREGATE SUMMARY" in table_str
    assert "PASSED (Quality Gate Met)" in table_str


def test_save_eval_report(tmp_path: Path):
    """Verify saving benchmark report writes valid JSON to disk."""
    sample_report = BenchmarkReport(
        timestamp="2026-08-15T00:00:00Z",
        total_queries=1,
        crops_evaluated=["tomato"],
        retrieval_mode="Standard Semantic",
        llm_enabled=False,
        mean_context_recall=1.0,
        mean_context_precision=1.0,
        mean_faithfulness=None,
        mean_answer_relevance=None,
        mean_composite_score=1.0,
        mean_latency_sec=0.05,
        quality_gate_passed=True,
        item_results=[],
    )

    out_file = tmp_path / "reports" / "eval_report.json"
    saved_path = save_eval_report(sample_report, out_file)

    assert saved_path.exists()
    with saved_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["total_queries"] == 1
    assert data["quality_gate_passed"] is True


# ===========================================================================
# 8. Evaluation Runner & CLI Execution Tests
# ===========================================================================

def test_run_evaluation_retrieval_only(tmp_path: Path):
    """Verify run_evaluation() executes end-to-end in retrieval-only mode."""
    test_subset = GOLDEN_DATASET[:2]
    out_file = tmp_path / "test_report.json"

    report = run_evaluation(
        dataset=test_subset,
        limit=2,
        no_llm=True,
        output_path=out_file,
        top_k=3,
        hybrid=True,
        rerank_flag=True,
    )

    assert isinstance(report, BenchmarkReport)
    assert report.total_queries == 2
    assert report.llm_enabled is False
    assert 0.0 <= report.mean_context_recall <= 1.0
    assert 0.0 <= report.mean_context_precision <= 1.0
    assert 0.0 <= report.mean_composite_score <= 1.0
    assert len(report.item_results) == 2
    assert out_file.exists()


def test_main_cli_execution(tmp_path: Path):
    """Verify main() CLI execution with argument parsing."""
    out_file = tmp_path / "cli_report.json"
    exit_code = main(["--limit", "2", "--no-llm", "--output", str(out_file), "--top-k", "3"])
    assert exit_code == 0
    assert out_file.exists()


def test_custom_dataset_cli(tmp_path: Path):
    """Verify custom dataset loading via CLI."""
    custom_data = [
        {
            "id": "custom-01",
            "crop": "tomato",
            "disease": "early blight",
            "query": "How to treat tomato early blight?",
            "ground_truth_answer": "Chlorothalonil is effective for tomato early blight.",
            "expected_active_ingredients": ["Chlorothalonil"],
            "expected_organic_remedies": ["Copper octanoate"],
        }
    ]
    dataset_file = tmp_path / "custom_dataset.json"
    dataset_file.write_text(json.dumps(custom_data), encoding="utf-8")

    out_file = tmp_path / "custom_report.json"
    exit_code = main([
        "--dataset", str(dataset_file),
        "--no-llm",
        "--output", str(out_file),
    ])
    assert exit_code == 0
    assert out_file.exists()
    with out_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["total_queries"] == 1
    assert data["item_results"][0]["id"] == "custom-01"
