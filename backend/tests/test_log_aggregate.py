"""Deterministic tests for monitoring/log_aggregate.py::aggregate() --
the Module 10 gap-closure (P6) formalizes error_rate as the explicit
aggregate metric (failed requests / total requests) the PDF asks for,
alongside the existing per-taxonomy-category breakdown -- these tests
pin that formula and the other rollups against small, hand-built
synthetic log records so a future change to the aggregation logic is
caught immediately, offline, with no real traffic required.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from monitoring.log_aggregate import _percentile, aggregate  # noqa: E402


class TestAggregateErrorRate:
    def test_error_rate_is_failed_over_total_requests(self):
        records = [
            {"message": "chat_query_handled", "processing_duration": 1.0},
            {"message": "chat_query_handled", "processing_duration": 1.0},
            {"message": "chat_query_handled", "processing_duration": 1.0},
            {"message": "request_failed", "taxonomy_category": "retriever"},
        ]
        agg = aggregate(records)
        assert agg["requests"] == 4
        assert agg["failures"] == 1
        assert agg["error_rate"] == 0.25

    def test_error_rate_is_zero_with_no_failures(self):
        records = [{"message": "chat_query_handled", "processing_duration": 0.5}] * 5
        agg = aggregate(records)
        assert agg["error_rate"] == 0.0

    def test_error_rate_is_zero_with_no_requests_not_a_zero_division_error(self):
        agg = aggregate([])
        assert agg["error_rate"] == 0.0
        assert agg["requests"] == 0

    def test_workflow_completion_rate_is_complement_of_error_rate(self):
        records = [
            {"message": "chat_query_handled", "processing_duration": 1.0},
            {"message": "chat_query_handled", "processing_duration": 1.0},
            {"message": "chat_query_handled", "processing_duration": 1.0},
            {"message": "request_failed", "taxonomy_category": "tool"},
        ]
        agg = aggregate(records)
        assert agg["workflow_completion_rate"] == round(1 - agg["error_rate"], 4)


class TestAggregateErrorRateByCategory:
    def test_aggregate_error_rate_does_not_replace_the_taxonomy_breakdown(self):
        """PDF requirement: expose BOTH the single aggregate error rate AND
        the per-category breakdown -- neither replaces the other."""
        records = [
            {"message": "chat_query_handled", "processing_duration": 1.0},
            {"message": "request_failed", "taxonomy_category": "retriever"},
            {"message": "request_failed", "taxonomy_category": "retriever"},
            {"message": "request_failed", "taxonomy_category": "tool"},
        ]
        agg = aggregate(records)
        assert agg["error_rate"] == 0.75  # single aggregate number, still present
        assert agg["error_rate_by_category"] == {"retriever": 2, "tool": 1}  # taxonomy, still present
        assert sum(agg["error_rate_by_category"].values()) == agg["failures"]

    def test_missing_taxonomy_category_falls_back_to_unknown(self):
        records = [{"message": "request_failed"}]
        agg = aggregate(records)
        assert agg["error_rate_by_category"] == {"unknown": 1}


class TestPercentile:
    def test_percentile_of_empty_list_is_zero(self):
        assert _percentile([], 95) == 0.0

    def test_p50_of_sorted_values(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert _percentile(values, 50) == 3.0

    def test_p99_is_close_to_max_for_small_sample(self):
        values = list(range(1, 11))  # 1..10
        assert _percentile([float(v) for v in values], 99) == 10.0

    def test_aggregate_computes_p50_p95_p99_from_processing_duration(self):
        records = [
            {"message": "chat_query_handled", "processing_duration": d}
            for d in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        ]
        agg = aggregate(records)
        assert agg["p50_latency_ms"] > 0
        assert agg["p95_latency_ms"] >= agg["p50_latency_ms"]
        assert agg["p99_latency_ms"] >= agg["p95_latency_ms"]


class TestAggregateToolAndRetrySignals:
    def test_tool_success_rate_per_tool(self):
        records = [
            {"message": "tool_invocation", "tool": "retrieval", "success": True},
            {"message": "tool_invocation", "tool": "retrieval", "success": True},
            {"message": "tool_invocation", "tool": "retrieval", "success": False},
            {"message": "tool_invocation", "tool": "web_search", "success": True},
        ]
        agg = aggregate(records)
        assert agg["tool_attempts"] == {"retrieval": 3, "web_search": 1}
        assert agg["tool_successes"] == {"retrieval": 2, "web_search": 1}
        assert agg["tool_success_rate"]["retrieval"] == 2 / 3  # aggregate() leaves this rate unrounded
        assert agg["tool_success_rate"]["web_search"] == 1.0

    def test_loop_capped_rate(self):
        records = [
            {"message": "chat_query_handled", "processing_duration": 1.0},
            {"message": "chat_query_handled", "processing_duration": 1.0},
            {"message": "loop_capped"},
        ]
        agg = aggregate(records)
        assert agg["loop_capped_rate"] == 0.5

    def test_token_and_cost_rollup(self):
        records = [
            {
                "message": "llm_generation_completed",
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "estimated_cost_usd": 0.001,
            },
            {
                "message": "llm_generation_completed",
                "prompt_tokens": 200,
                "completion_tokens": 100,
                "estimated_cost_usd": 0.002,
            },
        ]
        agg = aggregate(records)
        assert agg["total_tokens"] == 450
        assert agg["estimated_cost_usd"] == 0.003
