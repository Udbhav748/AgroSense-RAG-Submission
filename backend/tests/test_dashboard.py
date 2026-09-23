"""Deterministic tests for monitoring/dashboard.py -- the Module 10
gap-closure (P6) that validates the dashboard against real telemetry
rather than only claiming it exists. Previously untested.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from monitoring.dashboard import (  # noqa: E402
    _endpoint_breakdown,
    _requests_per_minute,
    _retry_activity,
)
from monitoring.log_aggregate import aggregate  # noqa: E402


class TestRetryActivity:
    def test_counts_retries_across_known_families(self):
        records = [
            {"message": "llm_generation_retrying", "provider": "groq", "request_id": "r1"},
            {"message": "llm_generation_completed", "request_id": "r1"},
            {"message": "web_search_retrying", "provider": "duckduckgo", "request_id": "r2"},
        ]
        retry = _retry_activity(records)
        assert retry["retry_events"] == 2
        assert retry["retried_requests"] == 2
        assert retry["providers"] == {"groq": 1, "duckduckgo": 1}

    def test_retry_success_rate_reflects_matching_completion_event(self):
        records = [
            {"message": "llm_generation_retrying", "provider": "groq", "request_id": "r1"},
            {"message": "llm_generation_completed", "request_id": "r1"},
            {"message": "llm_generation_retrying", "provider": "groq", "request_id": "r2"},
            # r2 never completes -- no matching completion event
        ]
        retry = _retry_activity(records)
        assert retry["retry_success_rate"] == 0.5

    def test_no_retries_gives_zero_events_not_an_error(self):
        retry = _retry_activity([{"message": "chat_query_handled", "processing_duration": 1.0}])
        assert retry["retry_events"] == 0
        assert retry["retry_success_rate"] == 0.0


class TestEndpointBreakdown:
    def test_counts_chat_and_failure_events_by_kind(self):
        records = [
            {"message": "chat_query_handled"},
            {"message": "chat_query_handled"},
            {"message": "request_failed"},
            {"message": "document_processing_completed"},
        ]
        breakdown = _endpoint_breakdown(records)
        assert breakdown == {
            "chat_query_handled": 2,
            "request_failed": 1,
            "upload_completed": 1,
        }


class TestRequestsPerMinute:
    def test_zero_with_no_timestamped_events(self):
        assert _requests_per_minute([{"message": "chat_query_handled"}]) == 0.0

    def test_counts_events_within_a_single_minute_bucket(self):
        records = [
            {"message": "chat_query_handled", "timestamp": "2026-01-01T00:00:00Z"},
            {"message": "chat_query_handled", "timestamp": "2026-01-01T00:00:30Z"},
        ]
        assert _requests_per_minute(records) == 2.0


class TestDashboardExposesRequiredViews:
    """TASK 7: the dashboard must expose availability, latency, error
    rate, tool success, retry activity, requests, token usage, and cost --
    verified here against the same aggregate() rollup the dashboard
    itself calls, using a small but realistic synthetic log set standing
    in for a captured application log."""

    def _sample_records(self) -> list[dict]:
        return [
            {"message": "chat_query_handled", "processing_duration": 1.2, "timestamp": "2026-01-01T00:00:00Z"},
            {"message": "chat_query_handled", "processing_duration": 0.8, "timestamp": "2026-01-01T00:00:05Z"},
            {"message": "request_failed", "taxonomy_category": "retriever", "timestamp": "2026-01-01T00:00:10Z"},
            {"message": "tool_invocation", "tool": "retrieval", "success": True, "latency_ms": 50.0},
            {"message": "tool_invocation", "tool": "web_search", "success": False, "latency_ms": 200.0},
            {"message": "llm_generation_retrying", "provider": "groq", "request_id": "r1"},
            {"message": "llm_generation_completed", "request_id": "r1", "prompt_tokens": 100, "completion_tokens": 40, "estimated_cost_usd": 0.0005},
        ]

    def test_all_required_dashboard_fields_are_present(self):
        records = self._sample_records()
        agg = aggregate(records)
        retry = _retry_activity(records)

        # availability
        assert "error_rate" in agg and "requests" in agg and "failures" in agg
        # latency
        assert {"p50_latency_ms", "p95_latency_ms", "p99_latency_ms"} <= agg.keys()
        # tool success
        assert "tool_success_rate" in agg and agg["tool_success_rate"]
        # retry activity
        assert retry["retry_events"] == 1
        # requests (endpoint/throughput view)
        assert _requests_per_minute(records) > 0
        # token usage / cost
        assert agg["total_tokens"] == 140
        assert agg["estimated_cost_usd"] == 0.0005
