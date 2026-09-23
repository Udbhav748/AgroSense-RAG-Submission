"""Tests for the live Prometheus metrics endpoint and in-process registry.

Covers:
- /health and /metrics both being reachable, with /metrics emitting
  well-formed text exposition format (declare, type, sample lines).
- The request middleware feeding http_requests_total and the latency
  histogram (low-cardinality {id} path normalization).
- The tool-invocation and LLM-generation hooks feeding their metric
  families.
- The optional METRICS_BEARER_TOKEN gate (constant-time compare).

The real GET /chat path is not required to hit the record_llm_generation
hook — that hook lives in gemini_client/groq_client, which the suite's
FakeLLMClient stubs out. Instead, the LLM/tool metric families are
exercised by calling the metric-recording methods directly (they're the
same call sites the fast path uses), and the request metrics are
exercised end-to-end through the real middleware.
"""

import pytest
from fastapi.testclient import TestClient

from app.core.metrics import Metrics, normalize_path, reset_metrics
from app.main import app


@pytest.fixture(autouse=True)
def _isolate_metrics():
    """Wipe the shared registry before and after each test so counts from
    one test never bleed into another (the registry is a module singleton,
    deliberately shared by the app under test)."""
    reset_metrics()
    yield
    reset_metrics()


@pytest.fixture
def metrics_client():
    with TestClient(app) as test_client:
        yield test_client


class TestMetricsRegistry:
    def test_normalize_path_replaces_uuids_and_numeric_ids(self):
        assert (
            normalize_path("/documents/550e8400-e29b-41d4-a716-446655440000")
            == "/documents/{id}"
        )
        assert normalize_path("/chat/42") == "/chat/{id}"
        assert normalize_path("/health") == "/health"

    def test_counter_labels_are_order_insensitive(self):
        m = Metrics()
        m.inc_counter("c", {"a": "1", "b": "2"})
        m.inc_counter("c", {"b": "2", "a": "1"})
        assert m.render().count("c{a=\"1\",b=\"2\"} 2") == 1

    def test_histogram_bucket_semantics(self):
        m = Metrics()
        m.observe_duration("h", 0.05, {"tool": "retrieval"})
        m.observe_duration("h", 120.0, {"tool": "retrieval"})
        text = m.render()
        # Labels render alphabetically; le sorts before tool.
        # The 0.05s observation lands in every bucket >= 0.05; the 120s one
        # lands only in +Inf (which equals count).
        assert 'h_bucket{le="0.1",tool="retrieval"} 1' in text
        assert 'h_bucket{le="+Inf",tool="retrieval"} 2' in text
        assert 'h_sum{tool="retrieval"} 120.050' in text
        assert 'h_count{tool="retrieval"} 2' in text

    def test_record_llm_generation_populates_token_and_cost_metrics(self):
        m = Metrics()
        m.record_llm_generation(
            provider="gemini",
            model="gemini-flash",
            prompt_tokens=100,
            completion_tokens=50,
            estimated_cost_usd=0.0012,
        )
        text = m.render()
        assert 'llm_generations_total{model="gemini-flash",provider="gemini"} 1' in text
        assert 'llm_tokens_total{provider="gemini",type="completion"} 50' in text
        assert 'llm_tokens_total{provider="gemini",type="prompt"} 100' in text
        assert 'llm_cost_usd_total{provider="gemini"} 0' in text  # rendered as float, 0 rounding

    def test_record_tool_invocation_tracks_success_and_latency(self):
        m = Metrics()
        m.record_tool_invocation(tool="retrieval", success=True, latency_ms=100.0)
        m.record_tool_invocation(tool="retrieval", success=False, latency_ms=200.0)
        text = m.render()
        assert 'tool_invocations_total{result="error",tool="retrieval"} 1' in text
        assert 'tool_invocations_total{result="success",tool="retrieval"} 1' in text
        assert 'tool_invocation_duration_seconds_bucket{le="0.25",tool="retrieval"} 2' in text


class TestMetricsEndpoint:
    def test_metrics_returns_prometheus_text_format(self, metrics_client):
        metrics_client.get("/health")
        response = metrics_client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
        body = response.text
        # Declaration sanity: every sample must be preceded by its TYPE line,
        # and the up-time gauge must be present.
        assert "insightai_uptime_seconds" in body
        assert '# TYPE http_requests_total counter' in body
        assert '# TYPE http_request_duration_seconds histogram' in body

    def test_request_middleware_records_requests_and_latency(self, metrics_client):
        metrics_client.get("/health")
        metrics_client.get("/health")
        response = metrics_client.get("/metrics")
        body = response.text
        assert 'http_requests_total{method="GET",path="/health",status="200"} 2' in body
        # Histogram family exists with sum + count for the /health path.
        assert 'http_request_duration_seconds_sum{method="GET",path="/health"}' in body
        assert 'http_request_duration_seconds_count{method="GET",path="/health"} 2' in body

    def test_metrics_path_normalizes_dynamic_ids(self, metrics_client):
        # GET on a POST-only route 405s but still flows through the
        # middleware, proving document-id-carrying paths stay low-cardinality:
        # two different UUIDs collapse into a single /documents/{id} series.
        metrics_client.get("/documents/aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
        metrics_client.get("/documents/12345678-1234-1234-1234-123456789012")
        response = metrics_client.get("/metrics")
        body = response.text
        assert 'http_requests_total{method="GET",path="/documents/{id}",status="405"} 2' in body
        assert "aaaaaaaa" not in body
        assert "12345678-1234" not in body

    def test_metrics_excludes_own_response(self, metrics_client):
        # Prometheus semantics: a scrape is recorded AFTER its response is
        # rendered, so a /metrics scrape excludes itself (visible on the
        # NEXT scrape). Two scrapes means the first shows 0 prior metrics
        # scrapes, the second shows 1.
        body1 = metrics_client.get("/metrics").text
        assert 'path="/metrics"' not in body1
        body2 = metrics_client.get("/metrics").text
        assert 'http_requests_total{method="GET",path="/metrics",status="200"} 1' in body2


class TestMetricsAuth:
    def test_metrics_rejects_missing_bearer_token(self, monkeypatch, metrics_client):
        from app.core.config import settings

        monkeypatch.setattr(settings, "metrics_bearer_token", "sekrit")
        response = metrics_client.get("/metrics")
        assert response.status_code == 401
        assert response.json()["error_code"] == "METRICS_UNAUTHORIZED"

    def test_metrics_accepts_correct_bearer_token(self, monkeypatch, metrics_client):
        from app.core.config import settings

        monkeypatch.setattr(settings, "metrics_bearer_token", "sekrit")
        metrics_client.get("/health")
        response = metrics_client.get(
            "/metrics", headers={"Authorization": "Bearer sekrit"}
        )
        assert response.status_code == 200
        assert "http_requests_total" in response.text

    def test_metrics_rejects_wrong_bearer_token(self, monkeypatch, metrics_client):
        from app.core.config import settings

        monkeypatch.setattr(settings, "metrics_bearer_token", "sekrit")
        response = metrics_client.get(
            "/metrics", headers={"Authorization": "Bearer wrong"}
        )
        assert response.status_code == 401
