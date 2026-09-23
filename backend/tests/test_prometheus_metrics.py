"""Unit tests for RAG Observability & Prometheus Metrics Exporter.

Verifies:
- Standard Prometheus text exposition format (# HELP, # TYPE, samples)
- Counter increments for RAG requests and vision disease diagnoses
- Histogram latency observations and bucket aggregations across RAG steps
- Gauges for retrieval chunk count, rerank score average, active vector chunks, and reflection score
- Timer context managers and LatencyTimer
- Reset isolation behavior
- GET /metrics HTTP endpoint response content-type and body
"""

import time
import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.core.metrics import reset_metrics as reset_core_metrics
from app.main import app
from app.services.metrics import (
    HISTOGRAM_BUCKETS_SECONDS,
    LatencyTimer,
    RAGMetricsService,
    export_prometheus_metrics,
    get_metrics_service,
    observe_latency,
    record_active_vector_chunks,
    record_latency,
    record_rag_request,
    record_reflection_score,
    record_rerank_score,
    record_retrieval_chunks,
    record_vision_inference,
    reset_metrics,
    set_active_vector_chunks,
    set_reflection_score_average,
    set_rerank_score_average,
    set_retrieval_chunks_count,
    timer,
)


@pytest.fixture(autouse=True)
def _isolate_metrics():
    """Reset both core and RAG metrics registries before and after each test."""
    reset_metrics()
    reset_core_metrics()
    yield
    reset_metrics()
    reset_core_metrics()


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


class TestPrometheusMetricsFormat:
    def test_export_contains_all_metric_declarations(self):
        """Exporter must declare HELP and TYPE for all core RAG observability metrics."""
        text = export_prometheus_metrics()

        assert "# HELP insightai_rag_requests_total" in text
        assert "# TYPE insightai_rag_requests_total counter" in text

        assert "# HELP insightai_rag_latency_seconds" in text
        assert "# TYPE insightai_rag_latency_seconds histogram" in text

        assert "# HELP insightai_rag_retrieval_chunks_count" in text
        assert "# TYPE insightai_rag_retrieval_chunks_count gauge" in text

        assert "# HELP insightai_rag_rerank_score_average" in text
        assert "# TYPE insightai_rag_rerank_score_average gauge" in text

        assert "# HELP insightai_vision_inferences_total" in text
        assert "# TYPE insightai_vision_inferences_total counter" in text

        assert "# HELP insightai_active_vector_chunks_total" in text
        assert "# TYPE insightai_active_vector_chunks_total gauge" in text

        assert "# HELP insightai_rag_reflection_score_average" in text
        assert "# TYPE insightai_rag_reflection_score_average gauge" in text

    def test_isolated_service_instance_export(self):
        """Isolated RAGMetricsService instances produce independent valid Prometheus text."""
        service = RAGMetricsService()
        service.record_rag_request(endpoint="/api/v1/query", status="success")
        service.set_active_vector_chunks(500)

        text = service.export_prometheus_metrics()
        assert 'insightai_rag_requests_total{endpoint="/api/v1/query",status="success"} 1' in text
        assert "insightai_active_vector_chunks_total 500" in text


class TestCounterIncrements:
    def test_rag_requests_counter_increments(self):
        """RAG requests counter correctly tallies by endpoint and status."""
        record_rag_request(endpoint="/api/v1/query", status="success")
        record_rag_request(endpoint="/api/v1/query", status="success")
        record_rag_request(endpoint="/api/v1/query", status="error")
        record_rag_request(endpoint="/api/v1/chat", status="success", amount=3)

        text = export_prometheus_metrics()
        assert 'insightai_rag_requests_total{endpoint="/api/v1/query",status="success"} 2' in text
        assert 'insightai_rag_requests_total{endpoint="/api/v1/query",status="error"} 1' in text
        assert 'insightai_rag_requests_total{endpoint="/api/v1/chat",status="success"} 3' in text

    def test_vision_inferences_counter_increments(self):
        """Vision diagnoses counter correctly tallies by crop and disease."""
        record_vision_inference(crop="Apple", disease="Apple_scab")
        record_vision_inference(crop="Apple", disease="Apple_scab")
        record_vision_inference(crop="Tomato", disease="Early_blight")
        record_vision_inference(crop="Corn", disease="healthy", amount=5)

        text = export_prometheus_metrics()
        assert 'insightai_vision_inferences_total{crop="Apple",disease="Apple_scab"} 2' in text
        assert 'insightai_vision_inferences_total{crop="Tomato",disease="Early_blight"} 1' in text
        assert 'insightai_vision_inferences_total{crop="Corn",disease="healthy"} 5' in text


class TestLatencyObservationRecording:
    def test_latency_histogram_buckets_and_summary(self):
        """Latency observations populate cumulative buckets, sum, and count."""
        # 0.04s should fall into le="0.05", "0.1", ..., "+Inf"
        record_latency(step="retrieval", duration_seconds=0.04)
        # 0.20s should fall into le="0.25", ..., "+Inf"
        observe_latency(step="retrieval", duration_seconds=0.20)

        text = export_prometheus_metrics()
        assert 'insightai_rag_latency_seconds_bucket{le="0.025",step="retrieval"} 0' in text
        assert 'insightai_rag_latency_seconds_bucket{le="0.05",step="retrieval"} 1' in text
        assert 'insightai_rag_latency_seconds_bucket{le="0.25",step="retrieval"} 2' in text
        assert 'insightai_rag_latency_seconds_bucket{le="+Inf",step="retrieval"} 2' in text
        assert 'insightai_rag_latency_seconds_count{step="retrieval"} 2' in text
        assert 'insightai_rag_latency_seconds_sum{step="retrieval"} 0.2400' in text

    def test_timer_context_manager(self):
        """timer context manager records elapsed execution duration."""
        with timer("embedding"):
            time.sleep(0.002)

        text = export_prometheus_metrics()
        assert 'insightai_rag_latency_seconds_count{step="embedding"} 1' in text
        assert 'insightai_rag_latency_seconds_bucket{le="+Inf",step="embedding"} 1' in text

    def test_latency_timer_class(self):
        """LatencyTimer class records step duration upon exit."""
        with LatencyTimer("reranking"):
            time.sleep(0.002)

        text = export_prometheus_metrics()
        assert 'insightai_rag_latency_seconds_count{step="reranking"} 1' in text


class TestGaugesAndReflectionScores:
    def test_retrieval_chunks_count_gauge(self):
        """Retrieval chunks count reflects last set value."""
        record_retrieval_chunks(8)
        assert "insightai_rag_retrieval_chunks_count 8" in export_prometheus_metrics()

        set_retrieval_chunks_count(12)
        assert "insightai_rag_retrieval_chunks_count 12" in export_prometheus_metrics()

    def test_rerank_score_average_computation(self):
        """Rerank scores correctly calculate running average."""
        record_rerank_score(0.80)
        record_rerank_score(0.90)

        text = export_prometheus_metrics()
        assert "insightai_rag_rerank_score_average 0.8500" in text

        # Test explicit override
        set_rerank_score_average(0.95)
        assert "insightai_rag_rerank_score_average 0.9500" in export_prometheus_metrics()

    def test_active_vector_chunks_gauge(self):
        """Active vector chunks count is reported as gauge."""
        record_active_vector_chunks(15420)
        assert "insightai_active_vector_chunks_total 15420" in export_prometheus_metrics()

        set_active_vector_chunks(16000)
        assert "insightai_active_vector_chunks_total 16000" in export_prometheus_metrics()

    def test_reflection_score_average_computation(self):
        """Reflection scores correctly calculate running average and support override."""
        record_reflection_score(0.85)
        record_reflection_score(0.95)

        text = export_prometheus_metrics()
        assert "insightai_rag_reflection_score_average 0.9000" in text

        set_reflection_score_average(0.98)
        assert "insightai_rag_reflection_score_average 0.9800" in export_prometheus_metrics()


class TestResetMetrics:
    def test_reset_clears_all_rag_metrics(self):
        """Reset wipes all counters, latencies, and gauges."""
        record_rag_request(endpoint="/api/v1/query", status="success")
        record_latency(step="retrieval", duration_seconds=0.5)
        record_vision_inference(crop="Apple", disease="healthy")
        record_retrieval_chunks(10)
        record_rerank_score(0.9)
        set_active_vector_chunks(200)

        reset_metrics()

        text = export_prometheus_metrics()
        assert 'insightai_rag_requests_total' in text
        assert 'endpoint="/api/v1/query"' not in text
        assert 'step="retrieval"' not in text
        assert 'crop="Apple"' not in text
        assert "insightai_rag_retrieval_chunks_count 0" in text
        assert "insightai_active_vector_chunks_total 0" in text
        assert "insightai_rag_rerank_score_average 0.0000" in text


class TestPrometheusEndpointIntegration:
    def test_get_metrics_endpoint_response(self, client):
        """GET /metrics returns 200 with text/plain; version=0.0.4 and formatted metrics."""
        record_rag_request(endpoint="/api/v1/query", status="success", amount=5)
        record_retrieval_chunks(6)
        record_rerank_score(0.88)
        record_vision_inference(crop="Potato", disease="Late_blight", amount=2)
        set_active_vector_chunks(4200)

        response = client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
        assert "version=0.0.4" in response.headers["content-type"]

        body = response.text
        assert 'insightai_rag_requests_total{endpoint="/api/v1/query",status="success"} 5' in body
        assert "insightai_rag_retrieval_chunks_count 6" in body
        assert "insightai_rag_rerank_score_average 0.8800" in body
        assert 'insightai_vision_inferences_total{crop="Potato",disease="Late_blight"} 2' in body
        assert "insightai_active_vector_chunks_total 4200" in body

    def test_metrics_endpoint_with_bearer_token(self, monkeypatch, client):
        """Bearer token authentication is enforced when configured."""
        monkeypatch.setattr(settings, "metrics_bearer_token", "prometheus-secret-token")

        # Unauthenticated request should return 401
        res_unauth = client.get("/metrics")
        assert res_unauth.status_code == 401
        assert res_unauth.json()["error_code"] == "METRICS_UNAUTHORIZED"

        # Authorized request with Bearer token should return 200
        res_auth = client.get(
            "/metrics", headers={"Authorization": "Bearer prometheus-secret-token"}
        )
        assert res_auth.status_code == 200
        assert "insightai_rag_requests_total" in res_auth.text
