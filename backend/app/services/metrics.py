"""RAG Observability & Prometheus Metrics Service for InsightAI-RAG.

Provides in-process metrics recording and a Prometheus text format exporter
(export_prometheus_metrics) for RAG pipeline observability, query latencies,
retrieval metrics, vision diagnoses, rerank scores, and reflection scores.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Generator

# Fixed histogram buckets in seconds for RAG pipeline steps (sub-millisecond to multi-minute)
HISTOGRAM_BUCKETS_SECONDS: tuple[float, ...] = (
    0.001,
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
    60.0,
)


def _escape_label_value(value: Any) -> str:
    """Escape a Prometheus label value per the text exposition format."""
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _format_labels(labels: dict[str, Any] | None) -> str:
    """Format labels as {k="v",k2="v2"} or empty string if None/empty."""
    if not labels:
        return ""
    inner = ",".join(f'{k}="{_escape_label_value(v)}"' for k, v in sorted(labels.items()))
    return "{" + inner + "}"


def _label_key(labels: dict[str, Any] | None) -> tuple[tuple[str, str], ...]:
    """Normalize labels into a hashable, sorted tuple of pairs."""
    if not labels:
        return ()
    return tuple(sorted((k, str(v)) for k, v in labels.items()))


class _Histogram:
    """Cumulative histogram matching the Prometheus data model."""

    __slots__ = ("cumulative", "sum", "count")

    def __init__(self) -> None:
        self.cumulative: list[float] = [0.0] * len(HISTOGRAM_BUCKETS_SECONDS)
        self.sum: float = 0.0
        self.count: float = 0.0

    def observe(self, value: float) -> None:
        self.sum += value
        self.count += 1.0
        for i, bound in enumerate(HISTOGRAM_BUCKETS_SECONDS):
            if value <= bound:
                self.cumulative[i] += 1.0

    def reset(self) -> None:
        self.cumulative = [0.0] * len(HISTOGRAM_BUCKETS_SECONDS)
        self.sum = 0.0
        self.count = 0.0


class RAGMetricsService:
    """Thread-safe in-process metrics registry and Prometheus exporter for InsightAI-RAG.

    Tracks:
    - RAG request counts by endpoint and status
    - RAG step latencies (embedding, retrieval, reranking, generation, reflection, etc.)
    - Retrieved chunks count per query
    - Average reranking scores
    - Vision disease diagnoses by crop and disease
    - Active vector chunks indexed in vector store
    - RAG self-reflection and grading scores
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._rag_requests: defaultdict[tuple[str, str], float] = defaultdict(float)
        self._rag_latencies: defaultdict[str, _Histogram] = defaultdict(_Histogram)
        self._retrieval_chunks_count: float = 0.0
        self._rerank_scores_sum: float = 0.0
        self._rerank_scores_count: int = 0
        self._rerank_score_override: float | None = None
        self._vision_inferences: defaultdict[tuple[str, str], float] = defaultdict(float)
        self._active_vector_chunks_total: float = 0.0
        self._reflection_scores_sum: float = 0.0
        self._reflection_scores_count: int = 0
        self._reflection_score_override: float | None = None

        # Generic custom metrics
        self._custom_counters: defaultdict[tuple[str, tuple[tuple[str, str], ...]], float] = defaultdict(float)
        self._custom_gauges: defaultdict[tuple[str, tuple[tuple[str, str], ...]], float] = defaultdict(float)
        self._custom_histograms: defaultdict[tuple[str, tuple[tuple[str, str], ...]], _Histogram] = defaultdict(_Histogram)

    def reset(self) -> None:
        """Reset all metrics to initial empty state."""
        with self._lock:
            self._rag_requests.clear()
            self._rag_latencies.clear()
            self._retrieval_chunks_count = 0.0
            self._rerank_scores_sum = 0.0
            self._rerank_scores_count = 0
            self._rerank_score_override = None
            self._vision_inferences.clear()
            self._active_vector_chunks_total = 0.0
            self._reflection_scores_sum = 0.0
            self._reflection_scores_count = 0
            self._reflection_score_override = None
            self._custom_counters.clear()
            self._custom_gauges.clear()
            self._custom_histograms.clear()

    # --- RAG Request Counts --------------------------------------------------

    def record_rag_request(self, endpoint: str, status: str = "success", amount: float = 1.0) -> None:
        """Increment the RAG request counter for a specific endpoint and status."""
        with self._lock:
            self._rag_requests[(endpoint, status)] += amount

    # --- Latency Tracking ----------------------------------------------------

    def record_latency(self, step: str, duration_seconds: float) -> None:
        """Record an observed execution duration in seconds for a specific RAG step."""
        with self._lock:
            self._rag_latencies[step].observe(duration_seconds)

    def observe_latency(self, step: str, duration_seconds: float) -> None:
        """Alias for record_latency."""
        self.record_latency(step, duration_seconds)

    @contextmanager
    def timer(self, step: str) -> Generator[None, None, None]:
        """Context manager to measure and record execution latency of a RAG pipeline step."""
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            self.record_latency(step, duration)

    # --- Retrieval & Rerank Metrics ------------------------------------------

    def record_retrieval_chunks(self, count: int | float) -> None:
        """Set the number of document chunks retrieved for RAG context."""
        with self._lock:
            self._retrieval_chunks_count = float(count)

    def set_retrieval_chunks_count(self, count: int | float) -> None:
        """Alias for record_retrieval_chunks."""
        self.record_retrieval_chunks(count)

    def record_rerank_score(self, score: float) -> None:
        """Record a rerank relevance score sample to compute running average."""
        with self._lock:
            self._rerank_scores_sum += float(score)
            self._rerank_scores_count += 1

    def set_rerank_score_average(self, score: float) -> None:
        """Explicitly set the rerank score average gauge."""
        with self._lock:
            self._rerank_score_override = float(score)

    @property
    def rerank_score_average(self) -> float:
        """Return the current rerank score average."""
        with self._lock:
            if self._rerank_score_override is not None:
                return self._rerank_score_override
            if self._rerank_scores_count > 0:
                return self._rerank_scores_sum / self._rerank_scores_count
            return 0.0

    # --- Vision Diagnostics --------------------------------------------------

    def record_vision_inference(self, crop: str, disease: str, amount: float = 1.0) -> None:
        """Increment the plant disease vision inference counter."""
        with self._lock:
            self._vision_inferences[(crop, disease)] += amount

    # --- Active Vector Store Chunks ------------------------------------------

    def record_active_vector_chunks(self, count: int | float) -> None:
        """Set the total number of active vector chunks in the vector store."""
        with self._lock:
            self._active_vector_chunks_total = float(count)

    def set_active_vector_chunks(self, count: int | float) -> None:
        """Alias for record_active_vector_chunks."""
        self.record_active_vector_chunks(count)

    # --- RAG Reflection / Faithfulness Scores --------------------------------

    def record_reflection_score(self, score: float) -> None:
        """Record a RAG reflection / answer faithfulness evaluation score."""
        with self._lock:
            self._reflection_scores_sum += float(score)
            self._reflection_scores_count += 1

    def set_reflection_score_average(self, score: float) -> None:
        """Explicitly set the average reflection score gauge."""
        with self._lock:
            self._reflection_score_override = float(score)

    @property
    def reflection_score_average(self) -> float:
        """Return the current reflection score average."""
        with self._lock:
            if self._reflection_score_override is not None:
                return self._reflection_score_override
            if self._reflection_scores_count > 0:
                return self._reflection_scores_sum / self._reflection_scores_count
            return 0.0

    # --- Generic Metric Registration ----------------------------------------

    def inc_counter(self, name: str, labels: dict[str, Any] | None = None, amount: float = 1.0) -> None:
        with self._lock:
            self._custom_counters[(name, _label_key(labels))] += amount

    def set_gauge(self, name: str, value: float, labels: dict[str, Any] | None = None) -> None:
        with self._lock:
            self._custom_gauges[(name, _label_key(labels))] = float(value)

    def observe_histogram(self, name: str, value: float, labels: dict[str, Any] | None = None) -> None:
        with self._lock:
            self._custom_histograms[(name, _label_key(labels))].observe(value)

    # --- Prometheus Exporter -------------------------------------------------

    def export_prometheus_metrics(self) -> str:
        """Format all recorded metrics into standard Prometheus text exposition format."""
        with self._lock:
            lines: list[str] = []

            # 1. RAG requests counter
            lines.append("# HELP insightai_rag_requests_total Total count of InsightAI RAG requests by endpoint and status.")
            lines.append("# TYPE insightai_rag_requests_total counter")
            if self._rag_requests:
                for (endpoint, status), count in sorted(self._rag_requests.items()):
                    labels = _format_labels({"endpoint": endpoint, "status": status})
                    lines.append(f"insightai_rag_requests_total{labels} {count:.0f}")

            # 2. Latency histogram
            lines.append("# HELP insightai_rag_latency_seconds Latency of InsightAI RAG pipeline processing steps in seconds.")
            lines.append("# TYPE insightai_rag_latency_seconds histogram")
            for step, hist in sorted(self._rag_latencies.items()):
                for i, bound in enumerate(HISTOGRAM_BUCKETS_SECONDS):
                    bucket_labels = _format_labels({"le": f"{bound:g}", "step": step})
                    lines.append(f"insightai_rag_latency_seconds_bucket{bucket_labels} {hist.cumulative[i]:.0f}")
                inf_labels = _format_labels({"le": "+Inf", "step": step})
                lines.append(f"insightai_rag_latency_seconds_bucket{inf_labels} {hist.count:.0f}")
                sum_labels = _format_labels({"step": step})
                lines.append(f"insightai_rag_latency_seconds_sum{sum_labels} {hist.sum:.4f}")
                lines.append(f"insightai_rag_latency_seconds_count{sum_labels} {hist.count:.0f}")

            # 3. Retrieval chunks count
            lines.append("# HELP insightai_rag_retrieval_chunks_count Number of document chunks retrieved for RAG context.")
            lines.append("# TYPE insightai_rag_retrieval_chunks_count gauge")
            lines.append(f"insightai_rag_retrieval_chunks_count {self._retrieval_chunks_count:.0f}")

            # 4. Rerank score average
            rerank_avg = (
                self._rerank_score_override
                if self._rerank_score_override is not None
                else (self._rerank_scores_sum / self._rerank_scores_count if self._rerank_scores_count > 0 else 0.0)
            )
            lines.append("# HELP insightai_rag_rerank_score_average Average reranking relevance score of retrieved chunks.")
            lines.append("# TYPE insightai_rag_rerank_score_average gauge")
            lines.append(f"insightai_rag_rerank_score_average {rerank_avg:.4f}")

            # 5. Vision inferences counter
            lines.append("# HELP insightai_vision_inferences_total Total count of plant disease vision inferences by crop and disease.")
            lines.append("# TYPE insightai_vision_inferences_total counter")
            if self._vision_inferences:
                for (crop, disease), count in sorted(self._vision_inferences.items()):
                    labels = _format_labels({"crop": crop, "disease": disease})
                    lines.append(f"insightai_vision_inferences_total{labels} {count:.0f}")

            # 6. Active vector chunks
            lines.append("# HELP insightai_active_vector_chunks_total Total number of active vector chunks indexed in vector store.")
            lines.append("# TYPE insightai_active_vector_chunks_total gauge")
            lines.append(f"insightai_active_vector_chunks_total {self._active_vector_chunks_total:.0f}")

            # 7. RAG reflection score average
            refl_avg = (
                self._reflection_score_override
                if self._reflection_score_override is not None
                else (self._reflection_scores_sum / self._reflection_scores_count if self._reflection_scores_count > 0 else 0.0)
            )
            lines.append("# HELP insightai_rag_reflection_score_average Average RAG self-reflection and faithfulness score.")
            lines.append("# TYPE insightai_rag_reflection_score_average gauge")
            lines.append(f"insightai_rag_reflection_score_average {refl_avg:.4f}")

            # Custom counters
            for (name, label_key), val in sorted(self._custom_counters.items()):
                labels = dict(label_key)
                lines.append(f"# HELP {name} Counter incremented by observed events.")
                lines.append(f"# TYPE {name} counter")
                lines.append(f"{name}{_format_labels(labels)} {val:.0f}")

            # Custom gauges
            for (name, label_key), val in sorted(self._custom_gauges.items()):
                labels = dict(label_key)
                lines.append(f"# HELP {name} Gauge set to the last observed value.")
                lines.append(f"# TYPE {name} gauge")
                lines.append(f"{name}{_format_labels(labels)} {val:.4f}")

            # Custom histograms
            for (name, label_key), hist in sorted(self._custom_histograms.items()):
                labels = dict(label_key)
                lines.append(f"# HELP {name} Observation durations in seconds.")
                lines.append(f"# TYPE {name} histogram")
                for i, bound in enumerate(HISTOGRAM_BUCKETS_SECONDS):
                    bucket_labels = dict(labels)
                    bucket_labels["le"] = f"{bound:g}"
                    lines.append(f"{name}_bucket{_format_labels(bucket_labels)} {hist.cumulative[i]:.0f}")
                inf_labels = dict(labels)
                inf_labels["le"] = "+Inf"
                lines.append(f"{name}_bucket{_format_labels(inf_labels)} {hist.count:.0f}")
                lines.append(f"{name}_sum{_format_labels(labels)} {hist.sum:.4f}")
                lines.append(f"{name}_count{_format_labels(labels)} {hist.count:.0f}")

            return "\n".join(lines) + "\n"


# Process-wide singleton
_metrics_service = RAGMetricsService()


def get_metrics_service() -> RAGMetricsService:
    """Return the singleton RAGMetricsService instance."""
    return _metrics_service


def get_metrics() -> RAGMetricsService:
    """Alias for get_metrics_service."""
    return _metrics_service


def export_prometheus_metrics() -> str:
    """Format and export all RAG metrics into standard Prometheus text format."""
    return _metrics_service.export_prometheus_metrics()


def reset_metrics() -> None:
    """Reset all recorded metrics in the singleton instance."""
    _metrics_service.reset()


def record_rag_request(endpoint: str, status: str = "success", amount: float = 1.0) -> None:
    """Record a RAG request with endpoint and status labels."""
    _metrics_service.record_rag_request(endpoint, status, amount)


def record_latency(step: str, duration_seconds: float) -> None:
    """Record observed latency for a RAG step."""
    _metrics_service.record_latency(step, duration_seconds)


def observe_latency(step: str, duration_seconds: float) -> None:
    """Alias for record_latency."""
    _metrics_service.observe_latency(step, duration_seconds)


def record_retrieval_chunks(count: int | float) -> None:
    """Record the number of document chunks retrieved for RAG context."""
    _metrics_service.record_retrieval_chunks(count)


def set_retrieval_chunks_count(count: int | float) -> None:
    """Set the number of document chunks retrieved for RAG context."""
    _metrics_service.set_retrieval_chunks_count(count)


def record_rerank_score(score: float) -> None:
    """Record a reranking relevance score."""
    _metrics_service.record_rerank_score(score)


def set_rerank_score_average(score: float) -> None:
    """Explicitly set the rerank score average."""
    _metrics_service.set_rerank_score_average(score)


def record_vision_inference(crop: str, disease: str, amount: float = 1.0) -> None:
    """Record a vision diagnosis inference by crop and disease."""
    _metrics_service.record_vision_inference(crop, disease, amount)


def record_active_vector_chunks(count: int | float) -> None:
    """Set the total number of active vector chunks indexed in vector store."""
    _metrics_service.record_active_vector_chunks(count)


def set_active_vector_chunks(count: int | float) -> None:
    """Alias for record_active_vector_chunks."""
    _metrics_service.set_active_vector_chunks(count)


def record_reflection_score(score: float) -> None:
    """Record a RAG self-reflection and answer faithfulness score."""
    _metrics_service.record_reflection_score(score)


def set_reflection_score_average(score: float) -> None:
    """Explicitly set the average reflection score."""
    _metrics_service.set_reflection_score_average(score)


def timer(step: str) -> Generator[None, None, None]:
    """Context manager for timing a RAG execution step."""
    return _metrics_service.timer(step)


class LatencyTimer:
    """Context manager class for measuring execution latency of a RAG pipeline step."""

    def __init__(self, step: str, service: RAGMetricsService | None = None) -> None:
        self.step = step
        self.service = service or _metrics_service
        self.start_time: float = 0.0

    def __enter__(self) -> LatencyTimer:
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        duration = time.perf_counter() - self.start_time
        self.service.record_latency(self.step, duration)
