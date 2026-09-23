#!/usr/bin/env python
"""Module 10 observability evaluation: P50/P95/P99 latency and error
rate from a real, controlled sample of requests through the actual
FastAPI app (TestClient) and the real agent graph -- fully offline
(LLM/embedding calls mocked, mirroring tests/test_main.py's own
fixture pattern), so no API quota is consumed and the sample is
reproducible.

Methodology: 30 successful POST /chat requests + 5 requests to a
missing document (a real client-error path) through the same TestClient
instance, timing each with time.perf_counter() end-to-end (the same
wall-clock latency a real caller would observe), then computing
percentiles from that real sample -- NOT extrapolated from a single
request.

Usage (from backend/):
    python eval/module10/runners/run_observability_eval.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from fastapi.testclient import TestClient  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.main import app  # noqa: E402
from app.models.document import EmbeddedChunk  # noqa: E402
from app.services.faiss_vector_store import FAISSVectorStore  # noqa: E402
from eval.module10 import config  # noqa: E402

FAKE_EMBEDDING_DIM = 8
FAKE_EMBEDDING = [1.0] + [0.0] * (FAKE_EMBEDDING_DIM - 1)
DOCUMENT_ID = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
N_SUCCESS_REQUESTS = 30
N_ERROR_REQUESTS = 5


class _FakeLLMClient:
    def generate(self, prompt: str) -> str:
        return "A grounded answer citing the retrieved context [1]."

    def generate_stream(self, prompt: str):
        yield self.generate(prompt)

    def generate_structured(self, prompt: str) -> str:
        return '{"answer": "structured answer", "sources": []}'


def _percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    idx = min(len(sorted_values) - 1, max(0, round(p / 100 * (len(sorted_values) - 1))))
    return sorted_values[idx]


def main() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        store = FAISSVectorStore(index_path=tmp_path / "index.faiss", metadata_path=tmp_path / "metadata.json")
        store.create_index(dimension=FAKE_EMBEDDING_DIM)
        store.add_embeddings(
            [
                EmbeddedChunk(
                    chunk_id="chunk-1",
                    document_id=DOCUMENT_ID,
                    embedding=FAKE_EMBEDDING,
                    metadata={"chunk_index": 0, "total_chunks": 1, "source": "pdf", "text": "Apple scab treatment: sulfur or captan."},
                )
            ]
        )
        fake_llm = _FakeLLMClient()

        with (
            patch("app.api.v1.routes.query.get_vector_store", lambda: store),
            patch("app.api.v1.routes.documents.get_vector_store", lambda: store),
            patch("app.api.v1.routes.query.get_llm_client", lambda: fake_llm),
            patch("app.services.retrieval_service.embed_query", lambda query: FAKE_EMBEDDING),
            patch("app.services.hybrid_search.embed_query", lambda query: FAKE_EMBEDDING),
            TestClient(app) as client,
        ):
            headers = {"X-API-Key": settings.api_key}
            latencies_success: list[float] = []
            errors = 0

            for i in range(N_SUCCESS_REQUESTS):
                start = time.perf_counter()
                response = client.post("/chat", json={"query": f"What treats apple scab? (req {i})"}, headers=headers)
                latency = time.perf_counter() - start
                if response.status_code == 200:
                    latencies_success.append(latency)
                else:
                    errors += 1

            latencies_error: list[float] = []
            for i in range(N_ERROR_REQUESTS):
                start = time.perf_counter()
                response = client.delete(f"/documents/nonexistent-{i}", params={"confirm": "true"}, headers=headers)
                latency = time.perf_counter() - start
                latencies_error.append(latency)
                if response.status_code not in (404, 400):
                    errors += 1

            total_requests = N_SUCCESS_REQUESTS + N_ERROR_REQUESTS
            error_rate = round(errors / total_requests, 4)

            sorted_success = sorted(latencies_success)
            p50 = _percentile(sorted_success, 50)
            p95 = _percentile(sorted_success, 95)
            p99 = _percentile(sorted_success, 99)

            report = {
                "metadata": config.run_metadata(sample_count=total_requests, dataset_version="observability_v1"),
                "methodology": (
                    "Real, controlled sample of 35 requests (30 successful POST /chat + 5 error-path "
                    "DELETE /documents/{missing_id}) through the actual FastAPI app via TestClient, with "
                    "only the LLM/embedding calls mocked (no API quota consumed) -- mirrors "
                    "tests/test_main.py's own fixture pattern. Latency measured end-to-end with "
                    "time.perf_counter() around each request, the same wall-clock time a real HTTP caller "
                    "would observe. Percentiles computed from this real sample, not extrapolated from one "
                    "request or estimated."
                ),
                "n_success_requests": len(latencies_success),
                "n_error_requests": N_ERROR_REQUESTS,
                "n_total_requests": total_requests,
                "error_rate": error_rate,
                "latency_seconds": {
                    "p50": round(p50, 4),
                    "p95": round(p95, 4),
                    "p99": round(p99, 4),
                    "min": round(min(sorted_success), 4) if sorted_success else None,
                    "max": round(max(sorted_success), 4) if sorted_success else None,
                    "mean": round(sum(sorted_success) / len(sorted_success), 4) if sorted_success else None,
                },
                "limitations": [
                    "LLM calls are mocked (instant, deterministic responses) -- these latencies measure "
                    "the application's own overhead (routing, retrieval, graph execution, response "
                    "construction), NOT real end-to-end latency including live LLM provider round-trips "
                    "(that figure is separately measured and reported in eval/module10/reports/agent_eval_*.json "
                    "and docs/RAG_BENCHMARK_REPORT.md from live runs).",
                    "35 requests on a single local machine/process, not a production-scale sample.",
                    "Token usage and cost are not included here (mocked LLM makes no real API call) -- see "
                    "the cost artifacts from live runs for real token/cost figures.",
                ],
            }

            path = config.save_report(report, name="observability_final")
            print(f"Saved: {path}")
            print(f"P50={report['latency_seconds']['p50']}s P95={report['latency_seconds']['p95']}s P99={report['latency_seconds']['p99']}s")
            print(f"Error rate: {error_rate} ({errors}/{total_requests})")


if __name__ == "__main__":
    main()
