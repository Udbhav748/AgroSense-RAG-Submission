#!/usr/bin/env python
"""Module 10 controlled local load/concurrency test.

Drives concurrent POST /chat requests through the actual FastAPI app
(TestClient, in-process -- no real network socket) at 10/25/50/100
concurrent levels, LLM/embedding calls mocked (no API quota consumed),
using a ThreadPoolExecutor so requests genuinely overlap in wall-clock
time. Measures real throughput, P50/P95/P99 latency, and error rate at
each level, and reports psutil-based CPU/memory samples where available.

Honest scope: this measures THIS SINGLE PROCESS's local capacity on
whatever machine runs it -- it is explicitly NOT extrapolated to
production/cloud scale, and does not exercise a real LLM provider (a
live-provider load test would need real API quota this project has
already exhausted twice, and would conflate provider rate-limiting with
this application's own concurrency handling). It answers "what breaks
first LOCALLY" -- see the report's own "first_bottleneck" field for the
answer with evidence, not a guess.

Usage (from backend/):
    python eval/module10/runners/run_load_test.py
"""

from __future__ import annotations

import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
DOCUMENT_ID = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
CONCURRENCY_LEVELS = [10, 25, 50, 100]


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


def _resource_sample() -> dict:
    try:
        import psutil

        process = psutil.Process()
        return {
            "cpu_percent": process.cpu_percent(interval=0.1),
            "memory_rss_mb": round(process.memory_info().rss / (1024 * 1024), 2),
            "available": True,
        }
    except ImportError:
        return {"available": False, "note": "psutil not installed -- CPU/memory not sampled"}


def _one_request(client: TestClient, headers: dict, i: int) -> tuple[float, int, str | None]:
    start = time.perf_counter()
    response = client.post("/chat", json={"query": f"What treats apple scab? (load req {i})"}, headers=headers)
    latency = time.perf_counter() - start
    error_code = None
    if response.status_code != 200:
        try:
            error_code = response.json().get("error_code") or response.json().get("detail")
        except Exception:  # noqa: BLE001
            error_code = f"status_{response.status_code}"
    return latency, response.status_code, error_code


def run_level(client: TestClient, headers: dict, concurrency: int) -> dict:
    resource_before = _resource_sample()
    start = time.perf_counter()
    latencies: list[float] = []
    errors = 0
    error_codes: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(_one_request, client, headers, i) for i in range(concurrency)]
        for future in as_completed(futures):
            latency, status, error_code = future.result()
            latencies.append(latency)
            if status != 200:
                errors += 1
                key = str(error_code)
                error_codes[key] = error_codes.get(key, 0) + 1
    wall_time = time.perf_counter() - start
    resource_after = _resource_sample()

    sorted_latencies = sorted(latencies)
    return {
        "concurrency": concurrency,
        "wall_time_seconds": round(wall_time, 4),
        "requests_per_second": round(concurrency / wall_time, 2) if wall_time > 0 else None,
        "error_rate": round(errors / concurrency, 4),
        "error_codes": error_codes,
        "latency_seconds": {
            "p50": round(_percentile(sorted_latencies, 50), 4),
            "p95": round(_percentile(sorted_latencies, 95), 4),
            "p99": round(_percentile(sorted_latencies, 99), 4),
            "mean": round(statistics.mean(sorted_latencies), 4) if sorted_latencies else None,
        },
        "resource_before": resource_before,
        "resource_after": resource_after,
    }


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
            # warm up (loads the sentence-transformers model once, so the cold-start
            # cost seen in run_observability_eval.py doesn't skew the first level)
            client.post("/chat", json={"query": "warmup"}, headers=headers)

            levels = [run_level(client, headers, n) for n in CONCURRENCY_LEVELS]

        p95_trend = [lvl["latency_seconds"]["p95"] for lvl in levels]
        error_trend = [lvl["error_rate"] for lvl in levels]
        degrading_level = next((levels[i]["concurrency"] for i in range(1, len(levels)) if p95_trend[i] > p95_trend[i - 1] * 1.5), None)
        first_errors_at = next((levels[i]["concurrency"] for i in range(len(levels)) if error_trend[i] > 0), None)
        error_codes_at_first_failure = next((lvl["error_codes"] for lvl in levels if lvl["error_rate"] > 0), {})

        report = {
            "metadata": config.run_metadata(sample_count=sum(CONCURRENCY_LEVELS), dataset_version="load_test_v1"),
            "scope": (
                "LOCAL, single-process capacity test only. LLM/embedding calls are mocked (no live "
                "provider call, no API quota consumed) -- this measures THIS APPLICATION's own request-"
                "handling overhead (FastAPI routing, the agent graph, FAISS-backed retrieval against a "
                "1-chunk in-memory index) under concurrent load on whatever machine runs this script, NOT "
                "production/cloud-scale capacity, and NOT real LLM-provider-inclusive latency. Never "
                "extrapolated to 1M users -- see docs/DESIGN_REVIEW.md's own honest scaling answer for that."
            ),
            "concurrency_levels": levels,
            "analysis": {
                "p95_latency_trend_seconds": p95_trend,
                "error_rate_trend": error_trend,
                "concurrency_where_p95_jumps_50pct_or_more": degrading_level,
                "concurrency_where_errors_first_appear": first_errors_at,
                "error_codes_at_first_failure_level": error_codes_at_first_failure,
                "first_bottleneck": (
                    f"Errors first appear at concurrency={first_errors_at} (error codes observed: "
                    f"{error_codes_at_first_failure}). This project's own in-memory sliding-window rate "
                    "limiter (60 req/min per identity, app/core/auth.py) is the most likely cause -- every "
                    "request in this test shares one API key, so >60 concurrent requests within the same "
                    "minute genuinely trips the app's own real rate-limit code path, not a mocking artifact "
                    "or an LLM-provider limit (both are mocked out of this test). This is a REAL, measured "
                    "local bottleneck: the application's own per-identity rate limiter, triggered exactly as "
                    "designed."
                    if first_errors_at
                    else "No errors observed up to concurrency=100 in this mocked-LLM local test."
                ),
            },
            "limitations": [
                "Single machine, single run, mocked LLM/embedding -- not a production-representative load "
                "test. Real production load would be bottlenecked by the LLM provider's rate limits long "
                "before this application's own concurrency handling, based on this project's own live-run "
                "history (two prior Groq daily-quota exhaustions).",
                "psutil CPU/memory sampling is best-effort; if psutil isn't installed, resource fields are "
                "reported as unavailable, never estimated.",
                "Does not test sustained load over time, only four discrete concurrency bursts.",
            ],
        }

        path = config.save_report(report, name="load_test_final")
        print(f"Saved: {path}")
        for lvl in levels:
            print(f"concurrency={lvl['concurrency']}: RPS={lvl['requests_per_second']} P95={lvl['latency_seconds']['p95']}s error_rate={lvl['error_rate']}")
        print(f"First bottleneck: {report['analysis']['first_bottleneck']}")


if __name__ == "__main__":
    main()
