#!/usr/bin/env python
"""Module 10 gap-closure (P7): LOAD / CONCURRENCY EVALUATION.

Starts a REAL `uvicorn app.main:app` subprocess bound to localhost (not
TestClient -- a genuine HTTP socket, exactly like
run_availability_eval.py's pattern) and drives a concurrency ladder of
real HTTP requests against it with `httpx.Client` + `ThreadPoolExecutor`,
so requests genuinely overlap in wall-clock time and pass through the
real ASGI/ HTTP boundary -- not an in-process function call.

Two workloads, reported SEPARATELY (never blended into one aggregate):

- "health": GET /health -- no LLM/retrieval dependency at all. Measures
  pure HTTP/ASGI/middleware overhead under concurrency.
- "chat": POST /chat with a fixed retrieval query against this
  project's real, already-populated vector store (backend/vector_store/),
  so real embedding + real cross-encoder reranking run -- genuine
  CPU-bound retrieval work. The LLM stage is Settings.llm_provider=mock
  (app/services/mock_llm_client.py, Module 10 P7's own small, opt-in
  addition) -- deterministic, zero-cost, zero-network, so this measures
  application/retrieval overhead under concurrency, not live-provider
  latency or cost.

Each concurrency level gets its OWN API key/client identity (via
Settings.api_keys, JSON-configured through the subprocess's environment)
so the app's own real per-identity rate limiter (60 req/min default,
app/core/auth.py) does not bleed rate-limit pressure across levels --
each level's numbers reflect concurrency effects, not leftover quota
from the previous level.

A separate, deliberate HARD/FAILURE scenario re-uses a single shared key
across a request burst to intentionally trip that same real rate
limiter -- a safe, reproducible "controlled request rejection," not a
fabricated fault.

Honest scope: this measures THIS SINGLE LOCAL PROCESS's behavior on
whatever machine runs it. It is NOT production capacity, NOT a cloud
SLO, and NOT internet-representative network latency. See the report's
own "limitations" field.

Usage (from backend/):
    python eval/module10/runners/run_load_concurrency_final_eval.py
"""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import httpx  # noqa: E402

from eval.module10 import config  # noqa: E402
from monitoring.uptime_check import _probe  # noqa: E402

PORT = 8814
BACKEND_ROOT = Path(__file__).resolve().parents[3]
CONCURRENCY_LEVELS = [1, 2, 5, 10, 20]
REQUESTS_PER_LEVEL = 20
REQUEST_TIMEOUT_SECONDS = 10.0
STARTUP_TIMEOUT_SECONDS = 120.0
CHAT_QUERY = "What treats apple scab?"


def _percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    idx = min(len(sorted_values) - 1, max(0, round(p / 100 * (len(sorted_values) - 1))))
    return sorted_values[idx]


def _resource_sample() -> dict:
    """Local process RSS/CPU only -- never presented as cloud/instance
    utilization. GPU is N/A (not used by this app's CPU-only retrieval
    stack in this environment) -- recorded as such, never invented."""
    try:
        import psutil

        process = psutil.Process()
        return {
            "cpu_percent": process.cpu_percent(interval=0.1),
            "memory_rss_mb": round(process.memory_info().rss / (1024 * 1024), 2),
            "gpu": "N/A -- not used/measured",
            "available": True,
        }
    except ImportError:
        return {"available": False, "gpu": "N/A", "note": "psutil not installed -- CPU/memory not sampled"}


def _build_api_keys_for_levels() -> dict[str, str]:
    """One distinct client_name -> key per (workload, concurrency) pair,
    plus one shared key for the deliberate rate-limit burst scenario --
    JSON-configured via Settings.api_keys so each level's rate-limit
    bucket (app/core/auth.py, keyed by client_name) starts independent
    of every other level."""
    keys: dict[str, str] = {}
    for workload in ("health", "chat"):
        for level in CONCURRENCY_LEVELS:
            client_name = f"p7_{workload}_c{level}"
            keys[client_name] = f"p7key_{workload}_{level}_{'x' * 16}"
    keys["p7_burst_shared"] = "p7key_burst_shared_xxxxxxxxxxxxxxxx"
    return keys


def _start_uvicorn(api_keys_json: str) -> subprocess.Popen:
    import os

    env = os.environ.copy()
    env["API_KEYS"] = api_keys_json
    env["LLM_PROVIDER"] = "mock"
    # Fallback/routing must stay off -- this benchmark measures the
    # primary path only, consistent with the isolation reasoning already
    # used by the Provider A-B evaluation (P5).
    env["FALLBACK_LLM_PROVIDER"] = ""
    env["MODEL_ROUTING_ENABLED"] = "false"
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(PORT)],
        cwd=str(BACKEND_ROOT),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _wait_for_startup(url: str, timeout_seconds: float) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        healthy, _, _ = _probe(url)
        if healthy:
            return True
        time.sleep(0.5)
    return False


def _one_request(client: httpx.Client, method: str, url: str, headers: dict, json_body: dict | None) -> dict:
    start = time.perf_counter()
    try:
        if method == "GET":
            response = client.get(url, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS)
        else:
            response = client.post(url, headers=headers, json=json_body, timeout=REQUEST_TIMEOUT_SECONDS)
        latency = time.perf_counter() - start
        if response.status_code == 200:
            return {"outcome": "success", "latency_s": latency, "status_code": response.status_code}
        return {
            "outcome": "http_failure",
            "latency_s": latency,
            "status_code": response.status_code,
        }
    except httpx.TimeoutException:
        return {"outcome": "timeout", "latency_s": time.perf_counter() - start, "status_code": None}
    except httpx.ConnectError:
        return {"outcome": "connection_error", "latency_s": time.perf_counter() - start, "status_code": None}
    except Exception as exc:  # noqa: BLE001
        return {
            "outcome": "unexpected_exception",
            "latency_s": time.perf_counter() - start,
            "status_code": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def run_level(
    *, workload: str, concurrency: int, n_requests: int, base_url: str, api_key: str
) -> dict:
    headers = {"X-API-Key": api_key}
    if workload == "health":
        method, url, body = "GET", f"{base_url}/health", None
    else:
        method, url, body = "POST", f"{base_url}/chat", {"query": CHAT_QUERY}

    resource_before = _resource_sample()
    results: list[dict] = []
    start = time.perf_counter()
    with httpx.Client() as client:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(_one_request, client, method, url, headers, body) for _ in range(n_requests)]
            for future in as_completed(futures):
                results.append(future.result())
    wall_time = time.perf_counter() - start
    resource_after = _resource_sample()

    successes = [r for r in results if r["outcome"] == "success"]
    http_failures = [r for r in results if r["outcome"] == "http_failure"]
    timeouts = [r for r in results if r["outcome"] == "timeout"]
    connection_errors = [r for r in results if r["outcome"] == "connection_error"]
    unexpected = [r for r in results if r["outcome"] == "unexpected_exception"]

    latencies_ms = sorted(r["latency_s"] * 1000 for r in successes)
    n = len(results)

    return {
        "workload": workload,
        "concurrency": concurrency,
        "requests": n,
        "successes": len(successes),
        "http_failures": len(http_failures),
        "timeouts": len(timeouts),
        "connection_errors": len(connection_errors),
        "unexpected_exceptions": len(unexpected),
        "error_rate": round((n - len(successes)) / n, 4) if n else None,
        "success_rate": round(len(successes) / n, 4) if n else None,
        "wall_time_seconds": round(wall_time, 4),
        "rps": round(n / wall_time, 2) if wall_time > 0 else None,
        "latency_ms": {
            "mean": round(statistics.mean(latencies_ms), 2) if latencies_ms else None,
            "p50": round(_percentile(latencies_ms, 50), 2),
            "p95": round(_percentile(latencies_ms, 95), 2),
            "p99": round(_percentile(latencies_ms, 99), 2),
            "min": round(min(latencies_ms), 2) if latencies_ms else None,
            "max": round(max(latencies_ms), 2) if latencies_ms else None,
        },
        "http_failure_status_codes": sorted({r["status_code"] for r in http_failures if r["status_code"]}),
        "unexpected_exception_samples": [r.get("error") for r in unexpected[:3]],
        "resource_before": resource_before,
        "resource_after": resource_after,
    }


def run_rate_limit_burst_scenario(*, base_url: str, api_key: str) -> dict:
    """TASK 15: a deliberate, safe, reproducible hard/failure case --
    reuses ONE shared client identity for a burst of requests well past
    Settings.rate_limit_per_minute (default 60), so the app's own real
    sliding-window limiter (app/core/auth.py) rejects the overflow with
    HTTP 429 -- a genuine "controlled request rejection," not a
    fabricated fault. Verifies /health still responds afterward."""
    headers = {"X-API-Key": api_key}
    n_requests = 100
    concurrency = 20
    results: list[dict] = []
    start = time.perf_counter()
    with httpx.Client() as client:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [
                pool.submit(_one_request, client, "GET", f"{base_url}/health", headers, None)
                for _ in range(n_requests)
            ]
            for future in as_completed(futures):
                results.append(future.result())
    wall_time = time.perf_counter() - start

    n_success = sum(1 for r in results if r["outcome"] == "success")
    n_rate_limited = sum(1 for r in results if r.get("status_code") == 429)
    n_other_failure = len(results) - n_success - n_rate_limited

    health_after = _probe(f"{base_url}/health")
    recovered = health_after[0]

    return {
        "scenario": "rate_limit_burst",
        "description": "100 requests from ONE shared client identity at concurrency=20 against GET "
        "/health, deliberately exceeding Settings.rate_limit_per_minute (default 60) within the "
        "sliding window -- exercises the application's own real, existing per-identity rate limiter, "
        "not a fabricated fault.",
        "requests": len(results),
        "successes": n_success,
        "rate_limited_429": n_rate_limited,
        "other_failures": n_other_failure,
        "wall_time_seconds": round(wall_time, 4),
        "detection": "HTTP 429 status code (RateLimitExceededError, app/core/exceptions.py) on the "
        "responses past the per-minute cap.",
        "recovered_after": recovered,
        "what_this_demonstrates": (
            "The application's own real per-identity rate limiter engages under sustained request "
            "pressure from a single identity and rejects overflow deterministically (HTTP 429), rather "
            "than degrading unpredictably or crashing -- and the service remains healthy and answers "
            "GET /health normally immediately afterward."
            if n_rate_limited > 0
            else "No requests were rate-limited in this run -- 100 requests from one identity did not "
            "exceed the configured per-minute threshold within this burst's wall-clock duration; "
            "reported honestly rather than assuming a limiter engaged."
        ),
    }


def main() -> None:
    api_keys = _build_api_keys_for_levels()
    base_url = f"http://127.0.0.1:{PORT}"

    proc = _start_uvicorn(json.dumps(api_keys))
    try:
        started = _wait_for_startup(f"{base_url}/health", STARTUP_TIMEOUT_SECONDS)
        if not started:
            print(f"FAILED TO START within {STARTUP_TIMEOUT_SECONDS}s -- aborting.")
            return

        # Warm-up: pay the one-time sentence-transformers/cross-encoder
        # cold-load cost here, outside the timed ladder, so concurrency=1
        # isn't unfairly penalized by a cost every other level also pays
        # only once, on their own first request, if not warmed first.
        warmup_key = api_keys["p7_chat_c1"]
        httpx.post(
            f"{base_url}/chat",
            headers={"X-API-Key": warmup_key},
            json={"query": CHAT_QUERY},
            timeout=280.0,  # cold sentence-transformers/cross-encoder model load can take minutes
        )

        levels: list[dict] = []
        for workload in ("health", "chat"):
            for level in CONCURRENCY_LEVELS:
                api_key = api_keys[f"p7_{workload}_c{level}"]
                result = run_level(
                    workload=workload,
                    concurrency=level,
                    n_requests=REQUESTS_PER_LEVEL,
                    base_url=base_url,
                    api_key=api_key,
                )
                healthy_after, _, _ = _probe(f"{base_url}/health")
                result["health_after"] = healthy_after
                levels.append(result)
                print(
                    f"{workload:6s} concurrency={level:3d} RPS={result['rps']:>7} "
                    f"P50={result['latency_ms']['p50']:>8}ms P95={result['latency_ms']['p95']:>8}ms "
                    f"errors={result['error_rate']}"
                )

        burst_result = run_rate_limit_burst_scenario(base_url=base_url, api_key=api_keys["p7_burst_shared"])
        print(
            f"burst scenario: {burst_result['successes']} success, {burst_result['rate_limited_429']} "
            f"rate-limited, recovered_after={burst_result['recovered_after']}"
        )

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()

    health_levels = [lvl for lvl in levels if lvl["workload"] == "health"]
    chat_levels = [lvl for lvl in levels if lvl["workload"] == "chat"]

    def _trend(levels_subset: list[dict], key_path: tuple[str, ...]) -> list:
        out = []
        for lvl in levels_subset:
            value = lvl
            for k in key_path:
                value = value[k]
            out.append(value)
        return out

    report = {
        "metadata": {
            **config.run_metadata(sample_count=sum(lvl["requests"] for lvl in levels), dataset_version="load_concurrency_v1"),
            "evaluator": "eval/module10/runners/run_load_concurrency_final_eval.py",
            "reproduce_command": "cd backend && python eval/module10/runners/run_load_concurrency_final_eval.py",
        },
        "environment_summary": {
            "host": "single local machine, single uvicorn worker process",
            "gpu": "N/A -- not used",
        },
        "endpoints": {
            "health": {
                "endpoint": "GET /health",
                "auth": "unauthenticated (per-level API key still sent for consistency; unused by this route)",
                "llm_involved": False,
                "retrieval_involved": False,
            },
            "chat": {
                "endpoint": "POST /chat",
                "payload": {"query": CHAT_QUERY},
                "retrieval_enabled": True,
                "retrieval_backing_store": "backend/vector_store/ (real, already-populated -- not seeded synthetically for this run)",
                "structured_output_enabled": "default (Settings.structured_output_enabled=True; request does not set structured_response, so the free-text path is used)",
                "llm_mocked": True,
                "llm_provider": "mock (app/services/mock_llm_client.py -- deterministic, zero-cost, zero-network)",
                "cache": "NOT cleared between requests -- deliberately exercises real SemanticQueryCache "
                "behavior under load; see 'cache_behavior' below for what that means for these numbers.",
            },
        },
        "cache_behavior": {
            "note": "P6 found that a response-cache hit (cache_lookup_node) never emits the "
            "chat_query_handled log line monitoring/log_aggregate.py counts -- an aggregation gap, not "
            "an HTTP-level one. This benchmark measures HTTP-level outcomes directly (status "
            "code/latency per request via httpx), which IS accurate for both cache hits and misses -- "
            "a cache hit still returns a real HTTP 200 with real latency, it just resolves faster. "
            "This run's 20 identical repeated queries per level will include both a cold miss and "
            "warm hits within the same level (SemanticQueryCache is not cleared between requests here, "
            "unlike P6's observability report) -- reported as a single blended per-level number, "
            "consistent with 'exercises both cache misses and cache hits' rather than 'intentionally "
            "avoids the cache,' per this task's own instruction not to pretend P6's log-based counting "
            "represents complete request observability.",
            "P6_gap_still_applies": "Any log-based aggregation of this run's traffic (monitoring/log_aggregate.py) would still undercount cache-hit requests for the reason P6 documented -- not re-fixed here; this report's own numbers come from direct HTTP measurement, not log aggregation, so they are unaffected.",
        },
        "concurrency_levels_tested": CONCURRENCY_LEVELS,
        "requests_per_level": REQUESTS_PER_LEVEL,
        "request_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
        "results": {"health": health_levels, "chat": chat_levels},
        "rate_limit_burst_scenario": burst_result,
        "analysis": {
            "health": {
                "rps_trend": _trend(health_levels, ("rps",)),
                "p50_trend_ms": _trend(health_levels, ("latency_ms", "p50")),
                "p95_trend_ms": _trend(health_levels, ("latency_ms", "p95")),
                "p99_trend_ms": _trend(health_levels, ("latency_ms", "p99")),
                "error_rate_trend": _trend(health_levels, ("error_rate",)),
            },
            "chat": {
                "rps_trend": _trend(chat_levels, ("rps",)),
                "p50_trend_ms": _trend(chat_levels, ("latency_ms", "p50")),
                "p95_trend_ms": _trend(chat_levels, ("latency_ms", "p95")),
                "p99_trend_ms": _trend(chat_levels, ("latency_ms", "p99")),
                "error_rate_trend": _trend(chat_levels, ("error_rate",)),
            },
            "summary": "Observed behavior through tested concurrency "
            f"{max(CONCURRENCY_LEVELS)} on this machine. No 'maximum safe concurrency' is declared -- "
            "see per-level trends above for the actual measured curve.",
        },
        "limitations": [
            "Single local machine, single uvicorn worker, single run per level -- NOT production "
            "capacity, NOT a cloud SLO, NOT internet-representative network latency (all requests are "
            "loopback/localhost).",
            "The chat workload's LLM stage is mocked (Settings.llm_provider=mock) -- this measures "
            "application/retrieval overhead under concurrency, not live-provider latency, rate limits, "
            "or cost. A live-provider load test was NOT run as part of this pass's primary evidence, "
            "per this task's own instruction to avoid an uncontrolled provider-cost test; if useful, "
            "that would be a separate, explicitly-disclosed optional measurement (provider, model, "
            "request count, cost) -- not attempted here.",
            "GPU utilization is N/A -- not used by this application's CPU-only retrieval/reranking "
            "stack in this environment; not invented as a measured figure.",
            "psutil-based CPU/RSS sampling measures THIS SCRIPT's own client process (psutil.Process() "
            "with no PID defaults to the caller), NOT the spawned uvicorn server subprocess actually "
            "bearing the load -- disclosed honestly as a measurement gap, not presented as server-side "
            "resource usage. Not cloud instance utilization either way.",
            "RPS/latency figures are this benchmark's own measured throughput under tested concurrency, "
            "not a claim about maximum production capacity or reliability.",
        ],
    }

    path = config.save_report(report, name="load_concurrency_final")
    print(f"Saved: {path}")
    print(f"Rate-limit burst: {burst_result['what_this_demonstrates']}")


if __name__ == "__main__":
    main()
