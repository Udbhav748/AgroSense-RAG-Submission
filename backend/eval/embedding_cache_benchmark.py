"""Latency evidence for embedding_service.embed_query's LRU cache.

Sends the same query through embed_query() twice and times each call
separately: the first is a genuine cache miss (a real model.encode()
call), the second a cache hit (a dict/tuple lookup). Repeats over a few
distinct queries so the numbers aren't just one lucky sample, and prints
the miss/hit latency and the speedup ratio for each, plus an average.

Only needs the embedding model loaded -- no vector store, no LLM, no
network call, no indexed document precondition.

Usage (from backend/):
    python eval/embedding_cache_benchmark.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.embedding_service import _embed_query_cached, embed_query  # noqa: E402

QUERIES = [
    "What is a project according to the document?",
    "What is a Work Breakdown Structure?",
    "If a project's CPI is greater than 1, is that good or bad?",
    "What is Earned Value Management used for?",
]


def timed_call(query: str) -> float:
    start = time.perf_counter()
    embed_query(query)
    return time.perf_counter() - start


def main() -> None:
    # Force a cold start for the *model itself* before timing anything --
    # otherwise the first query's "miss" would unfairly include model-load
    # time, which has nothing to do with the embedding cache being
    # benchmarked here.
    print("Loading embedding model (untimed)...")
    embed_query("warm up the model, not the cache")
    _embed_query_cached.cache_clear()

    print(f"\n{'query':<55s}{'miss (s)':>12s}{'hit (s)':>12s}{'speedup':>12s}")
    print("-" * 91)

    miss_times = []
    hit_times = []

    for query in QUERIES:
        miss_time = timed_call(query)  # first call: genuine cache miss
        hit_time = timed_call(query)  # second call, same query: cache hit
        miss_times.append(miss_time)
        hit_times.append(hit_time)
        speedup = miss_time / hit_time if hit_time > 0 else float("inf")
        print(f"{query[:53]:<55s}{miss_time:>12.6f}{hit_time:>12.6f}{speedup:>11.1f}x")

    avg_miss = sum(miss_times) / len(miss_times)
    avg_hit = sum(hit_times) / len(hit_times)
    avg_speedup = avg_miss / avg_hit if avg_hit > 0 else float("inf")

    print("-" * 91)
    print(f"{'AVERAGE':<55s}{avg_miss:>12.6f}{avg_hit:>12.6f}{avg_speedup:>11.1f}x")
    print(
        f"\nCache-hit latency is {(1 - avg_hit / avg_miss) * 100:.1f}% lower than "
        f"cache-miss latency, averaged over {len(QUERIES)} distinct queries."
    )


if __name__ == "__main__":
    main()
