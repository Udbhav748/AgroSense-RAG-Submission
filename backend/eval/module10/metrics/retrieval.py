"""Retrieval metrics: Precision@K, Recall@K, Hit Rate@K, MRR.

Thin re-export of backend/eval/run_eval.py's existing implementations
(precision_at_k/recall_at_k/reciprocal_rank/hit_at_k, run_eval.py:187-236)
— relevance is defined identically (a chunk is relevant if it contains any
of a case's expected_chunk_keywords; see rag_eval.json's own _schema_note
for why keyword-matching, not chunk-ID matching, is this project's honest
choice given random-UUID-per-chunk ingestion). Nothing here recomputes
that logic a second, possibly-divergent way.
"""

from __future__ import annotations

from dataclasses import dataclass

from eval.run_eval import RETRIEVAL_EVAL_K, hit_at_k, precision_at_k, recall_at_k, reciprocal_rank

__all__ = [
    "RETRIEVAL_EVAL_K",
    "CaseRetrievalResult",
    "aggregate",
    "hit_at_k",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "score_case",
]


@dataclass
class CaseRetrievalResult:
    case_id: str
    retrieved_ids: list[str]
    gold_keywords: list[str]
    precision_at_5: float
    recall_at_5: float
    hit_at_5: bool
    reciprocal_rank: float


def score_case(case_id: str, chunks, keywords: list[str], k: int = RETRIEVAL_EVAL_K) -> CaseRetrievalResult:
    """Score one retrieval result against one case's expected_chunk_keywords.
    `chunks` is the list[RetrievedChunk] actually returned by retrieve()."""
    return CaseRetrievalResult(
        case_id=case_id,
        retrieved_ids=[c.chunk_id for c in chunks[:k]],
        gold_keywords=list(keywords),
        precision_at_5=precision_at_k(chunks, keywords, k),
        recall_at_5=recall_at_k(chunks, keywords, k),
        hit_at_5=hit_at_k(chunks, keywords, k),
        reciprocal_rank=reciprocal_rank(chunks, keywords, k),
    )


def aggregate(results: list[CaseRetrievalResult]) -> dict:
    """Mean P@5/Recall@5/MRR + Hit Rate@5 across cases that actually had
    gold keywords to score against (out-of-corpus/expected-fallback cases
    are excluded from the denominator — there's nothing to retrieve
    correctly by design, so folding them in would silently deflate the
    metric rather than reflect a real retrieval miss)."""
    scoreable = [r for r in results if r.gold_keywords]
    n = len(scoreable)
    if n == 0:
        return {
            "mean_precision_at_5": None,
            "mean_recall_at_5": None,
            "hit_rate_at_5": None,
            "mrr": None,
            "n_scoreable": 0,
            "n_total": len(results),
        }
    return {
        "mean_precision_at_5": round(sum(r.precision_at_5 for r in scoreable) / n, 4),
        "mean_recall_at_5": round(sum(r.recall_at_5 for r in scoreable) / n, 4),
        "hit_rate_at_5": round(sum(1 for r in scoreable if r.hit_at_5) / n, 4),
        "mrr": round(sum(r.reciprocal_rank for r in scoreable) / n, 4),
        "n_scoreable": n,
        "n_total": len(results),
    }
