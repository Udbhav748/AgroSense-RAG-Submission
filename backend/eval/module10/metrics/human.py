"""Human evaluation aggregation: per-dimension mean/median/stdev, and
Inter-Annotator Agreement ONLY when 2+ reviewers actually scored the same
items — never fabricated, never silently assumed.

Reuses docs/HUMAN_EVAL.md's own 7-dimension rubric (Correctness,
Helpfulness, Completeness, Safety, Tone, Groundedness, Citation Quality) —
this module just aggregates scores, it does not redefine the rubric.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

RUBRIC_DIMENSIONS = [
    "correctness",
    "helpfulness",
    "completeness",
    "safety",
    "tone",
    "groundedness",
    "citation_quality",
]

__all__ = [
    "RUBRIC_DIMENSIONS",
    "ReviewerScore",
    "aggregate_scores",
    "inter_annotator_agreement",
    "weighted_cohens_kappa",
    "kappa_by_dimension",
]


@dataclass
class ReviewerScore:
    case_id: str
    reviewer_id: str
    scores: dict[str, float | None]  # dimension -> 1-5, or None for N/A (e.g. conversational)


def aggregate_scores(scores: list[ReviewerScore]) -> dict:
    """Mean/median/stdev per dimension across every scored (case, reviewer)
    pair. stdev requires >=2 data points for a dimension or is reported as
    None (statistics.stdev raises on n<2 — caught explicitly, not
    swallowed into a fabricated 0.0)."""
    per_dimension: dict[str, list[float]] = {d: [] for d in RUBRIC_DIMENSIONS}
    for s in scores:
        for dim, value in s.scores.items():
            if value is not None and dim in per_dimension:
                per_dimension[dim].append(value)

    result = {}
    for dim, values in per_dimension.items():
        if not values:
            result[dim] = {"mean": None, "median": None, "stdev": None, "n": 0}
            continue
        result[dim] = {
            "mean": round(statistics.mean(values), 3),
            "median": round(statistics.median(values), 3),
            "stdev": round(statistics.stdev(values), 3) if len(values) >= 2 else None,
            "n": len(values),
        }
    return result


def inter_annotator_agreement(scores: list[ReviewerScore]) -> dict:
    """Per-dimension IAA (mean absolute pairwise score difference, and its
    complement as a 0-1 'agreement' figure: 1 - avg_diff/4, since scores
    are 1-5) computed ONLY over case_ids that have 2+ distinct reviewers.
    If no case has 2+ reviewers (the common situation in this project per
    docs/HUMAN_EVAL.md), returns the honest "not available" shape rather
    than a fabricated number — matching this project's existing, explicit
    convention (docs/HUMAN_EVAL.md: 'IAA = not available, reason = one
    reviewer')."""
    by_case: dict[str, list[ReviewerScore]] = {}
    for s in scores:
        by_case.setdefault(s.case_id, []).append(s)

    multi_reviewer_cases = {cid: group for cid, group in by_case.items() if len({s.reviewer_id for s in group}) >= 2}

    if not multi_reviewer_cases:
        reviewers = {s.reviewer_id for s in scores}
        return {
            "iaa": "not available",
            "reason": f"one reviewer ({len(reviewers)} reviewer(s) total, 0 cases with 2+ independent scores)",
        }

    per_dimension_diffs: dict[str, list[float]] = {d: [] for d in RUBRIC_DIMENSIONS}
    for _cid, group in multi_reviewer_cases.items():
        for dim in RUBRIC_DIMENSIONS:
            values = [s.scores.get(dim) for s in group if s.scores.get(dim) is not None]
            if len(values) < 2:
                continue
            # mean absolute pairwise difference across all reviewer pairs for this case+dimension
            diffs = [abs(a - b) for i, a in enumerate(values) for b in values[i + 1 :]]
            per_dimension_diffs[dim].extend(diffs)

    per_dimension_iaa = {}
    for dim, diffs in per_dimension_diffs.items():
        if not diffs:
            per_dimension_iaa[dim] = None
            continue
        avg_diff = sum(diffs) / len(diffs)
        per_dimension_iaa[dim] = round(1 - (avg_diff / 4), 4)  # 4 = max possible diff on a 1-5 scale

    return {
        "iaa": "computed",
        "n_multi_reviewer_cases": len(multi_reviewer_cases),
        "per_dimension_agreement": per_dimension_iaa,
    }


def weighted_cohens_kappa(ratings_a: list[int], ratings_b: list[int], *, categories: range | None = None) -> float | None:
    """Quadratic-weighted Cohen's kappa for two raters' ordinal ratings
    (Module 10 gap-closure, P8) -- the standard chance-corrected
    agreement statistic for 1-5 Likert-style ordinal data, preferred
    here over inter_annotator_agreement()'s simpler mean-absolute-
    pairwise-difference figure (kept above, unchanged, as an additional,
    already-existing metric) because kappa corrects for the agreement
    two raters would reach by chance alone.

    ratings_a/ratings_b must be equal-length, paired (same case order),
    with N/A entries already filtered out by the caller -- this function
    computes over whatever it's given, it does not know about case_ids.

    Quadratic weights: disagreement penalty grows with the SQUARE of the
    distance between two ratings (a 1-vs-5 disagreement counts far more
    than a 4-vs-5 one) -- the standard choice for ordinal (not purely
    nominal) categories, matching how a 1-5 rubric is actually used.

    Returns None (not 0.0 or NaN) when there are fewer than 2 paired
    ratings -- kappa is undefined, not zero, with an empty/singleton
    sample.

    Formula (standard, e.g. Cohen 1968 / scikit-learn's
    cohen_kappa_score(weights="quadratic")): kappa = 1 - (weighted
    observed disagreement / weighted expected disagreement under
    independence).
    """
    if len(ratings_a) != len(ratings_b):
        raise ValueError(f"ratings_a and ratings_b must be equal length, got {len(ratings_a)} and {len(ratings_b)}")
    n = len(ratings_a)
    if n < 2:
        return None

    if categories is None:
        all_values = set(ratings_a) | set(ratings_b)
        categories = range(min(all_values), max(all_values) + 1)
    cats = list(categories)
    k = len(cats)
    cat_index = {c: i for i, c in enumerate(cats)}

    # Observed confusion matrix
    observed = [[0] * k for _ in range(k)]
    for a, b in zip(ratings_a, ratings_b):
        observed[cat_index[a]][cat_index[b]] += 1

    # Marginals
    row_totals = [sum(row) for row in observed]
    col_totals = [sum(observed[r][c] for r in range(k)) for c in range(k)]

    # Quadratic weights: w[i][j] = (i - j)^2 / (k - 1)^2
    denom = (k - 1) ** 2 if k > 1 else 1
    weights = [[((i - j) ** 2) / denom for j in range(k)] for i in range(k)]

    observed_disagreement = sum(
        weights[i][j] * observed[i][j] for i in range(k) for j in range(k)
    ) / n
    expected_disagreement = sum(
        weights[i][j] * (row_totals[i] * col_totals[j]) / n for i in range(k) for j in range(k)
    ) / n

    if expected_disagreement == 0:
        # Both raters agree on every single item AND used only one
        # category between them -- no variance to correct for chance
        # against. Perfect agreement, not undefined.
        return 1.0

    return round(1 - (observed_disagreement / expected_disagreement), 4)


def kappa_by_dimension(scores: list[ReviewerScore]) -> dict:
    """Weighted Cohen's kappa per rubric dimension, computed only over
    case_ids that have ratings from exactly 2 distinct reviewers for
    that dimension (N/A entries excluded). Returns the honest
    "not available" shape (matching inter_annotator_agreement()'s own
    convention) when fewer than 2 reviewers scored any shared case.
    """
    by_case: dict[str, dict[str, ReviewerScore]] = {}
    for s in scores:
        by_case.setdefault(s.case_id, {})[s.reviewer_id] = s

    two_reviewer_cases = {cid: revs for cid, revs in by_case.items() if len(revs) >= 2}
    if not two_reviewer_cases:
        reviewers = {s.reviewer_id for s in scores}
        return {
            "kappa": "not available",
            "reason": f"fewer than 2 reviewers share any case ({len(reviewers)} reviewer(s) total)",
        }

    # Only the first two distinct reviewers per case are used for a
    # pairwise kappa -- this project's design is exactly two reviewers,
    # so this is not a simplification that loses data in practice.
    result: dict[str, dict] = {}
    per_dim_pairs: dict[str, tuple[list[int], list[int]]] = {d: ([], []) for d in RUBRIC_DIMENSIONS}
    n_missing_by_dim: dict[str, int] = dict.fromkeys(RUBRIC_DIMENSIONS, 0)

    for _cid, revs in two_reviewer_cases.items():
        reviewer_ids = sorted(revs)[:2]
        s_a, s_b = revs[reviewer_ids[0]], revs[reviewer_ids[1]]
        for dim in RUBRIC_DIMENSIONS:
            va, vb = s_a.scores.get(dim), s_b.scores.get(dim)
            if va is None or vb is None:
                n_missing_by_dim[dim] += 1
                continue
            per_dim_pairs[dim][0].append(int(va))
            per_dim_pairs[dim][1].append(int(vb))

    for dim, (a_vals, b_vals) in per_dim_pairs.items():
        kappa = weighted_cohens_kappa(a_vals, b_vals, categories=range(1, 6))
        result[dim] = {
            "kappa": kappa,
            "n_paired_ratings": len(a_vals),
            "n_missing_or_na": n_missing_by_dim[dim],
        }

    return {
        "kappa": "computed",
        "n_cases_with_two_reviewers": len(two_reviewer_cases),
        "per_dimension_kappa": result,
        "interpretation": "Quadratic-weighted Cohen's kappa: 1.0 = perfect agreement, 0.0 = agreement no "
        "better than chance, negative = worse than chance. No subjective label (e.g. 'excellent "
        "agreement') is attached -- report the numeric coefficient and let the reader apply whatever "
        "threshold convention they use.",
    }
