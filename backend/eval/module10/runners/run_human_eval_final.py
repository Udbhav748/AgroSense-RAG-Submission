#!/usr/bin/env python
"""Module 10 gap-closure (P8): human evaluation + second reviewer +
Inter-Annotator Agreement.

Loads Reviewer 1's existing, real ratings
(eval/module10/human_eval/reviewer_1_ratings.json, transcribed verbatim
from docs/HUMAN_EVAL.md's already-scored table) and, if present, a real
Reviewer 2 ratings file (filled in independently from the blinded
packet -- eval/module10/human_eval/generate_reviewer2_packet.py).

If Reviewer 2 data is not present, this prints "SECOND REVIEWER DATA
REQUIRED" and computes only what a single reviewer supports (per-
dimension means/medians) -- it never fabricates a second reviewer's
scores, and never reports an IAA figure without two real, independent
reviewers.

Usage (from backend/):
    python eval/module10/runners/run_human_eval_final.py
    python eval/module10/runners/run_human_eval_final.py --reviewer2-file path/to/filled_in.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from eval.module10 import config  # noqa: E402
from eval.module10.human_eval.cases import CASES_BY_ID, DATASET_VERSION  # noqa: E402
from eval.module10.human_eval.schema import ReviewerDataError, validate_reviewer_file  # noqa: E402
from eval.module10.metrics.human import (  # noqa: E402
    RUBRIC_DIMENSIONS,
    ReviewerScore,
    aggregate_scores,
    inter_annotator_agreement,
    kappa_by_dimension,
)

HUMAN_EVAL_DIR = Path(__file__).resolve().parents[1] / "human_eval"
REVIEWER_1_PATH = HUMAN_EVAL_DIR / "reviewer_1_ratings.json"


def _load_reviewer_file(path: Path) -> list[ReviewerScore]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = validate_reviewer_file(data)  # raises ReviewerDataError with a specific message on any problem
    reviewer_id = data["reviewer_id"]
    return [ReviewerScore(case_id=row["case_id"], reviewer_id=reviewer_id, scores=row["ratings"]) for row in rows]


def _disagreement_stats(scores: list[ReviewerScore]) -> dict:
    by_case: dict[str, dict[str, ReviewerScore]] = {}
    for s in scores:
        by_case.setdefault(s.case_id, {})[s.reviewer_id] = s

    two_reviewer_cases = {cid: revs for cid, revs in by_case.items() if len(revs) >= 2}
    if not two_reviewer_cases:
        return {"available": False, "reason": "fewer than 2 reviewers share any case"}

    per_case_disagreement: list[dict] = []
    per_dimension_max_diff: dict[str, float] = dict.fromkeys(RUBRIC_DIMENSIONS, 0.0)
    dims_with_disagreement: dict[str, int] = dict.fromkeys(RUBRIC_DIMENSIONS, 0)

    for cid, revs in two_reviewer_cases.items():
        reviewer_ids = sorted(revs)[:2]
        s_a, s_b = revs[reviewer_ids[0]], revs[reviewer_ids[1]]
        per_dim_diff = {}
        total_diff = 0.0
        n_compared = 0
        for dim in RUBRIC_DIMENSIONS:
            va, vb = s_a.scores.get(dim), s_b.scores.get(dim)
            if va is None or vb is None:
                continue
            diff = abs(va - vb)
            per_dim_diff[dim] = diff
            total_diff += diff
            n_compared += 1
            per_dimension_max_diff[dim] = max(per_dimension_max_diff[dim], diff)
            if diff > 0:
                dims_with_disagreement[dim] += 1
        per_case_disagreement.append(
            {
                "case_id": cid,
                "per_dimension_diff": per_dim_diff,
                "mean_abs_diff": round(total_diff / n_compared, 3) if n_compared else None,
            }
        )

    per_case_disagreement.sort(key=lambda r: r["mean_abs_diff"] or 0.0, reverse=True)

    return {
        "available": True,
        "n_cases_compared": len(two_reviewer_cases),
        "hardest_disagreement_cases": per_case_disagreement[:5],
        "max_diff_by_dimension": per_dimension_max_diff,
        "cases_with_any_disagreement_by_dimension": dims_with_disagreement,
    }


def _find_hard_cases(scores: list[ReviewerScore]) -> dict:
    """TASK 11: lowest-scoring cases and cases with notable per-dimension
    disagreement, drawn only from actually-evaluated cases -- no
    artificial failure examples."""
    by_case: dict[str, list[ReviewerScore]] = {}
    for s in scores:
        by_case.setdefault(s.case_id, []).append(s)

    case_means: list[tuple[str, float]] = []
    for cid, group in by_case.items():
        values = [v for s in group for v in s.scores.values() if v is not None]
        if values:
            case_means.append((cid, sum(values) / len(values)))
    case_means.sort(key=lambda x: x[1])

    return {
        "lowest_scoring_cases": [
            {"case_id": cid, "mean_score_all_reviewers_all_dims": round(mean, 3), "case_type": CASES_BY_ID[cid].case_type}
            for cid, mean in case_means[:5]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reviewer2-file",
        type=Path,
        default=None,
        help="Path to a filled-in Reviewer 2 ratings file. If omitted, looks for a file matching "
        "eval/module10/human_eval/reviewer_2_ratings*.json.",
    )
    args = parser.parse_args()

    try:
        reviewer1_scores = _load_reviewer_file(REVIEWER_1_PATH)
    except ReviewerDataError as exc:
        print(f"Reviewer 1 data invalid: {exc}")
        sys.exit(1)
    print(f"Loaded Reviewer 1: {len(reviewer1_scores)} case ratings.")

    reviewer2_path = args.reviewer2_file
    if reviewer2_path is None:
        candidates = sorted(HUMAN_EVAL_DIR.glob("reviewer_2_ratings*.json"))
        reviewer2_path = candidates[-1] if candidates else None

    reviewer2_scores: list[ReviewerScore] = []
    reviewer2_status = "absent"
    if reviewer2_path is not None and reviewer2_path.is_file():
        try:
            reviewer2_scores = _load_reviewer_file(reviewer2_path)
            reviewer2_status = "valid"
            print(f"Loaded Reviewer 2 ({reviewer2_path.name}): {len(reviewer2_scores)} case ratings.")
        except ReviewerDataError as exc:
            reviewer2_status = f"invalid: {exc}"
            print(f"Reviewer 2 file found but INVALID ({reviewer2_path.name}): {exc}")

    all_scores = reviewer1_scores + reviewer2_scores

    if not reviewer2_scores:
        print()
        print("=" * 60)
        print("SECOND REVIEWER DATA REQUIRED")
        print("=" * 60)
        print(
            "No valid Reviewer 2 ratings file was found. IAA cannot be computed or claimed with only "
            "one reviewer. Generate the blinded packet and have an independent human reviewer fill it "
            "in:\n"
            "  cd backend && python eval/module10/human_eval/generate_reviewer2_packet.py\n"
            "Then re-run this command with:\n"
            "  cd backend && python eval/module10/runners/run_human_eval_final.py "
            "--reviewer2-file <path to filled-in reviewer 2 file>"
        )

    reviewer1_summary = aggregate_scores(reviewer1_scores)
    reviewer2_summary = aggregate_scores(reviewer2_scores) if reviewer2_scores else None
    combined_summary = aggregate_scores(all_scores) if reviewer2_scores else None
    simple_iaa = inter_annotator_agreement(all_scores)
    kappa_result = kappa_by_dimension(all_scores)
    disagreement = _disagreement_stats(all_scores) if reviewer2_scores else {"available": False, "reason": "one reviewer"}
    hard_cases = _find_hard_cases(all_scores)

    report = {
        "metadata": {
            **config.run_metadata(sample_count=len(CASES_BY_ID), dataset_version=DATASET_VERSION),
            "evaluator": "eval/module10/runners/run_human_eval_final.py",
        },
        "case_count": len(CASES_BY_ID),
        "reviewer_count": 2 if reviewer2_scores else 1,
        "reviewer_2_status": reviewer2_status,
        "rubric_version": "docs/HUMAN_EVAL.md's 7-dimension 1-5 rubric, unchanged by this pass",
        "reviewer_1_summary": reviewer1_summary,
        "reviewer_2_summary": reviewer2_summary,
        "combined_summary": combined_summary,
        "iaa_simple_mean_abs_diff": simple_iaa,
        "iaa_weighted_cohens_kappa": kappa_result,
        "disagreement_statistics": disagreement,
        "hard_cases": hard_cases,
        "privacy": {
            "pii_scan": "The 24 cases (queries + system outputs) contain no names, email addresses, "
            "phone numbers, or other personal data -- they are PMP-course-document and plant-pathology "
            "domain questions plus canned conversational replies. Case IDs (case_001..case_024) are used "
            "throughout rather than any reviewer-identifying information beyond a chosen reviewer_id "
            "string.",
        },
        "limitations": [
            "Reviewer 1's ratings were transcribed from docs/HUMAN_EVAL.md's existing markdown table, "
            "not re-scored -- a faithful transcription, not a new judgment pass.",
            "Blinding depends on the human reviewer actually following the documented protocol (not "
            "opening docs/HUMAN_EVAL.md before scoring) -- software cannot fully enforce this.",
            "Weighted Cohen's kappa on N=24 cases (or fewer per-dimension, where N/A entries reduce the "
            "paired sample) is a small-sample estimate -- no significance test is reported.",
            "IAA is computed only for dimensions/cases where both reviewers gave a non-null rating; "
            "N/A entries (e.g. Groundedness on conversational cases) are excluded from that dimension's "
            "pairing, not imputed.",
        ]
        + (["IAA CANNOT BE CLAIMED: only one reviewer's ratings are present."] if not reviewer2_scores else []),
    }

    path = config.save_report(report, name="human_eval_final")
    print(f"\nSaved: {path}")
    print(f"Case count: {report['case_count']}  Reviewer count: {report['reviewer_count']}")
    if reviewer2_scores:
        print("Weighted Cohen's kappa by dimension:")
        for dim, d in kappa_result.get("per_dimension_kappa", {}).items():
            print(f"  {dim}: kappa={d['kappa']}  n_paired={d['n_paired_ratings']}")
    else:
        print("IAA: not available (one reviewer) -- see 'SECOND REVIEWER DATA REQUIRED' above.")


if __name__ == "__main__":
    main()
