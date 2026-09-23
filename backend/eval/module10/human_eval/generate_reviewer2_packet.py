#!/usr/bin/env python
"""Module 10 gap-closure (P8): generates a blinded, self-contained JSON
scoring packet for an independent second human reviewer.

Blinding/independence measures:
- Reviewer 1's ratings are NEVER read or included -- this script does
  not even import reviewer_1_ratings.json.
- Case order is deterministically shuffled (seeded) so Reviewer 2 does
  not see the same case_001..case_024 order Reviewer 1's own scored
  table in docs/HUMAN_EVAL.md presents them in -- reduces order/anchor
  bias without making the packet non-reproducible.
- The packet is self-contained (query, system_output, evidence, rubric)
  so Reviewer 2 never needs to open docs/HUMAN_EVAL.md (which contains
  Reviewer 1's scores) to do the review.

Complete independence ultimately depends on the human reviewer actually
following the blinded protocol (not seeking out docs/HUMAN_EVAL.md
before scoring) -- this is a real, disclosed limitation, not something
software can fully enforce.

Usage (from backend/):
    python eval/module10/human_eval/generate_reviewer2_packet.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from eval.module10 import config  # noqa: E402
from eval.module10.human_eval.cases import CASES, DATASET_VERSION  # noqa: E402
from eval.module10.metrics.human import RUBRIC_DIMENSIONS  # noqa: E402

# Fixed seed -- the shuffle is deterministic/reproducible (re-running
# this script produces the identical blinded order), not randomized per
# run, which would make it impossible to regenerate the same packet.
BLIND_ORDER_SEED = 20260921

RUBRIC_TEXT = (
    "Score each dimension 1-5 (or null for Groundedness/Citation Quality on cases with no retrieval, "
    "e.g. conversational). See docs/HUMAN_EVAL.md's Rubric section for the full 1-5 anchor text per "
    "dimension. Do not open docs/HUMAN_EVAL.md's scored table or Inter-Annotator Agreement section "
    "until AFTER you finish scoring every case below -- that section contains Reviewer 1's scores."
)


def build_packet() -> dict:
    rng = random.Random(BLIND_ORDER_SEED)
    shuffled = list(CASES)
    rng.shuffle(shuffled)

    return {
        "reviewer_id": "reviewer_2",
        "dataset_version": DATASET_VERSION,
        "instructions": [
            "Do not open docs/HUMAN_EVAL.md or docs/human_eval/reviewer2_form.md before finishing -- "
            "both contain Reviewer 1's scores.",
            "Score every case below independently, using only the query/system_output/evidence given "
            "here plus the rubric in docs/HUMAN_EVAL.md's Rubric section (anchors only, not scores).",
            "Fill each dimension with an integer 1-5, or null where the case involves no retrieval "
            "(Groundedness/Citation Quality on conversational cases).",
            "Save this file (or a copy) with your ratings filled in, then run: "
            "cd backend && python eval/module10/runners/run_human_eval_final.py "
            "--reviewer2-file <path to your filled-in file>",
        ],
        "rubric_summary": RUBRIC_TEXT,
        "rubric_dimensions": RUBRIC_DIMENSIONS,
        "case_order_note": "Case order below is deterministically shuffled (not case_001..case_024 in "
        "sequence) to reduce order/anchoring bias relative to Reviewer 1's own presentation order.",
        "cases": [
            {
                "case_id": c.case_id,
                "case_type": c.case_type,
                "query": c.query,
                "system_output": c.system_output,
                "evidence": c.evidence,
                "ratings": {dim: None for dim in RUBRIC_DIMENSIONS},
                "comment": "",
            }
            for c in shuffled
        ],
    }


def main() -> None:
    packet = build_packet()
    out_dir = Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"reviewer_2_packet_{config.utc_timestamp()}.json"
    path.write_text(json.dumps(packet, indent=2), encoding="utf-8")
    print(f"Saved blinded reviewer-2 packet: {path}")
    print(f"Cases: {len(packet['cases'])}  Reviewer-1 scores included: No")


if __name__ == "__main__":
    main()
