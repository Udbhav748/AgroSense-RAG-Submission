"""Module 10 gap-closure (P8): tests for the human-evaluation second-
reviewer/IAA infrastructure -- schema validation, the blinding packet,
and weighted Cohen's kappa against hand-derived fixtures.

Does NOT test against fabricated "expected IAA" constants -- the kappa
fixtures below are independently computed by hand (see the test
docstrings) using the standard quadratic-weighted kappa formula, not
copied from this implementation's own output.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.module10.human_eval.cases import CASES, CASES_BY_ID  # noqa: E402
from eval.module10.human_eval.generate_reviewer2_packet import build_packet  # noqa: E402
from eval.module10.human_eval.schema import ReviewerDataError, validate_reviewer_file  # noqa: E402
from eval.module10.metrics.human import (  # noqa: E402
    RUBRIC_DIMENSIONS,
    ReviewerScore,
    kappa_by_dimension,
    weighted_cohens_kappa,
)


def _valid_reviewer_data(reviewer_id: str = "reviewer_x") -> dict:
    return {
        "reviewer_id": reviewer_id,
        "ratings": [
            {
                "case_id": c.case_id,
                "ratings": dict.fromkeys(RUBRIC_DIMENSIONS, 3),
                "comment": "",
            }
            for c in CASES
        ],
    }


class TestSchemaValidation:
    def test_valid_data_passes(self):
        rows = validate_reviewer_file(_valid_reviewer_data())
        assert len(rows) == len(CASES)

    def test_missing_reviewer_id_fails(self):
        data = _valid_reviewer_data()
        del data["reviewer_id"]
        with pytest.raises(ReviewerDataError, match="reviewer_id"):
            validate_reviewer_file(data)

    def test_missing_dimension_fails(self):
        data = _valid_reviewer_data()
        del data["ratings"][0]["ratings"]["safety"]
        with pytest.raises(ReviewerDataError, match="missing required dimension"):
            validate_reviewer_file(data)

    def test_invalid_score_out_of_range_fails(self):
        data = _valid_reviewer_data()
        data["ratings"][0]["ratings"]["correctness"] = 6
        with pytest.raises(ReviewerDataError, match="between 1 and 5"):
            validate_reviewer_file(data)

    def test_invalid_score_type_fails(self):
        data = _valid_reviewer_data()
        data["ratings"][0]["ratings"]["correctness"] = "five"
        with pytest.raises(ReviewerDataError, match="must be an integer"):
            validate_reviewer_file(data)

    def test_null_score_is_valid_na(self):
        data = _valid_reviewer_data()
        data["ratings"][0]["ratings"]["groundedness"] = None
        rows = validate_reviewer_file(data)  # must not raise
        assert rows[0]["ratings"]["groundedness"] is None

    def test_duplicate_case_id_fails(self):
        data = _valid_reviewer_data()
        data["ratings"].append(data["ratings"][0])
        with pytest.raises(ReviewerDataError, match="duplicate case_id"):
            validate_reviewer_file(data)

    def test_invalid_case_id_fails(self):
        data = _valid_reviewer_data()
        data["ratings"][0]["case_id"] = "case_999"
        with pytest.raises(ReviewerDataError, match="not a valid case_id"):
            validate_reviewer_file(data)

    def test_missing_case_fails(self):
        data = _valid_reviewer_data()
        data["ratings"].pop()
        with pytest.raises(ReviewerDataError, match="missing ratings for"):
            validate_reviewer_file(data)

    def test_malformed_missing_ratings_key_fails(self):
        with pytest.raises(ReviewerDataError, match="ratings"):
            validate_reviewer_file({"reviewer_id": "x"})

    def test_malformed_row_missing_case_id_fails(self):
        data = _valid_reviewer_data()
        del data["ratings"][0]["case_id"]
        with pytest.raises(ReviewerDataError, match="case_id"):
            validate_reviewer_file(data)


class TestRealReviewer1DataIsValid:
    """The actual, real reviewer-1 file (transcribed from
    docs/HUMAN_EVAL.md) must itself pass the same validation the
    schema module enforces on any reviewer file."""

    def test_reviewer_1_ratings_file_validates(self):
        path = Path(__file__).resolve().parents[1] / "eval" / "module10" / "human_eval" / "reviewer_1_ratings.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = validate_reviewer_file(data)
        assert len(rows) == 24


class TestBlindingPacket:
    def test_packet_contains_all_24_cases(self):
        packet = build_packet()
        assert len(packet["cases"]) == 24
        assert {c["case_id"] for c in packet["cases"]} == set(CASES_BY_ID)

    def test_packet_ratings_are_all_blank(self):
        packet = build_packet()
        for case in packet["cases"]:
            assert all(v is None for v in case["ratings"].values())

    def test_packet_never_contains_reviewer_1_data(self):
        """Structural blinding check: the packet-building function never
        reads reviewer_1_ratings.json's file content at all (comments
        that merely mention "reviewer 1" in prose are fine -- what
        matters is that no code path opens that file)."""
        import inspect

        from eval.module10.human_eval import generate_reviewer2_packet as mod

        source = inspect.getsource(mod.build_packet)
        assert "reviewer_1_ratings" not in source
        assert "open(" not in source  # build_packet reads nothing from disk at all

        packet = build_packet()
        assert "reviewer_1" not in json.dumps(packet).lower()

    def test_packet_order_is_not_sequential_case_id_order(self):
        """Blinded order differs from the canonical case_001..case_024
        sequence Reviewer 1's own docs/HUMAN_EVAL.md table presents."""
        packet = build_packet()
        ids = [c["case_id"] for c in packet["cases"]]
        assert ids != sorted(ids)

    def test_packet_generation_is_deterministic(self):
        """Re-running the generator produces the identical blinded order
        (reproducible), not a fresh random shuffle each time."""
        packet_a = build_packet()
        packet_b = build_packet()
        assert [c["case_id"] for c in packet_a["cases"]] == [c["case_id"] for c in packet_b["cases"]]

    def test_packet_includes_system_output_and_evidence_self_contained(self):
        packet = build_packet()
        for case in packet["cases"]:
            assert case["system_output"]
            assert "evidence" in case


class TestWeightedCohensKappa:
    """Fixtures independently hand-derived (see comments) using the
    standard formula: kappa = 1 - (weighted observed disagreement /
    weighted expected disagreement under independence), quadratic
    weights w[i][j] = (i-j)^2 / (k-1)^2.
    """

    def test_perfect_agreement_gives_kappa_1(self):
        # A=[1,2,3], B=[1,2,3] -- identical ratings, 3 categories.
        # Hand-derived: observed_disagreement=0 => kappa=1.0 exactly.
        assert weighted_cohens_kappa([1, 2, 3], [1, 2, 3], categories=range(1, 4)) == 1.0

    def test_chance_level_agreement_gives_kappa_0(self):
        # A=[1,1,0,0], B=[1,0,0,1], categories {0,1} -- balanced 2x2
        # confusion matrix (1 each off-diagonal cell, 1 each diagonal
        # cell). Hand-derived via standard unweighted-kappa cross-check
        # (quadratic weights reduce to 0/1 for k=2): po=0.5, pe=0.5,
        # kappa=(po-pe)/(1-pe)=0.0 exactly.
        assert weighted_cohens_kappa([1, 1, 0, 0], [1, 0, 0, 1], categories=range(0, 2)) == 0.0

    def test_intermediate_disagreement_matches_hand_calculation(self):
        # A=[1,2,3,1], B=[1,3,2,1], categories {1,2,3}.
        # Hand-derived (quadratic weights, denom=(3-1)^2=4):
        # observed_disagreement = 0.125, expected_disagreement = 0.34375
        # kappa = 1 - (0.125/0.34375) = 0.636363... -> rounds to 0.6364
        result = weighted_cohens_kappa([1, 2, 3, 1], [1, 3, 2, 1], categories=range(1, 4))
        assert result == pytest.approx(0.6364, abs=0.0001)

    def test_fewer_than_two_ratings_returns_none(self):
        assert weighted_cohens_kappa([3], [3]) is None
        assert weighted_cohens_kappa([], []) is None

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            weighted_cohens_kappa([1, 2], [1])


class TestKappaByDimension:
    def test_one_reviewer_returns_not_available(self):
        scores = [ReviewerScore(case_id="case_001", reviewer_id="r1", scores={"correctness": 5})]
        result = kappa_by_dimension(scores)
        assert result["kappa"] == "not available"

    def test_two_reviewers_same_case_computes_kappa(self):
        scores = [
            ReviewerScore(case_id="case_001", reviewer_id="r1", scores={d: 5 for d in RUBRIC_DIMENSIONS}),
            ReviewerScore(case_id="case_001", reviewer_id="r2", scores={d: 5 for d in RUBRIC_DIMENSIONS}),
            ReviewerScore(case_id="case_002", reviewer_id="r1", scores={d: 1 for d in RUBRIC_DIMENSIONS}),
            ReviewerScore(case_id="case_002", reviewer_id="r2", scores={d: 1 for d in RUBRIC_DIMENSIONS}),
        ]
        result = kappa_by_dimension(scores)
        assert result["kappa"] == "computed"
        assert result["per_dimension_kappa"]["correctness"]["kappa"] == 1.0

    def test_na_entries_excluded_from_pairing_not_imputed(self):
        scores = [
            ReviewerScore(case_id="case_001", reviewer_id="r1", scores={"groundedness": None, "safety": 5}),
            ReviewerScore(case_id="case_001", reviewer_id="r2", scores={"groundedness": None, "safety": 5}),
        ]
        result = kappa_by_dimension(scores)
        assert result["per_dimension_kappa"]["groundedness"]["n_paired_ratings"] == 0
        assert result["per_dimension_kappa"]["groundedness"]["kappa"] is None


class TestCaseAlignment:
    def test_reviewer_1_and_2_share_identical_case_id_set(self):
        """Exact case alignment: whatever reviewer 2 eventually submits
        must cover exactly the same 24 case_ids reviewer 1 used -- no
        silent subset/superset mismatch."""
        r1_ids = {c.case_id for c in CASES}
        packet_ids = {c["case_id"] for c in build_packet()["cases"]}
        assert r1_ids == packet_ids


class TestDeterministicRunnerOutput:
    def test_run_human_eval_final_reports_second_reviewer_required_without_one(self, capsys, monkeypatch, tmp_path):
        from eval.module10.runners import run_human_eval_final as runner_mod

        # Never writes a real, timestamped artifact into the tracked
        # reports/ directory as a side effect of running the test suite --
        # redirected to a throwaway path, but the runner's own logic
        # (loading, validating, printing) still runs for real.
        monkeypatch.setattr(runner_mod.config, "save_report", lambda report, name: tmp_path / f"{name}.json")
        monkeypatch.setattr(sys, "argv", ["run_human_eval_final.py"])

        runner_mod.main()

        captured = capsys.readouterr()
        assert "SECOND REVIEWER DATA REQUIRED" in captured.out
        assert "not available (one reviewer)" in captured.out

    def test_deterministic_case_count_and_reviewer_count(self, capsys, monkeypatch, tmp_path):
        from eval.module10.runners import run_human_eval_final as runner_mod

        monkeypatch.setattr(runner_mod.config, "save_report", lambda report, name: tmp_path / f"{name}.json")
        monkeypatch.setattr(sys, "argv", ["run_human_eval_final.py"])

        runner_mod.main()
        first = capsys.readouterr().out

        runner_mod.main()
        second = capsys.readouterr().out

        assert "Case count: 24  Reviewer count: 1" in first
        assert "Case count: 24  Reviewer count: 1" in second
