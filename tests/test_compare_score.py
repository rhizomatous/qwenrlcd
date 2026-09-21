from __future__ import annotations

import json
from pathlib import Path

import pytest

from qwenrlcd.compare_score import compare_score_reports


def _summary(rps: float, mae: float, far: int, bias: float) -> dict:
    return {
        "count": 10,
        "unique_target_modes": 8,
        "far_mode_miss": far,
        "model": {
            "ranked_probability_score": rps,
            "expected_score_mae": mae,
            "expected_score_bias": bias,
        },
    }


def test_compare_score_reports_computes_candidate_minus_baseline() -> None:
    baseline = {
        "overall": _summary(0.1, 0.5, 3, 0.2),
        "by_dimension": {"helpsteer2/correctness": _summary(0.2, 0.8, 4, 0.3)},
    }
    candidate = {
        "overall": _summary(0.08, 0.4, 1, 0.1),
        "by_dimension": {"helpsteer2/correctness": _summary(0.15, 0.6, 2, 0.0)},
    }

    rows = compare_score_reports(baseline, candidate)

    assert [row["name"] for row in rows] == ["overall", "helpsteer2/correctness"]
    assert rows[0]["rps_delta"] == pytest.approx(-0.02)
    assert rows[0]["expected_score_mae_delta"] == pytest.approx(-0.1)
    assert rows[0]["far_miss_rate_delta"] == pytest.approx(-0.25)


def test_compare_score_reports_rejects_different_evaluation_sets() -> None:
    baseline = {"overall": _summary(0.1, 0.5, 3, 0.2), "by_dimension": {}}
    candidate = {"overall": _summary(0.1, 0.5, 3, 0.2), "by_dimension": {}}
    candidate["overall"]["count"] = 9
    with pytest.raises(ValueError, match="count changed"):
        compare_score_reports(baseline, candidate)


def test_rps_ablation_changes_only_output_and_ordinal_weight() -> None:
    root = Path(__file__).parents[1]
    with (root / "configs/qwen3_1_7b_core_option_50k.json").open() as handle:
        baseline = json.load(handle)
    with (root / "configs/qwen3_1_7b_core_option_50k_rps.json").open() as handle:
        candidate = json.load(handle)

    assert baseline.pop("output_dir") != candidate.pop("output_dir")
    assert baseline.pop("ordinal_rps_weight") == 0.0
    assert candidate.pop("ordinal_rps_weight") == 1.0
    assert candidate == baseline
