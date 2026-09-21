from __future__ import annotations

import pytest

from qwenrlcd.diagnose_score import (
    ranked_probability_score,
    select_source_bundles,
    summarize_score_predictions,
)
from qwenrlcd.schema import DecisionBundle

from .test_schema import example_dict


def _row(source: str, question_id: str, predicted: list[float], target: list[float]) -> dict:
    return {
        "source": source,
        "bundle_id": f"bundle-{question_id}",
        "question_id": question_id,
        "state_excerpt": "short state",
        "instructions": "How good?",
        "options": [{"key": str(index)} for index in range(len(target))],
        "probabilities": predicted,
        "target": target,
    }


def test_ranked_probability_score_respects_ordinal_distance() -> None:
    assert ranked_probability_score([1, 0, 0], [1, 0, 0]) == pytest.approx(0)
    assert ranked_probability_score([0, 1, 0], [1, 0, 0]) == pytest.approx(0.5)
    assert ranked_probability_score([0, 0, 1], [1, 0, 0]) == pytest.approx(1)
    assert ranked_probability_score([0.5, 0.5], [1, 0]) == pytest.approx(0.25)
    with pytest.raises(ValueError):
        ranked_probability_score([1], [1])


def test_score_report_groups_dimensions_and_excludes_tied_mode_targets() -> None:
    rows = [
        _row("dynasent", "sentiment", [0, 1, 0], [1, 0, 0]),
        _row("helpsteer2", "helpfulness", [0, 0, 0, 0, 1], [1, 0, 0, 0, 0]),
        _row("helpsteer2", "helpfulness", [0, 1, 0, 0, 0], [0.5, 0.5, 0, 0, 0]),
    ]
    report = summarize_score_predictions(rows, examples_per_group=1)

    assert report["overall"]["count"] == 3
    assert report["overall"]["adjacent_mode_miss"] == 1
    assert report["overall"]["far_mode_miss"] == 1
    assert report["overall"]["underpredicted_mode"] == 0
    assert report["overall"]["overpredicted_mode"] == 2
    assert report["overall"]["target_mode_ties_excluded"] == 1
    assert report["by_source"]["dynasent"]["model"]["expected_score_mae"] == pytest.approx(1)
    assert report["by_source"]["dynasent"]["model"]["expected_score_bias"] == pytest.approx(1)
    assert report["by_dimension"]["helpsteer2/helpfulness"]["count"] == 2
    assert len(report["worst_examples_by_dimension"]) == 2
    worst = report["worst_examples_by_dimension"][1]
    assert worst["predicted_expected_score"] == pytest.approx(4)
    assert worst["target_expected_score"] == pytest.approx(0)
    assert worst["expected_score_error"] == pytest.approx(4)
    assert worst["predicted_mode"] == 4
    assert worst["target_modes"] == [0]


def test_score_report_rejects_misaligned_or_unordered_levels() -> None:
    row = _row("a", "q", [0.5, 0.5], [1, 0])
    row["options"][1]["key"] = "3"
    with pytest.raises(ValueError, match="ordered"):
        summarize_score_predictions([row])
    row["options"].pop()
    with pytest.raises(ValueError, match="align"):
        summarize_score_predictions([row])


def test_select_source_bundles_preserves_only_requested_source() -> None:
    first_value = example_dict()
    first_value["source"] = "first"
    second_value = example_dict()
    second_value["id"] = "second"
    second_value["source"] = "second"
    bundles = [DecisionBundle.from_dict(first_value), DecisionBundle.from_dict(second_value)]

    assert select_source_bundles(bundles, "second") == [bundles[1]]
    with pytest.raises(ValueError, match="no bundles"):
        select_source_bundles(bundles, "missing")
