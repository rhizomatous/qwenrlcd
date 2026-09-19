from __future__ import annotations

import pytest

from qwenrlcd.diagnose_choice import summarize_choice_predictions


def _row(
    source: str, question_id: str, predicted: list[float], target: list[float]
) -> dict:
    return {
        "source": source,
        "bundle_id": f"bundle-{question_id}",
        "question_id": question_id,
        "state_excerpt": "A short state",
        "instructions": "Which option?",
        "options": [
            {"key": f"option-{index}", "description": f"Description {index}"}
            for index in range(len(target))
        ],
        "probabilities": predicted,
        "target": target,
    }


def test_choice_diagnostic_cross_tabs_baselines_and_worst_examples() -> None:
    rows = [
        _row("a", "right", [0.9, 0.1], [1.0, 0.0]),
        _row("a", "wrong", [0.9, 0.1], [0.0, 1.0]),
        _row("b", "three-way", [0.1, 0.2, 0.7], [0.0, 0.0, 1.0]),
    ]
    report = summarize_choice_predictions(rows, examples_per_source=1)

    assert report["overall"]["count"] == 3
    assert report["overall"]["model"]["accuracy"] == pytest.approx(2 / 3)
    assert report["overall"]["uniform_random_accuracy"] == pytest.approx(4 / 9)
    assert report["by_source"]["a"]["count"] == 2
    assert report["by_cardinality"]["2"]["count"] == 2
    assert report["by_source_cardinality"]["b/3"]["count"] == 1
    assert report["by_cardinality"]["2"]["uniform"]["brier"] == pytest.approx(0.5)
    assert [item["question_id"] for item in report["worst_examples_by_source"]] == [
        "wrong", "three-way"
    ]
    worst = report["worst_examples_by_source"][0]
    assert worst["predicted_key"] == "option-0"
    assert worst["target_key"] == "option-1"
    assert worst["options"][1]["target"] == 1.0


def test_choice_diagnostic_rejects_misaligned_options() -> None:
    row = _row("a", "bad", [0.5, 0.5], [0.0, 1.0])
    row["options"].pop()
    with pytest.raises(ValueError, match="must align"):
        summarize_choice_predictions([row])
