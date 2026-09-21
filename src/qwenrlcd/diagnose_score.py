from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict
from typing import Any

from .metrics import calibration_metrics


def ranked_probability_score(prediction: Sequence[float], target: Sequence[float]) -> float:
    """Mean squared cumulative-distribution error over the K-1 ordinal boundaries."""
    if len(prediction) != len(target) or len(target) < 2:
        raise ValueError("ordinal distributions need the same number of at least two levels")
    predicted_cdf = target_cdf = total = 0.0
    for predicted, expected in zip(prediction[:-1], target[:-1], strict=True):
        predicted_cdf += predicted
        target_cdf += expected
        total += (predicted_cdf - target_cdf) ** 2
    return total / (len(target) - 1)


def _score_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("score diagnostic needs at least one prediction")
    predictions = [row["probabilities"] for row in rows]
    targets = [row["target"] for row in rows]
    uniform = [[1.0 / len(target)] * len(target) for target in targets]

    def ordinal_metrics(candidate: list[list[float]]) -> dict[str, float]:
        return {
            "expected_score_mae": sum(
                abs(
                    sum(index * value for index, value in enumerate(prediction))
                    - sum(index * value for index, value in enumerate(target))
                )
                for prediction, target in zip(candidate, targets, strict=True)
            ) / len(rows),
            "normalized_expected_score_mae": sum(
                abs(
                    sum(index * value for index, value in enumerate(prediction))
                    - sum(index * value for index, value in enumerate(target))
                ) / (len(target) - 1)
                for prediction, target in zip(candidate, targets, strict=True)
            ) / len(rows),
            "ranked_probability_score": sum(
                ranked_probability_score(prediction, target)
                for prediction, target in zip(candidate, targets, strict=True)
            ) / len(rows),
        }

    distances = []
    ties = 0
    for prediction, target in zip(predictions, targets, strict=True):
        maximum = max(target)
        target_modes = [
            index for index, value in enumerate(target)
            if math.isclose(value, maximum, abs_tol=1e-8, rel_tol=0)
        ]
        if len(target_modes) != 1:
            ties += 1
            continue
        predicted_mode = max(range(len(prediction)), key=prediction.__getitem__)
        distances.append(abs(predicted_mode - target_modes[0]))

    return {
        "count": len(rows),
        "target_mode_ties_excluded": ties,
        "unique_target_modes": len(distances),
        "exact_mode": distances.count(0),
        "adjacent_mode_miss": distances.count(1),
        "far_mode_miss": sum(distance >= 2 for distance in distances),
        "mean_mode_distance": sum(distances) / len(distances) if distances else None,
        "model": {
            **asdict(calibration_metrics(predictions, targets)),
            **ordinal_metrics(predictions),
        },
        "uniform": {**asdict(calibration_metrics(uniform, targets)), **ordinal_metrics(uniform)},
    }


def summarize_score_predictions(
    rows: list[dict[str, Any]], *, examples_per_group: int = 2
) -> dict[str, Any]:
    if not rows:
        raise ValueError("score diagnostic needs at least one prediction")
    if examples_per_group < 0:
        raise ValueError("examples_per_group must be non-negative")
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_dimension: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        count = len(row["options"])
        if count != len(row["target"]) or count != len(row["probabilities"]):
            raise ValueError("score prediction, target, and options must align")
        if [option["key"] for option in row["options"]] != [str(i) for i in range(count)]:
            raise ValueError("score levels must be ordered from 0 to K-1")
        by_source[row["source"]].append(row)
        by_dimension[f"{row['source']}/{row['question_id']}"].append(row)

    examples = []
    for group, group_rows in sorted(by_dimension.items()):
        for row in sorted(
            group_rows,
            key=lambda item: ranked_probability_score(item["probabilities"], item["target"]),
            reverse=True,
        )[:examples_per_group]:
            examples.append({
                "group": group,
                "bundle_id": row["bundle_id"],
                "state_excerpt": row["state_excerpt"],
                "instructions": row["instructions"],
                "predicted": row["probabilities"],
                "target": row["target"],
                "ranked_probability_score": ranked_probability_score(
                    row["probabilities"], row["target"]
                ),
            })
    return {
        "overall": _score_summary(rows),
        "by_source": {key: _score_summary(value) for key, value in sorted(by_source.items())},
        "by_dimension": {
            key: _score_summary(value) for key, value in sorted(by_dimension.items())
        },
        "worst_examples_by_dimension": examples,
    }
