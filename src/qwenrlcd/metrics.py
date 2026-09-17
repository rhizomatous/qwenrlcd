from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True, slots=True)
class CalibrationMetrics:
    accuracy: float
    brier: float
    negative_log_likelihood: float
    expected_calibration_error: float


def calibration_metrics(
    probabilities: Sequence[Sequence[float]],
    targets: Sequence[Sequence[float]],
    *,
    bins: int = 10,
) -> CalibrationMetrics:
    if len(probabilities) != len(targets) or not probabilities:
        raise ValueError("probabilities and targets must have equal, non-zero length")

    correct: list[float] = []
    confidences: list[float] = []
    brier_total = 0.0
    nll_total = 0.0

    for predicted, target in zip(probabilities, targets, strict=True):
        if len(predicted) != len(target) or not predicted:
            raise ValueError("each probability and target vector must have equal length")
        predicted_index = max(range(len(predicted)), key=predicted.__getitem__)
        target_index = max(range(len(target)), key=target.__getitem__)
        correct.append(float(predicted_index == target_index))
        confidences.append(float(predicted[predicted_index]))
        brier_total += sum((p - y) ** 2 for p, y in zip(predicted, target, strict=True))
        nll_total -= sum(
            y * math.log(max(p, 1e-12))
            for p, y in zip(predicted, target, strict=True)
        )

    ece = 0.0
    count = len(correct)
    for bin_index in range(bins):
        lower = bin_index / bins
        upper = (bin_index + 1) / bins
        members = [
            index
            for index, confidence in enumerate(confidences)
            if lower <= confidence < upper or (bin_index == bins - 1 and confidence == 1.0)
        ]
        if not members:
            continue
        bin_accuracy = sum(correct[index] for index in members) / len(members)
        bin_confidence = sum(confidences[index] for index in members) / len(members)
        ece += len(members) / count * abs(bin_accuracy - bin_confidence)

    return CalibrationMetrics(
        accuracy=sum(correct) / count,
        brier=brier_total / count,
        negative_log_likelihood=nll_total / count,
        expected_calibration_error=ece,
    )
