from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields
from typing import Any


@dataclass(frozen=True, slots=True)
class CalibrationMetrics:
    accuracy: float
    brier: float
    negative_log_likelihood: float
    expected_calibration_error: float
    expected_accuracy: float
    target_entropy: float
    kl_divergence: float
    js_divergence: float


def calibration_metrics(
    probabilities: Sequence[Sequence[float]],
    targets: Sequence[Sequence[float]],
    *,
    bins: int = 10,
) -> CalibrationMetrics:
    if len(probabilities) != len(targets) or not probabilities:
        raise ValueError("probabilities and targets must have equal, non-zero length")
    if bins < 1:
        raise ValueError("bins must be positive")

    correct: list[float] = []
    expected_correct: list[float] = []
    confidences: list[float] = []
    brier_total = 0.0
    nll_total = 0.0
    entropy_total = 0.0
    js_total = 0.0

    for predicted, target in zip(probabilities, targets, strict=True):
        if len(predicted) != len(target) or not predicted:
            raise ValueError("each probability and target vector must have equal length")
        if any(not math.isfinite(value) or value < 0 for value in (*predicted, *target)):
            raise ValueError("probabilities and targets must be finite and non-negative")
        if not math.isclose(sum(predicted), 1.0, abs_tol=1e-4) or not math.isclose(
            sum(target), 1.0, abs_tol=1e-4
        ):
            raise ValueError("probabilities and targets must sum to one")
        predicted_index = max(range(len(predicted)), key=predicted.__getitem__)
        target_index = max(range(len(target)), key=target.__getitem__)
        correct.append(float(predicted_index == target_index))
        expected_correct.append(float(target[predicted_index]))
        confidences.append(float(predicted[predicted_index]))
        brier_total += sum((p - y) ** 2 for p, y in zip(predicted, target, strict=True))
        nll_total -= sum(
            y * math.log(max(p, 1e-12))
            for p, y in zip(predicted, target, strict=True)
        )
        entropy_total -= sum(y * math.log(y) for y in target if y > 0)
        midpoint = [(p + y) / 2 for p, y in zip(predicted, target, strict=True)]
        js_total += 0.5 * sum(
            p * math.log(p / m)
            for p, m in zip(predicted, midpoint, strict=True)
            if p > 0
        ) + 0.5 * sum(
            y * math.log(y / m)
            for y, m in zip(target, midpoint, strict=True)
            if y > 0
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
        bin_accuracy = sum(expected_correct[index] for index in members) / len(members)
        bin_confidence = sum(confidences[index] for index in members) / len(members)
        ece += len(members) / count * abs(bin_accuracy - bin_confidence)

    return CalibrationMetrics(
        accuracy=sum(correct) / count,
        brier=brier_total / count,
        negative_log_likelihood=nll_total / count,
        expected_calibration_error=ece,
        expected_accuracy=sum(expected_correct) / count,
        target_entropy=entropy_total / count,
        kl_divergence=max(0.0, (nll_total - entropy_total) / count),
        js_divergence=js_total / count,
    )


def _metric_dict(questions: Sequence[dict[str, Any]]) -> dict[str, float | int]:
    metrics = calibration_metrics(
        [question["probabilities"] for question in questions],
        [question["target"] for question in questions],
    )
    uniform = _uniform_metrics(questions)
    return {
        "count": len(questions), **asdict(metrics),
        "uniform_brier": uniform.brier,
        "uniform_kl_divergence": uniform.kl_divergence,
    }


def _uniform_metrics(questions: Sequence[dict[str, Any]]) -> CalibrationMetrics:
    targets = [question["target"] for question in questions]
    return calibration_metrics(
        [[1.0 / len(target)] * len(target) for target in targets], targets
    )


def _macro_metrics(groups: Sequence[Sequence[dict[str, Any]]]) -> dict[str, float | int]:
    summaries = [calibration_metrics(
        [question["probabilities"] for question in group],
        [question["target"] for question in group],
    ) for group in groups]
    uniform = [_uniform_metrics(group) for group in groups]
    return {
        "count": len(summaries),
        **{
            field.name: sum(getattr(summary, field.name) for summary in summaries)
            / len(summaries)
            for field in fields(CalibrationMetrics)
        },
        "uniform_brier": sum(value.brier for value in uniform) / len(uniform),
        "uniform_kl_divergence": sum(value.kl_divergence for value in uniform)
        / len(uniform),
    }


def summarize_validation(bundles: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Summarize valid decisions without letting long bundles hide short ones."""
    if not bundles:
        raise ValueError("validation contains no bundles")
    by_source: dict[str, list[dict[str, Any]]] = {}
    source_bundles: dict[str, list[Sequence[dict[str, Any]]]] = {}
    by_type: dict[str, list[dict[str, Any]]] = {}
    by_choice_count: dict[str, list[dict[str, Any]]] = {}
    by_question_count: dict[str, list[dict[str, Any]]] = {}
    by_entropy: dict[str, list[dict[str, Any]]] = {}
    questions: list[dict[str, Any]] = []
    for bundle in bundles:
        bundle_questions = bundle["questions"]
        if not bundle_questions:
            raise ValueError("validation bundle has no questions")
        question_count = len(bundle_questions)
        source = bundle["source"]
        source_bundles.setdefault(source, []).append(bundle_questions)
        for question in bundle_questions:
            choice_count = len(question["target"])
            entropy = -sum(p * math.log(p) for p in question["target"] if p > 0)
            normalized_entropy = entropy / math.log(choice_count)
            entropy_band = (
                "low" if normalized_entropy < 1 / 3 else
                "medium" if normalized_entropy < 2 / 3 else "high"
            )
            questions.append(question)
            by_source.setdefault(source, []).append(question)
            by_type.setdefault(question["type"], []).append(question)
            by_choice_count.setdefault(str(choice_count), []).append(question)
            by_question_count.setdefault(str(question_count), []).append(question)
            by_entropy.setdefault(entropy_band, []).append(question)

    overall = _metric_dict(questions)
    source_summaries = {
        source: {
            **_macro_metrics(source_bundles[source]),
            "questions": len(by_source[source]),
        }
        for source in sorted(by_source)
    }
    return {
        "loss": sum(float(bundle["loss"]) for bundle in bundles) / len(bundles),
        "bundles": len(bundles),
        **overall,
        "bundle_macro": _macro_metrics([bundle["questions"] for bundle in bundles]),
        "source_macro": {
            "count": len(source_summaries),
            **{
                field.name: sum(summary[field.name] for summary in source_summaries.values())
                / len(source_summaries)
                for field in fields(CalibrationMetrics)
            },
            "uniform_brier": sum(
                summary["uniform_brier"] for summary in source_summaries.values()
            ) / len(source_summaries),
            "uniform_kl_divergence": sum(
                summary["uniform_kl_divergence"] for summary in source_summaries.values()
            ) / len(source_summaries),
        },
        "by_source": source_summaries,
        "by_type": {key: _metric_dict(value) for key, value in sorted(by_type.items())},
        "by_choice_count": {
            key: _metric_dict(value) for key, value in sorted(by_choice_count.items())
        },
        "by_question_count": {
            key: _metric_dict(value) for key, value in sorted(by_question_count.items())
        },
        "by_target_entropy": {
            key: _metric_dict(value) for key, value in sorted(by_entropy.items())
        },
    }
