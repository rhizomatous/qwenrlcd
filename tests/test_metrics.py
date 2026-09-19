from __future__ import annotations

import math

import pytest

from qwenrlcd.metrics import calibration_metrics, summarize_validation


def test_perfect_predictions_have_zero_error() -> None:
    metrics = calibration_metrics([[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]])
    assert metrics.accuracy == 1.0
    assert metrics.brier == 0.0
    assert metrics.negative_log_likelihood == 0.0
    assert metrics.expected_calibration_error == 0.0


def test_confident_wrong_prediction_is_penalized() -> None:
    metrics = calibration_metrics([[0.9, 0.1]], [[0.0, 1.0]])
    assert metrics.accuracy == 0.0
    assert metrics.brier == pytest.approx(1.62)
    assert metrics.expected_calibration_error == pytest.approx(0.9)


def test_soft_target_calibration_uses_expected_outcome_not_argmax_agreement() -> None:
    metrics = calibration_metrics([[0.9, 0.1]], [[0.6, 0.4]])
    assert metrics.accuracy == 1.0
    assert metrics.expected_accuracy == pytest.approx(0.6)
    assert metrics.expected_calibration_error == pytest.approx(0.3)
    assert metrics.target_entropy == pytest.approx(-0.6 * math.log(0.6) - 0.4 * math.log(0.4))
    assert metrics.kl_divergence > 0
    assert metrics.js_divergence > 0

    matched = calibration_metrics([[0.6, 0.4]], [[0.6, 0.4]])
    assert matched.expected_calibration_error == pytest.approx(0.0)
    assert matched.kl_divergence == pytest.approx(0.0)
    assert matched.js_divergence == pytest.approx(0.0)


def test_validation_reports_source_type_and_bundle_macros() -> None:
    correct = {"type": "choice", "probabilities": [0.9, 0.1], "target": [1.0, 0.0]}
    wrong = {"type": "noul", "probabilities": [0.9, 0.1], "target": [0.0, 1.0]}
    summary = summarize_validation(
        [
            {"id": "short", "source": "source_a", "loss": 0.2, "questions": [correct]},
            {"id": "long", "source": "source_b", "loss": 1.0, "questions": [wrong] * 3},
        ]
    )
    assert summary["loss"] == pytest.approx(0.6)
    assert summary["accuracy"] == pytest.approx(0.25)
    assert summary["bundle_macro"]["accuracy"] == pytest.approx(0.5)
    assert summary["source_macro"]["accuracy"] == pytest.approx(0.5)
    assert summary["by_source"]["source_a"]["count"] == 1
    assert summary["by_source"]["source_b"]["questions"] == 3
    assert summary["by_type"]["noul"]["count"] == 3
    assert summary["by_question_count"]["3"]["count"] == 3
    assert summary["by_choice_count"]["2"]["count"] == 4


def test_source_metrics_weight_bundles_equally_within_source() -> None:
    right = {"type": "choice", "probabilities": [1.0, 0.0], "target": [1.0, 0.0]}
    wrong = {"type": "choice", "probabilities": [1.0, 0.0], "target": [0.0, 1.0]}
    summary = summarize_validation([
        {"id": "short", "source": "same", "loss": 0.0, "questions": [right]},
        {"id": "long", "source": "same", "loss": 1.0, "questions": [wrong] * 3},
    ])
    assert summary["accuracy"] == pytest.approx(0.25)
    assert summary["by_source"]["same"]["accuracy"] == pytest.approx(0.5)
    assert summary["source_macro"]["accuracy"] == pytest.approx(0.5)
