from __future__ import annotations

import pytest

from qwenrlcd.metrics import calibration_metrics


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
