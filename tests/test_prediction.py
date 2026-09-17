from __future__ import annotations

import pytest

from qwenrlcd.prediction import format_bundle_prediction
from qwenrlcd.schema import DecisionBundle

from .test_schema import example_dict


def test_formats_mixed_typed_predictions() -> None:
    bundle = DecisionBundle.from_dict(example_dict())
    result = format_bundle_prediction(
        bundle,
        (
            (0.8, 0.2),
            (0.25, 0.75),
            (0.1, 0.6, 0.3),
        ),
    )

    assert result["id"] == "example"
    assert result["questions"]["route"] == {
        "type": "choice",
        "choice": "returns",
        "confidence": 0.8,
        "probabilities": {"returns": 0.8, "shipping": 0.2},
    }
    assert result["questions"]["urgent"] == {
        "type": "noul",
        "noul": 0.75,
        "probabilities": {"false": 0.25, "true": 0.75},
    }
    assert result["questions"]["severity"]["score"] == pytest.approx(1.2)
    assert result["questions"]["severity"]["confidence"] == 0.6
    assert result["questions"]["severity"]["legend"] == {
        "0": "low",
        "1": "medium",
        "2": "high",
    }


def test_rejects_wrong_probability_count() -> None:
    bundle = DecisionBundle.from_dict(example_dict())
    with pytest.raises(ValueError, match="expected 3 probability rows"):
        format_bundle_prediction(bundle, ((0.5, 0.5),))
