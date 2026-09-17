from __future__ import annotations

import random

import pytest

from qwenrlcd.schema import DecisionRecord


def example_dict() -> dict:
    return {
        "id": "example",
        "state": "A damaged parcel",
        "question": {
            "type": "choice",
            "instructions": "Where should it go?",
            "options": [
                {"key": "returns", "description": "Damaged items"},
                {"key": "shipping", "description": "Missing items"},
            ],
        },
        "target": {"returns": 0.75, "shipping": 0.25},
    }


def test_soft_target_vector_follows_option_order() -> None:
    record = DecisionRecord.from_dict(example_dict())
    assert record.target_vector() == [0.75, 0.25]

    permuted = record.permuted(random.Random(4))
    expected = [record.target[option.key] for option in permuted.question.options]
    assert permuted.target_vector() == expected


def test_target_must_sum_to_one() -> None:
    value = example_dict()
    value["target"] = {"returns": 0.2}
    with pytest.raises(ValueError, match="sum to one"):
        DecisionRecord.from_dict(value)


def test_target_cannot_name_unknown_option() -> None:
    value = example_dict()
    value["target"] = {"elsewhere": 1.0}
    with pytest.raises(ValueError, match="unknown options"):
        DecisionRecord.from_dict(value)
