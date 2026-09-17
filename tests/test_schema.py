from __future__ import annotations

import random

import pytest

from qwenrlcd.schema import DecisionBundle, QuestionType


def example_dict() -> dict:
    return {
        "id": "example",
        "state": {"message": "A damaged parcel", "delay_days": 3},
        "questions": {
            "route": {
                "type": "choice",
                "instructions": "Where should it go?",
                "criteria": {
                    "returns": "Damaged items",
                    "shipping": "Missing items",
                },
                "target": {"returns": 0.75, "shipping": 0.25},
            },
            "urgent": {
                "type": "noul",
                "instructions": "Is this urgent?",
                "target": {"false": 0.2, "true": 0.8},
            },
            "severity": {
                "type": "score",
                "instructions": "How severe is it?",
                "criteria": ["low", "medium", "high"],
                "target": {"1": 0.6, "2": 0.4},
            },
        },
    }


def test_bundle_parses_mixed_questions_and_round_trips() -> None:
    bundle = DecisionBundle.from_dict(example_dict())

    assert [question.type for question in bundle.questions] == [
        QuestionType.CHOICE,
        QuestionType.NOUL,
        QuestionType.SCORE,
    ]
    assert bundle.questions[0].target_vector() == [0.75, 0.25]
    assert bundle.questions[2].expected_score() == pytest.approx(1.4)
    assert DecisionBundle.from_dict(bundle.to_dict()) == bundle


def test_permutation_preserves_targets_and_ordered_types() -> None:
    bundle = DecisionBundle.from_dict(example_dict())
    permuted = bundle.permuted(random.Random(7))

    assert {question.id for question in permuted.questions} == {
        question.id for question in bundle.questions
    }
    by_id = {question.id: question for question in permuted.questions}
    assert [option.key for option in by_id["urgent"].options] == ["false", "true"]
    assert [option.key for option in by_id["severity"].options] == ["0", "1", "2"]
    for question in permuted.questions:
        assert sum(question.target_vector()) == pytest.approx(1.0)


def test_target_must_sum_to_one() -> None:
    value = example_dict()
    value["questions"]["route"]["target"] = {"returns": 0.2}
    with pytest.raises(ValueError, match="sum to one"):
        DecisionBundle.from_dict(value)


def test_target_cannot_name_unknown_option() -> None:
    value = example_dict()
    value["questions"]["route"]["target"] = {"elsewhere": 1.0}
    with pytest.raises(ValueError, match="unknown options"):
        DecisionBundle.from_dict(value)
