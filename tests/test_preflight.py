from __future__ import annotations

import pytest

from qwenrlcd.data import DecisionDataset
from qwenrlcd.preflight import inspect_token_lengths
from qwenrlcd.schema import DecisionBundle

from .test_data import WhitespaceTokenizer
from .test_schema import example_dict


def test_preflight_reports_lengths_and_sources() -> None:
    value = example_dict()
    value["source"] = "fixture"
    bundle = DecisionBundle.from_dict(value)
    tokenizer = WhitespaceTokenizer()

    report = inspect_token_lengths(
        [bundle], tokenizer,
        max_length=512, max_questions=32, max_choices=255,
    )
    assert report["bundles"] == 1
    assert report["source_counts"] == {"fixture": 1}
    assert report["question_count_max"] == 3
    assert report["over_max_length"] == 0
    assert report["packed_tokens"]["max"] > 0

    too_short = inspect_token_lengths(
        [bundle], tokenizer,
        max_length=report["packed_tokens"]["max"] - 1,
        max_questions=32, max_choices=255,
    )
    assert too_short["over_max_length"] == 1
    assert too_short["over_max_length_examples"][0]["id"] == bundle.id


def test_training_can_reject_state_truncation() -> None:
    bundle = DecisionBundle.from_dict(example_dict())
    tokenizer = WhitespaceTokenizer()
    report = inspect_token_lengths(
        [bundle], tokenizer,
        max_length=512, max_questions=32, max_choices=255,
    )
    dataset = DecisionDataset(
        [bundle], tokenizer,
        max_length=report["packed_tokens"]["max"] - 1,
        max_choices=255, max_questions=32,
        shuffle=False, seed=17, require_no_state_truncation=True,
    )
    with pytest.raises(ValueError, match="state truncation is disabled"):
        dataset[0]
