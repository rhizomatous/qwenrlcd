from __future__ import annotations

import json
from collections import Counter

import pytest

from qwenrlcd.hf_data import (
    HFDatasetBundles,
    bundle_from_dataset_row,
    select_question_type,
    source_stratified_indices,
)
from qwenrlcd.schema import DecisionBundle, QuestionType

from .test_schema import example_dict


def test_hf_row_uses_decisions_and_source_without_provenance_details() -> None:
    row = {
        "schema_version": "qwenrlcd.bundle.v1",
        "id": "example",
        "state_json": json.dumps({"message": "hello"}),
        "questions": [{
            "id": "sentiment",
            "type": "choice",
            "instructions_json": json.dumps("What is the sentiment?"),
            "options": [
                {"key": "negative", "description_json": json.dumps("negative")},
                {"key": "positive", "description_json": json.dumps("positive")},
            ],
            "target": [0.25, 0.75],
        }],
        "provenance": {"source": "fixture", "details_json": "not parsed"},
    }
    bundle = bundle_from_dataset_row(row)
    assert bundle.source == "fixture"
    assert bundle.state == {"message": "hello"}
    assert bundle.questions[0].target_vector() == [0.25, 0.75]
    assert HFDatasetBundles([row])[0] == bundle
    assert len(HFDatasetBundles([row])) == 1


def test_source_stratified_sample_preserves_five_percent_slice() -> None:
    ids = [f"example-{i}" for i in range(1000)]
    sources = ["main"] * 950 + ["probability"] * 50
    indices = source_stratified_indices(
        ids, sources, limit=200, seed=17, split="train"
    )
    assert len(indices) == 200
    assert len(set(indices)) == 200
    assert Counter(sources[index] for index in indices) == {
        "main": 190,
        "probability": 10,
    }
    reordered = list(reversed(list(zip(ids, sources, strict=True))))
    reversed_indices = source_stratified_indices(
        [item[0] for item in reordered],
        [item[1] for item in reordered],
        limit=200, seed=17, split="train",
    )
    assert {ids[index] for index in indices} == {
        reordered[index][0] for index in reversed_indices
    }
    assert indices != source_stratified_indices(
        ids, sources, limit=200, seed=18, split="train"
    )


def test_source_stratified_sample_requires_source_coverage() -> None:
    with pytest.raises(ValueError, match="at least one bundle per source"):
        source_stratified_indices(
            ["a", "b"], ["first", "second"], limit=1, seed=1, split="train"
        )


def test_question_type_filter_keeps_bundle_and_option_targets() -> None:
    bundle = DecisionBundle.from_dict(example_dict())
    selected = select_question_type([bundle], QuestionType.CHOICE)
    assert len(selected) == 1
    assert selected[0].id == bundle.id
    assert selected[0].state == bundle.state
    assert [question.id for question in selected[0].questions] == ["route"]
    assert selected[0].questions[0].target_vector() == [0.75, 0.25]
    with pytest.raises(ValueError, match="no score questions"):
        select_question_type(selected, QuestionType.SCORE)
