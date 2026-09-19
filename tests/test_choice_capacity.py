from __future__ import annotations

from collections import Counter

import pytest

from qwenrlcd.choice_capacity import select_capacity_bundles, write_capacity_fixture
from qwenrlcd.data import read_jsonl
from qwenrlcd.schema import DecisionBundle, QuestionType

from .test_schema import example_dict


def choice_bundle(source: str, index: int) -> DecisionBundle:
    value = example_dict()
    value["id"] = f"{source}-{index}"
    value["source"] = source
    return DecisionBundle.from_dict(value)


def test_selection_is_balanced_stable_and_choice_only() -> None:
    bundles = [
        choice_bundle(source, index)
        for source in ("one", "two", "three", "four")
        for index in range(6)
    ]
    selected = select_capacity_bundles(bundles, per_source=3, seed=17)

    assert len(selected) == 12
    assert Counter(bundle.source for bundle in selected) == {
        "one": 3, "two": 3, "three": 3, "four": 3,
    }
    assert all(len(bundle.questions) == 1 for bundle in selected)
    assert all(bundle.questions[0].type is QuestionType.CHOICE for bundle in selected)
    assert selected == select_capacity_bundles(
        list(reversed(bundles)), per_source=3, seed=17
    )
    assert {bundle.id for bundle in selected} != {
        bundle.id for bundle in select_capacity_bundles(bundles, per_source=3, seed=18)
    }


def test_selection_requires_four_sources_and_enough_rows() -> None:
    bundles = [choice_bundle(source, 0) for source in ("one", "two", "three")]
    with pytest.raises(ValueError, match="expected 4 Choice sources"):
        select_capacity_bundles(bundles, per_source=1, seed=17)
    bundles.append(choice_bundle("four", 0))
    with pytest.raises(ValueError, match="only 1 Choice bundles"):
        select_capacity_bundles(bundles, per_source=2, seed=17)
    with pytest.raises(ValueError, match="per_source must be positive"):
        select_capacity_bundles(bundles, per_source=0, seed=17)


def test_fixture_reuses_identical_content_but_refuses_changes(tmp_path) -> None:
    bundles = [choice_bundle("one", 0)]
    path = tmp_path / "nested" / "capacity.jsonl"
    assert write_capacity_fixture(bundles, path)
    assert read_jsonl(path) == bundles
    assert not write_capacity_fixture(bundles, path)
    with pytest.raises(ValueError, match="existing capacity fixture differs"):
        write_capacity_fixture([choice_bundle("one", 1)], path)
    assert read_jsonl(path) == bundles
