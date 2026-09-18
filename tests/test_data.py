from __future__ import annotations

from pathlib import Path

import pytest

from qwenrlcd.data import DecisionDataset, build_tree_attention_pattern, read_jsonl, write_jsonl
from qwenrlcd.schema import DecisionBundle
from qwenrlcd.synthetic import generate_bundles

from .test_schema import example_dict


class WhitespaceTokenizer:
    pad_token_id = 0

    def __init__(self) -> None:
        self.vocabulary: dict[str, int] = {"<pad>": self.pad_token_id}

    def __call__(self, text: str, *, add_special_tokens: bool) -> dict[str, list[int]]:
        assert add_special_tokens is False
        ids = []
        for token in text.split():
            if token not in self.vocabulary:
                self.vocabulary[token] = len(self.vocabulary)
            ids.append(self.vocabulary[token])
        return {"input_ids": ids}


def test_jsonl_round_trip(tmp_path) -> None:
    bundles = generate_bundles(6, seed=9)
    path = tmp_path / "bundles.jsonl"
    write_jsonl(bundles, path)
    assert read_jsonl(path) == bundles


def test_synthetic_bundle_contains_all_question_types() -> None:
    bundle = generate_bundles(1, seed=3)[0]
    assert {question.type.value for question in bundle.questions} == {
        "choice",
        "noul",
        "score",
    }


def test_overfit_fixture_has_two_hard_labeled_bundles() -> None:
    path = Path(__file__).parents[1] / "data" / "overfit.jsonl"
    bundles = read_jsonl(path)

    assert len(bundles) == 2
    assert all(
        question.target is not None
        and max(question.target.values()) == 1.0
        and sum(question.target.values()) == 1.0
        for bundle in bundles
        for question in bundle.questions
    )


def test_tree_attention_isolates_branches() -> None:
    pattern = build_tree_attention_pattern(
        sequence_length=8,
        state_length=3,
        branch_spans=((3, 5), (5, 8)),
    )

    assert pattern[2][:3] == (True, True, True)
    assert pattern[4][:5] == (True, True, True, True, True)
    assert not any(pattern[4][5:])
    assert pattern[7][:3] == (True, True, True)
    assert not any(pattern[7][3:5])
    assert pattern[7][5:8] == (True, True, True)


def test_dataset_resets_logical_positions_for_each_question() -> None:
    tokenizer = WhitespaceTokenizer()
    bundle = DecisionBundle.from_dict(example_dict())
    dataset = DecisionDataset(
        [bundle],
        tokenizer,
        max_length=512,
        max_choices=255,
        max_questions=16,
        shuffle=False,
        seed=1,
    )

    example = dataset[0]
    state_length = example["state_length"]
    for start, end in example["branch_spans"]:
        assert example["position_ids"][start] == state_length
        assert example["position_ids"][end - 1] == state_length + (end - start) - 1
    assert example["decision_indices"] == [end - 1 for _, end in example["branch_spans"]]


def test_bundle_and_singleton_have_identical_branch_inputs() -> None:
    tokenizer = WhitespaceTokenizer()
    bundle = DecisionBundle.from_dict(example_dict())

    def encode(value: DecisionBundle) -> dict:
        return DecisionDataset(
            [value],
            tokenizer,
            max_length=512,
            max_choices=255,
            max_questions=16,
            shuffle=False,
            seed=1,
        )[0]

    bundled = encode(bundle)
    for question_index, question in enumerate(bundle.questions):
        singleton = encode(DecisionBundle(f"single-{question.id}", bundle.state, (question,)))
        bundled_start, bundled_end = bundled["branch_spans"][question_index]
        single_start, single_end = singleton["branch_spans"][0]

        assert bundled["input_ids"][: bundled["state_length"]] == singleton["input_ids"][
            : singleton["state_length"]
        ]
        assert bundled["input_ids"][bundled_start:bundled_end] == singleton["input_ids"][
            single_start:single_end
        ]
        assert bundled["position_ids"][bundled_start:bundled_end] == singleton[
            "position_ids"
        ][single_start:single_end]


def test_dataset_requires_targets_only_in_training_mode() -> None:
    value = example_dict()
    for question in value["questions"].values():
        question.pop("target")
    bundle = DecisionBundle.from_dict(value)
    tokenizer = WhitespaceTokenizer()

    training_dataset = DecisionDataset(
        [bundle],
        tokenizer,
        max_length=512,
        max_choices=255,
        max_questions=16,
        shuffle=False,
        seed=1,
    )
    with pytest.raises(ValueError, match="has no training target"):
        training_dataset[0]

    inference_dataset = DecisionDataset(
        [bundle],
        tokenizer,
        max_length=512,
        max_choices=255,
        max_questions=16,
        shuffle=False,
        seed=1,
        require_targets=False,
    )
    assert inference_dataset[0]["targets"] == [
        [0.0] * len(question.options) for question in bundle.questions
    ]
