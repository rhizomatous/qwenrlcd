from __future__ import annotations

import random
from pathlib import Path

import pytest

from qwenrlcd.data import (
    DecisionCollator,
    DecisionDataset,
    build_option_attention_pattern,
    read_jsonl,
    training_view_bundle,
    write_jsonl,
)
from qwenrlcd.schema import DecisionBundle, Question
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


def test_tree_attention_isolates_questions_and_options() -> None:
    pattern = build_option_attention_pattern(
        sequence_length=14,
        state_length=2,
        question_spans=((2, 4, ((4, 6), (6, 8))), (8, 10, ((10, 12), (12, 14)))),
    )

    assert pattern[3][:4] == (True, True, True, True)
    assert pattern[5][:6] == (True,) * 6
    assert not any(pattern[5][6:])
    assert pattern[7][:4] == (True,) * 4
    assert not any(pattern[7][4:6])
    assert pattern[7][6:8] == (True, True)
    assert pattern[9][:2] == (True, True)
    assert not any(pattern[9][2:8])
    assert pattern[11][:2] == (True, True)
    assert not any(pattern[11][2:8])
    assert pattern[11][8:12] == (True,) * 4


def test_dataset_resets_logical_positions_for_each_question_and_option() -> None:
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
    for question_start, question_end, option_spans in example["option_tree_spans"]:
        assert example["position_ids"][question_start] == state_length
        option_position = state_length + question_end - question_start
        for start, end in option_spans:
            assert example["position_ids"][start] == option_position
            assert example["position_ids"][end - 1] == option_position + end - start - 1
    assert example["decision_indices"] == [
        [end - 1 for _, end in option_spans]
        for _, _, option_spans in example["option_tree_spans"]
    ]


def test_dataset_keeps_only_source_and_question_type_for_slices() -> None:
    value = example_dict()
    value["source"] = "fixture_source"
    bundle = DecisionBundle.from_dict(value)
    dataset = DecisionDataset(
        [bundle], WhitespaceTokenizer(),
        max_length=512, max_choices=255, max_questions=16,
        shuffle=True, seed=7,
    )
    example = dataset[0]
    assert example["source"] == "fixture_source"
    assert set(example["question_types"]) == {"choice", "noul", "score"}
    assert bundle.permuted(random.Random(7)).source == "fixture_source"


def test_training_view_recreates_dataset_permutation_and_packed_inputs() -> None:
    first = DecisionBundle.from_dict(example_dict())
    second_value = example_dict()
    second_value["id"] = "second"
    second = DecisionBundle.from_dict(second_value)
    bundles = [first, second]
    tokenizer = WhitespaceTokenizer()
    training = DecisionDataset(
        bundles, tokenizer, max_length=512, max_choices=255, max_questions=16,
        shuffle=True, seed=17,
    )
    training.set_epoch(3)
    for index in range(len(bundles)):
        exact_view = training_view_bundle(bundles, index, epoch=3, seed=17)
        unshuffled = DecisionDataset(
            [exact_view], tokenizer, max_length=512, max_choices=255,
            max_questions=16, shuffle=False, seed=17,
        )
        assert training[index] == unshuffled[0]


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
        bundled_start, bundled_end, bundled_options = bundled["option_tree_spans"][question_index]
        single_start, single_end, single_options = singleton["option_tree_spans"][0]

        assert bundled["input_ids"][: bundled["state_length"]] == singleton["input_ids"][
            : singleton["state_length"]
        ]
        for (left_start, left_end), (right_start, right_end) in zip(
            [(bundled_start, bundled_end), *bundled_options],
            [(single_start, single_end), *single_options],
            strict=True,
        ):
            assert bundled["input_ids"][left_start:left_end] == singleton["input_ids"][
                right_start:right_end
            ]
            assert bundled["position_ids"][left_start:left_end] == singleton[
                "position_ids"
            ][right_start:right_end]


def test_option_reordering_preserves_keyed_visible_tokens_and_positions() -> None:
    tokenizer = WhitespaceTokenizer()
    source = DecisionBundle.from_dict(example_dict())
    question = source.questions[0]
    reversed_question = Question(
        question.id, question.type, question.instructions,
        tuple(reversed(question.options)), question.target,
    )
    bundles = [
        DecisionBundle("original", source.state, (question,)),
        DecisionBundle("reversed", source.state, (reversed_question,)),
    ]
    dataset = DecisionDataset(
        bundles, tokenizer, max_length=512, max_choices=255,
        max_questions=32, shuffle=False, seed=17,
    )

    def visible_by_key(bundle: DecisionBundle, packed: dict) -> dict:
        pattern = build_option_attention_pattern(
            len(packed["input_ids"]), packed["state_length"], packed["option_tree_spans"]
        )
        result = {}
        for option, marker in zip(
            bundle.questions[0].options, packed["decision_indices"][0], strict=True
        ):
            visible = [index for index, allowed in enumerate(pattern[marker]) if allowed]
            result[option.key] = (
                [packed["input_ids"][index] for index in visible],
                [packed["position_ids"][index] for index in visible],
            )
        return result

    assert visible_by_key(bundles[0], dataset[0]) == visible_by_key(bundles[1], dataset[1])


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


def test_collator_pads_only_to_batch_question_and_option_counts() -> None:
    torch = pytest.importorskip("torch")
    tokenizer = WhitespaceTokenizer()
    full = DecisionBundle.from_dict(example_dict())
    singleton = DecisionBundle("singleton", full.state, (full.questions[0],))
    dataset = DecisionDataset(
        [full, singleton], tokenizer, max_length=512,
        max_choices=255, max_questions=32, shuffle=False, seed=17,
    )
    batch = DecisionCollator(tokenizer, max_choices=255, max_questions=32)(
        [dataset[0], dataset[1]]
    )

    assert batch["decision_indices"].shape == (2, 3, 3)
    assert batch["targets"].shape == (2, 3, 3)
    assert batch["question_mask"].tolist() == [[True, True, True], [True, False, False]]
    assert batch["score_mask"].tolist() == [[False, False, True], [False, False, False]]
    assert batch["num_choices"].tolist() == [[2, 2, 3], [2, 0, 0]]
    assert batch["question_ids"][1] == ["route", None, None]
    torch.testing.assert_close(batch["targets"][0, 0, :2], torch.tensor([0.75, 0.25]))
