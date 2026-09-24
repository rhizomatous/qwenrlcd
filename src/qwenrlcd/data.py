from __future__ import annotations

import json
import random
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from .formatting import (
    render_option,
    render_question_prefix,
    render_state,
)
from .schema import DecisionBundle, QuestionType

TOKEN_ROLE_PADDING = 0
TOKEN_ROLE_STATE = 1
TOKEN_ROLE_QUESTION = 2
TOKEN_ROLE_OPTION = 3

DECISION_MODEL_INPUT_KEYS = (
    "input_ids",
    "position_ids",
    "tree_attention_mask",
    "token_roles",
    "token_question_indices",
    "token_option_indices",
    "decision_indices",
)


def decision_model_inputs(batch: dict[str, Any], device: Any | None = None) -> dict[str, Any]:
    """Select the topology tensors accepted by :class:`DecisionModel`."""
    inputs = {key: batch[key] for key in DECISION_MODEL_INPUT_KEYS if key in batch}
    if device is not None:
        inputs = {key: value.to(device) for key, value in inputs.items()}
    return inputs


def read_jsonl(path: str | Path) -> list[DecisionBundle]:
    bundles: list[DecisionBundle] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                bundles.append(DecisionBundle.from_dict(json.loads(line)))
            except Exception as exc:
                raise ValueError(f"invalid bundle at {path}:{line_number}: {exc}") from exc
    if not bundles:
        raise ValueError(f"no bundles found in {path}")
    return bundles


def write_jsonl(bundles: Iterable[DecisionBundle], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for bundle in bundles:
            handle.write(json.dumps(bundle.to_dict(), ensure_ascii=False) + "\n")


def training_view_bundle(
    bundles: Sequence[DecisionBundle], index: int, *, epoch: int, seed: int
) -> DecisionBundle:
    """Recreate the exact question/option permutation used for a training row."""
    rng = random.Random(seed + epoch * len(bundles) + index)
    return bundles[index].permuted(rng)


def build_option_attention_pattern(
    sequence_length: int,
    state_length: int,
    question_spans: Sequence[tuple[int, int, Sequence[tuple[int, int]]]],
) -> tuple[tuple[bool, ...], ...]:
    """State → question prefix → isolated option leaves, all in one causal pass."""
    if not 0 < state_length <= sequence_length:
        raise ValueError("state_length must be within the sequence")
    allowed = [[False] * sequence_length for _ in range(sequence_length)]
    for query in range(state_length):
        for key in range(query + 1):
            allowed[query][key] = True

    previous_end = state_length
    for question_start, question_end, options in question_spans:
        if question_start != previous_end or not question_start < question_end:
            raise ValueError("question prefixes must follow the previous branch")
        for query in range(question_start, question_end):
            for key in range(state_length):
                allowed[query][key] = True
            for key in range(question_start, query + 1):
                allowed[query][key] = True
        previous_end = question_end
        if not options:
            raise ValueError("each question requires option branches")
        for option_start, option_end in options:
            if option_start != previous_end or not option_start < option_end:
                raise ValueError("option branches must follow the question prefix")
            for query in range(option_start, option_end):
                for key in range(state_length):
                    allowed[query][key] = True
                for key in range(question_start, question_end):
                    allowed[query][key] = True
                for key in range(option_start, query + 1):
                    allowed[query][key] = True
            previous_end = option_end
    if previous_end != sequence_length:
        raise ValueError("option branches must cover the complete sequence")
    return tuple(tuple(row) for row in allowed)


class DecisionDataset(Sequence[dict[str, Any]]):
    """Pack a state, question prefixes, and isolated option leaves in one sequence."""

    def __init__(
        self,
        bundles: Sequence[DecisionBundle],
        tokenizer: Any,
        *,
        max_length: int,
        max_choices: int,
        max_questions: int,
        shuffle: bool,
        seed: int,
        require_targets: bool = True,
        require_no_state_truncation: bool = False,
    ) -> None:
        self.bundles = bundles
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_choices = max_choices
        self.max_questions = max_questions
        self.shuffle = shuffle
        self.seed = seed
        self.require_targets = require_targets
        self.require_no_state_truncation = require_no_state_truncation
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.bundles)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _encode(self, text: str) -> list[int]:
        return list(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def __getitem__(self, index: int) -> dict[str, Any]:
        bundle = self.bundles[index]
        if self.shuffle:
            bundle = training_view_bundle(
                self.bundles, index, epoch=self.epoch, seed=self.seed
            )

        if len(bundle.questions) > self.max_questions:
            raise ValueError(
                f"bundle {bundle.id} has {len(bundle.questions)} questions, "
                f"above configured maximum {self.max_questions}"
            )
        for question in bundle.questions:
            if len(question.options) > self.max_choices:
                raise ValueError(
                    f"question {question.id} has {len(question.options)} choices, "
                    f"above configured maximum {self.max_choices}"
                )
            if self.require_targets and question.target is None:
                raise ValueError(f"question {question.id} has no training target")

        state_ids = self._encode(render_state(bundle))
        question_ids = [
            self._encode(render_question_prefix(question)) for question in bundle.questions
        ]
        option_ids = [
            [self._encode(render_option(option)) for option in question.options]
            for question in bundle.questions
        ]
        if any(not prefix for prefix in question_ids) or any(
            not option for options in option_ids for option in options
        ):
            raise ValueError(f"bundle {bundle.id} contains an empty option branch")
        branch_token_count = sum(map(len, question_ids)) + sum(
            len(option) for options in option_ids for option in options
        )
        state_budget = self.max_length - branch_token_count
        if state_budget < 1:
            raise ValueError(
                f"bundle {bundle.id} question branches require {branch_token_count} tokens, "
                f"leaving no state space within max_length={self.max_length}"
            )
        if len(state_ids) > state_budget:
            if self.require_no_state_truncation:
                raise ValueError(
                    f"bundle {bundle.id} needs {len(state_ids) + branch_token_count} "
                    f"tokens, above max_length={self.max_length}; state truncation is disabled"
                )
            state_ids = state_ids[-state_budget:]

        input_ids = list(state_ids)
        position_ids = list(range(len(state_ids)))
        packed: dict[str, Any] = {
            "id": bundle.id,
            "source": bundle.source or "unspecified",
            "question_ids": [question.id for question in bundle.questions],
            "question_types": [question.type.value for question in bundle.questions],
            "state_length": len(state_ids),
            "num_choices": [len(question.options) for question in bundle.questions],
            "targets": [
                question.target_vector()
                if question.target is not None
                else [0.0] * len(question.options)
                for question in bundle.questions
            ],
        }
        option_tree_spans: list[tuple[int, int, list[tuple[int, int]]]] = []
        option_decision_indices: list[list[int]] = []
        for prefix, options in zip(question_ids, option_ids, strict=True):
            question_start = len(input_ids)
            input_ids.extend(prefix)
            question_end = len(input_ids)
            position_ids.extend(range(len(state_ids), len(state_ids) + len(prefix)))
            spans: list[tuple[int, int]] = []
            indices: list[int] = []
            for option in options:
                option_start = len(input_ids)
                input_ids.extend(option)
                option_end = len(input_ids)
                spans.append((option_start, option_end))
                indices.append(option_end - 1)
                position_ids.extend(range(
                    len(state_ids) + len(prefix),
                    len(state_ids) + len(prefix) + len(option),
                ))
            option_tree_spans.append((question_start, question_end, spans))
            option_decision_indices.append(indices)
        packed["option_tree_spans"] = option_tree_spans
        packed["decision_indices"] = option_decision_indices
        packed["input_ids"] = input_ids
        packed["position_ids"] = position_ids
        return packed


class DecisionCollator:
    def __init__(
        self,
        tokenizer: Any,
        max_choices: int,
        max_questions: int,
        *,
        compact_attention_topology: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_choices = max_choices
        self.max_questions = max_questions
        self.compact_attention_topology = compact_attention_topology

    def __call__(self, examples: Sequence[dict[str, Any]]) -> dict[str, Any]:
        import torch

        batch_size = len(examples)
        max_sequence_length = max(len(example["input_ids"]) for example in examples)
        question_slots = max(len(example["decision_indices"]) for example in examples)
        choice_slots = max(max(example["num_choices"]) for example in examples)
        if question_slots > self.max_questions or choice_slots > self.max_choices:
            raise ValueError("batch exceeds configured question or option capacity")
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            raise ValueError("tokenizer must define pad_token_id")

        input_ids = torch.full(
            (batch_size, max_sequence_length), pad_token_id, dtype=torch.long
        )
        position_ids = torch.zeros((batch_size, max_sequence_length), dtype=torch.long)
        tree_attention_mask = None
        if not self.compact_attention_topology:
            tree_attention_mask = torch.zeros(
                (batch_size, max_sequence_length, max_sequence_length), dtype=torch.bool
            )
        token_roles = torch.full(
            (batch_size, max_sequence_length), TOKEN_ROLE_PADDING, dtype=torch.int32
        )
        token_question_indices = torch.full(
            (batch_size, max_sequence_length), -1, dtype=torch.int32
        )
        token_option_indices = torch.full(
            (batch_size, max_sequence_length), -1, dtype=torch.int32
        )
        decision_indices = torch.zeros(
            (batch_size, question_slots, choice_slots), dtype=torch.long
        )
        num_choices = torch.zeros((batch_size, question_slots), dtype=torch.long)
        question_mask = torch.zeros((batch_size, question_slots), dtype=torch.bool)
        targets = torch.zeros(
            (batch_size, question_slots, choice_slots), dtype=torch.float32
        )
        score_mask = torch.zeros((batch_size, question_slots), dtype=torch.bool)
        question_ids: list[list[str | None]] = []
        question_types: list[list[str | None]] = []

        for row, example in enumerate(examples):
            sequence_length = len(example["input_ids"])
            question_count = len(example["decision_indices"])
            input_ids[row, :sequence_length] = torch.tensor(example["input_ids"])
            position_ids[row, :sequence_length] = torch.tensor(example["position_ids"])
            state_length = example["state_length"]
            token_roles[row, :state_length] = TOKEN_ROLE_STATE
            for question_index, (question_start, question_end, option_spans) in enumerate(
                example["option_tree_spans"]
            ):
                token_roles[row, question_start:question_end] = TOKEN_ROLE_QUESTION
                token_question_indices[row, question_start:question_end] = question_index
                for option_index, (option_start, option_end) in enumerate(option_spans):
                    token_roles[row, option_start:option_end] = TOKEN_ROLE_OPTION
                    token_question_indices[row, option_start:option_end] = question_index
                    token_option_indices[row, option_start:option_end] = option_index

            if tree_attention_mask is not None:
                pattern = build_option_attention_pattern(
                    sequence_length, state_length, example["option_tree_spans"]
                )
                tree_attention_mask[row, :sequence_length, :sequence_length] = torch.tensor(
                    pattern, dtype=torch.bool
                )
                for padding_index in range(sequence_length, max_sequence_length):
                    tree_attention_mask[row, padding_index, padding_index] = True

            for question_index, indices in enumerate(example["decision_indices"]):
                decision_indices[row, question_index, :len(indices)] = torch.tensor(
                    indices, dtype=torch.long
                )
            num_choices[row, :question_count] = torch.tensor(
                example["num_choices"], dtype=torch.long
            )
            question_mask[row, :question_count] = True
            for question_index, target in enumerate(example["targets"]):
                targets[row, question_index, : len(target)] = torch.tensor(target)

            ids = list(example["question_ids"])
            question_ids.append(ids + [None] * (question_slots - len(ids)))
            types = list(example["question_types"])
            question_types.append(types + [None] * (question_slots - len(types)))
            score_mask[row, :question_count] = torch.tensor(
                [question_type == QuestionType.SCORE.value for question_type in types]
            )

        batch = {
            "ids": [example["id"] for example in examples],
            "sources": [example["source"] for example in examples],
            "question_ids": question_ids,
            "question_types": question_types,
            "input_ids": input_ids,
            "position_ids": position_ids,
            "token_roles": token_roles,
            "token_question_indices": token_question_indices,
            "token_option_indices": token_option_indices,
            "decision_indices": decision_indices,
            "num_choices": num_choices,
            "question_mask": question_mask,
            "score_mask": score_mask,
            "targets": targets,
        }
        if tree_attention_mask is not None:
            batch["tree_attention_mask"] = tree_attention_mask
        return batch
