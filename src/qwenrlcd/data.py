from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable, Sequence

from .formatting import render_question, render_state
from .schema import DecisionBundle


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


def build_tree_attention_pattern(
    sequence_length: int,
    state_length: int,
    branch_spans: Sequence[tuple[int, int]],
) -> tuple[tuple[bool, ...], ...]:
    """Return a causal shared-prefix mask with mutually isolated question branches."""
    if not 0 < state_length <= sequence_length:
        raise ValueError("state_length must be within the sequence")
    allowed = [[False] * sequence_length for _ in range(sequence_length)]

    for query in range(state_length):
        for key in range(query + 1):
            allowed[query][key] = True

    for start, end in branch_spans:
        if not state_length <= start < end <= sequence_length:
            raise ValueError(f"invalid branch span {(start, end)}")
        for query in range(start, end):
            for key in range(state_length):
                allowed[query][key] = True
            for key in range(start, query + 1):
                allowed[query][key] = True

    return tuple(tuple(row) for row in allowed)


class DecisionDataset(Sequence[dict[str, Any]]):
    """Pack one state and isolated question branches into one model sequence."""

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
    ) -> None:
        self.bundles = list(bundles)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_choices = max_choices
        self.max_questions = max_questions
        self.shuffle = shuffle
        self.seed = seed
        self.require_targets = require_targets
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
            rng = random.Random(self.seed + self.epoch * len(self.bundles) + index)
            bundle = bundle.permuted(rng)

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
        branch_ids = [self._encode(render_question(question)) for question in bundle.questions]
        if any(not branch for branch in branch_ids):
            raise ValueError(f"bundle {bundle.id} contains an empty question branch")

        branch_token_count = sum(map(len, branch_ids))
        state_budget = self.max_length - branch_token_count
        if state_budget < 1:
            raise ValueError(
                f"bundle {bundle.id} question branches require {branch_token_count} tokens, "
                f"leaving no state space within max_length={self.max_length}"
            )
        if len(state_ids) > state_budget:
            state_ids = state_ids[-state_budget:]

        input_ids = list(state_ids)
        position_ids = list(range(len(state_ids)))
        branch_spans: list[tuple[int, int]] = []
        decision_indices: list[int] = []
        for branch in branch_ids:
            start = len(input_ids)
            input_ids.extend(branch)
            end = len(input_ids)
            branch_spans.append((start, end))
            decision_indices.append(end - 1)
            position_ids.extend(range(len(state_ids), len(state_ids) + len(branch)))

        return {
            "id": bundle.id,
            "question_ids": [question.id for question in bundle.questions],
            "input_ids": input_ids,
            "position_ids": position_ids,
            "state_length": len(state_ids),
            "branch_spans": branch_spans,
            "decision_indices": decision_indices,
            "num_choices": [len(question.options) for question in bundle.questions],
            "targets": [
                question.target_vector()
                if question.target is not None
                else [0.0] * len(question.options)
                for question in bundle.questions
            ],
        }


class DecisionCollator:
    def __init__(self, tokenizer: Any, max_choices: int, max_questions: int) -> None:
        self.tokenizer = tokenizer
        self.max_choices = max_choices
        self.max_questions = max_questions

    def __call__(self, examples: Sequence[dict[str, Any]]) -> dict[str, Any]:
        import torch

        batch_size = len(examples)
        max_sequence_length = max(len(example["input_ids"]) for example in examples)
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            raise ValueError("tokenizer must define pad_token_id")

        input_ids = torch.full(
            (batch_size, max_sequence_length), pad_token_id, dtype=torch.long
        )
        position_ids = torch.zeros((batch_size, max_sequence_length), dtype=torch.long)
        tree_attention_mask = torch.zeros(
            (batch_size, max_sequence_length, max_sequence_length), dtype=torch.bool
        )
        decision_indices = torch.zeros(
            (batch_size, self.max_questions), dtype=torch.long
        )
        num_choices = torch.zeros((batch_size, self.max_questions), dtype=torch.long)
        question_mask = torch.zeros((batch_size, self.max_questions), dtype=torch.bool)
        targets = torch.zeros(
            (batch_size, self.max_questions, self.max_choices), dtype=torch.float32
        )
        question_ids: list[list[str | None]] = []

        for row, example in enumerate(examples):
            sequence_length = len(example["input_ids"])
            question_count = len(example["decision_indices"])
            input_ids[row, :sequence_length] = torch.tensor(example["input_ids"])
            position_ids[row, :sequence_length] = torch.tensor(example["position_ids"])
            pattern = build_tree_attention_pattern(
                sequence_length, example["state_length"], example["branch_spans"]
            )
            tree_attention_mask[row, :sequence_length, :sequence_length] = torch.tensor(
                pattern, dtype=torch.bool
            )
            for padding_index in range(sequence_length, max_sequence_length):
                tree_attention_mask[row, padding_index, padding_index] = True

            decision_indices[row, :question_count] = torch.tensor(
                example["decision_indices"], dtype=torch.long
            )
            num_choices[row, :question_count] = torch.tensor(
                example["num_choices"], dtype=torch.long
            )
            question_mask[row, :question_count] = True
            for question_index, target in enumerate(example["targets"]):
                targets[row, question_index, : len(target)] = torch.tensor(target)

            ids = list(example["question_ids"])
            question_ids.append(ids + [None] * (self.max_questions - len(ids)))

        return {
            "ids": [example["id"] for example in examples],
            "question_ids": question_ids,
            "input_ids": input_ids,
            "position_ids": position_ids,
            "tree_attention_mask": tree_attention_mask,
            "decision_indices": decision_indices,
            "num_choices": num_choices,
            "question_mask": question_mask,
            "targets": targets,
        }
