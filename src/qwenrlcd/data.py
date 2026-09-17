from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from .formatting import render_record
from .schema import DecisionRecord


def read_jsonl(path: str | Path) -> list[DecisionRecord]:
    records: list[DecisionRecord] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(DecisionRecord.from_dict(json.loads(line)))
            except Exception as exc:
                raise ValueError(f"invalid record at {path}:{line_number}: {exc}") from exc
    if not records:
        raise ValueError(f"no records found in {path}")
    return records


def write_jsonl(records: Iterable[DecisionRecord], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")


class DecisionDataset(Sequence[dict[str, Any]]):
    """Tokenizes records and deterministically permutes options per epoch."""

    def __init__(
        self,
        records: Sequence[DecisionRecord],
        tokenizer: Any,
        *,
        max_length: int,
        max_choices: int,
        shuffle_options: bool,
        seed: int,
    ) -> None:
        self.records = list(records)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_choices = max_choices
        self.shuffle_options = shuffle_options
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        if self.shuffle_options:
            rng = random.Random(self.seed + self.epoch * len(self.records) + index)
            record = record.permuted(rng)

        if len(record.question.options) > self.max_choices:
            raise ValueError(
                f"record {record.id} has {len(record.question.options)} options, "
                f"above configured maximum {self.max_choices}"
            )

        encoded = self.tokenizer(
            render_record(record),
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
        )
        return {
            "id": record.id,
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "num_choices": len(record.question.options),
            "target": record.target_vector(),
        }


class DecisionCollator:
    def __init__(self, tokenizer: Any, max_choices: int) -> None:
        self.tokenizer = tokenizer
        self.max_choices = max_choices

    def __call__(self, examples: Sequence[dict[str, Any]]) -> dict[str, Any]:
        import torch

        padded = self.tokenizer.pad(
            [
                {
                    "input_ids": example["input_ids"],
                    "attention_mask": example["attention_mask"],
                }
                for example in examples
            ],
            padding=True,
            return_tensors="pt",
        )
        targets = torch.zeros((len(examples), self.max_choices), dtype=torch.float32)
        num_choices = torch.tensor(
            [example["num_choices"] for example in examples], dtype=torch.long
        )
        for row, example in enumerate(examples):
            target = torch.tensor(example["target"], dtype=torch.float32)
            targets[row, : target.numel()] = target

        padded["decision_indices"] = padded["attention_mask"].sum(dim=1) - 1
        padded["num_choices"] = num_choices
        padded["targets"] = targets
        return dict(padded)


def batches(items: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]
