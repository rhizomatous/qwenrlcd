from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .schema import DecisionBundle, Option, Question, QuestionType

DATASET_SCHEMA_VERSION = "qwenrlcd.bundle.v1"


def bundle_from_dataset_row(row: Mapping[str, Any]) -> DecisionBundle:
    """Read the dataset's decision fields, not its audit/provenance details."""
    if row["schema_version"] != DATASET_SCHEMA_VERSION:
        raise ValueError(f"unsupported dataset schema: {row['schema_version']}")
    questions = []
    for raw_question in row["questions"]:
        options = tuple(
            Option(
                key=str(raw_option["key"]),
                description=json.loads(raw_option["description_json"]),
            )
            for raw_option in raw_question["options"]
        )
        questions.append(
            Question(
                id=str(raw_question["id"]),
                type=QuestionType(raw_question["type"]),
                instructions=json.loads(raw_question["instructions_json"]),
                options=options,
                target={
                    option.key: float(probability)
                    for option, probability in zip(
                        options, raw_question["target"], strict=True
                    )
                },
            )
        )
    return DecisionBundle(
        id=str(row["id"]),
        state=json.loads(row["state_json"]),
        questions=tuple(questions),
        source=str(row["provenance"]["source"]),
    )


class HFDatasetBundles(Sequence[DecisionBundle]):
    """Lazy adapter for a standard Hugging Face dataset split."""

    def __init__(self, dataset: Any) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> DecisionBundle:
        return bundle_from_dataset_row(self.dataset[index])


def load_hf_bundles(path: str, config: str, split: str) -> HFDatasetBundles:
    from datasets import load_dataset

    local_path = Path(path).expanduser()
    dataset_path = str(local_path.resolve()) if local_path.exists() else path
    return HFDatasetBundles(load_dataset(dataset_path, config, split=split))
