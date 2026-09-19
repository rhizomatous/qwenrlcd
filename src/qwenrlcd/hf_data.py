from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
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


def source_stratified_indices(
    ids: Sequence[str],
    sources: Sequence[str],
    *,
    limit: int,
    seed: int,
    split: str,
) -> list[int]:
    """Choose a stable, roughly proportional subset without splitting bundles."""
    if len(ids) != len(sources) or not ids:
        raise ValueError("IDs and sources must have equal, non-zero length")
    if not 1 <= limit <= len(ids):
        raise ValueError(f"sample limit must be between 1 and {len(ids)}")
    groups: dict[str, list[tuple[bytes, str, int]]] = defaultdict(list)
    for index, (bundle_id, source) in enumerate(zip(ids, sources, strict=True)):
        digest = hashlib.sha256(f"{seed}\0{split}\0{bundle_id}".encode()).digest()
        groups[source].append((digest, bundle_id, index))
    if limit < len(groups):
        raise ValueError("sample limit must include at least one bundle per source")

    counts = Counter({source: len(group) for source, group in groups.items()})
    fractions = {source: limit * count / len(ids) for source, count in counts.items()}
    quotas = {source: math.floor(fraction) for source, fraction in fractions.items()}
    extra = limit - sum(quotas.values())
    for source in sorted(
        groups,
        key=lambda name: (-(fractions[name] - quotas[name]), name),
    )[:extra]:
        quotas[source] += 1
    for source in sorted(name for name, quota in quotas.items() if quota == 0):
        donor = max(
            (name for name, quota in quotas.items() if quota > 1),
            key=lambda name: (quotas[name], name),
        )
        quotas[donor] -= 1
        quotas[source] = 1

    selected = [
        index
        for source, group in groups.items()
        for _, _, index in sorted(group)[:quotas[source]]
    ]
    return sorted(selected)


def _id_and_source_columns(dataset: Any) -> tuple[list[str], list[str]]:
    """Read only Arrow ID/source columns, not every row's text or provenance."""
    table = dataset.data.table
    id_chunks = table.column("id").chunks
    provenance_chunks = table.column("provenance").chunks
    if len(id_chunks) != len(provenance_chunks):
        raise ValueError("dataset ID and provenance Arrow chunks are misaligned")
    ids: list[str] = []
    sources: list[str] = []
    for id_chunk, provenance_chunk in zip(id_chunks, provenance_chunks, strict=True):
        if len(id_chunk) != len(provenance_chunk):
            raise ValueError("dataset ID and provenance Arrow chunks are misaligned")
        ids.extend(id_chunk.to_pylist())
        sources.extend(provenance_chunk.field("source").to_pylist())
    return ids, sources


def load_hf_bundles(
    path: str,
    config: str,
    split: str,
    *,
    sample_size: int | None = None,
    seed: int = 17,
) -> HFDatasetBundles:
    from datasets import load_dataset

    local_path = Path(path).expanduser()
    dataset_path = str(local_path.resolve()) if local_path.exists() else path
    dataset = load_dataset(dataset_path, config, split=split)
    if sample_size is not None:
        ids, sources = _id_and_source_columns(dataset)
        indices = source_stratified_indices(
            ids, sources, limit=sample_size, seed=seed, split=split
        )
        dataset = dataset.select(indices)
    return HFDatasetBundles(dataset)


def load_configured_bundles(config: Mapping[str, Any], split: str) -> Sequence[DecisionBundle]:
    if split not in {"train", "validation"}:
        raise ValueError(f"unsupported configured split: {split}")
    if "dataset_path" in config:
        if "train_file" in config or "validation_file" in config:
            raise ValueError("configure either dataset_path or JSONL files, not both")
        return load_hf_bundles(
            str(config["dataset_path"]),
            str(config.get("dataset_config", "core")),
            split,
            sample_size=config.get(f"{split}_bundle_limit"),
            seed=int(config["seed"]),
        )
    if "train_bundle_limit" in config or "validation_bundle_limit" in config:
        raise ValueError("bundle limits require dataset_path")
    from .data import read_jsonl

    return read_jsonl(config[f"{split}_file"])
