from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from .data import read_jsonl, write_jsonl
from .hf_data import load_configured_bundles, select_question_type
from .schema import DecisionBundle, QuestionType
from .train import load_config


def select_capacity_bundles(
    bundles: Sequence[DecisionBundle], *, per_source: int, seed: int,
    expected_sources: int = 4,
) -> list[DecisionBundle]:
    """Select the same small, source-balanced Choice set regardless of row order."""
    if per_source < 1:
        raise ValueError("per_source must be positive")
    choices = select_question_type(bundles, QuestionType.CHOICE)
    groups: dict[str, list[DecisionBundle]] = defaultdict(list)
    for bundle in choices:
        groups[bundle.source or "unspecified"].append(bundle)
    if len(groups) != expected_sources:
        raise ValueError(f"expected {expected_sources} Choice sources; found {sorted(groups)}")

    selected = []
    for source, group in sorted(groups.items()):
        if len(group) < per_source:
            raise ValueError(f"source {source} has only {len(group)} Choice bundles")
        ranked = sorted(
            group,
            key=lambda bundle: (
                hashlib.sha256(f"{seed}\0choice-capacity\0{bundle.id}".encode()).digest(),
                bundle.id,
            ),
        )
        selected.extend(ranked[:per_source])
    return selected


def write_capacity_fixture(bundles: Sequence[DecisionBundle], path: Path) -> bool:
    """Write once; reruns may reuse but never silently replace a different fixture."""
    if path.exists():
        if read_jsonl(path) != list(bundles):
            raise ValueError(f"existing capacity fixture differs: {path}")
        return False
    write_jsonl(bundles, path)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a private Choice capacity fixture")
    parser.add_argument("--pilot-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-source", type=int, default=8)
    args = parser.parse_args()
    pilot_config = load_config(args.pilot_config)
    bundles = select_capacity_bundles(
        load_configured_bundles(pilot_config, "train"),
        per_source=args.per_source,
        seed=int(pilot_config["seed"]),
    )
    cardinalities = {
        len(question.options) for bundle in bundles for question in bundle.questions
    }
    required = {3, 6, 28}
    if not required <= cardinalities:
        raise ValueError(f"fixture must include Choice cardinalities {sorted(required)}")
    created = write_capacity_fixture(bundles, args.output)
    print(
        f"capacity fixture: {'created' if created else 'reused'} {args.output} "
        f"bundles={len(bundles)} questions={sum(len(b.questions) for b in bundles)} "
        f"cardinalities={sorted(cardinalities)}"
    )


if __name__ == "__main__":
    main()
