from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .formatting import render_option, render_question_prefix, render_state
from .hf_data import load_configured_bundles
from .schema import DecisionBundle
from .train import load_config


def _percentile(sorted_values: Sequence[int], fraction: float) -> int:
    return sorted_values[math.ceil(fraction * len(sorted_values)) - 1]


def inspect_token_lengths(
    bundles: Sequence[DecisionBundle],
    tokenizer: Any,
    *,
    max_length: int,
    max_questions: int,
    max_choices: int,
) -> dict[str, Any]:
    if not bundles:
        raise ValueError("cannot inspect an empty split")
    packed_lengths: list[int] = []
    question_counts: list[int] = []
    source_counts: Counter[str] = Counter()
    over_budget: list[dict[str, Any]] = []
    for bundle in bundles:
        if len(bundle.questions) > max_questions:
            raise ValueError(f"bundle {bundle.id} exceeds max_questions={max_questions}")
        if any(len(question.options) > max_choices for question in bundle.questions):
            raise ValueError(f"bundle {bundle.id} exceeds max_choices={max_choices}")
        state_length = len(tokenizer(render_state(bundle), add_special_tokens=False)["input_ids"])
        branch_length = sum(
            len(tokenizer(render_question_prefix(question), add_special_tokens=False)["input_ids"])
            + sum(
                len(tokenizer(render_option(option), add_special_tokens=False)["input_ids"])
                for option in question.options
            )
            for question in bundle.questions
        )
        packed_length = state_length + branch_length
        packed_lengths.append(packed_length)
        question_counts.append(len(bundle.questions))
        source_counts[bundle.source or "unspecified"] += 1
        if packed_length > max_length:
            over_budget.append(
                {"id": bundle.id, "source": bundle.source, "tokens": packed_length}
            )
    packed_lengths.sort()
    return {
        "bundles": len(bundles),
        "source_counts": dict(sorted(source_counts.items())),
        "question_count_max": max(question_counts),
        "packed_tokens": {
            "min": packed_lengths[0],
            "p50": _percentile(packed_lengths, 0.50),
            "p95": _percentile(packed_lengths, 0.95),
            "p99": _percentile(packed_lengths, 0.99),
            "max": packed_lengths[-1],
        },
        "over_max_length": len(over_budget),
        "over_max_length_examples": over_budget[:5],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect pilot token lengths without a GPU")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config["model_id"], trust_remote_code=bool(config.get("trust_remote_code", True))
    )
    failed = False
    for split in ("train", "validation"):
        summary = inspect_token_lengths(
            load_configured_bundles(config, split),
            tokenizer,
            max_length=int(config["max_length"]),
            max_questions=int(config["max_questions"]),
            max_choices=int(config["max_choices"]),
        )
        print(split, json.dumps(summary, sort_keys=True))
        failed |= summary["over_max_length"] > 0
    if failed:
        raise SystemExit("pilot has bundles exceeding max_length; do not train with truncation")


if __name__ == "__main__":
    main()
