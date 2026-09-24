from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, TextIO


def _open_output(path: Path | None) -> tuple[TextIO, bool]:
    if path is None:
        return sys.stdout, False
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("w", encoding="utf-8"), True


def main() -> None:
    parser = argparse.ArgumentParser(description="Run typed parallel decision inference")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    from .data import (
        DecisionCollator,
        DecisionDataset,
        decision_model_inputs,
        read_jsonl,
    )
    from .losses import mask_invalid_choices
    from .model import DecisionModel
    from .prediction import format_bundle_prediction
    from .train import load_config

    config: dict[str, Any] = load_config(args.run_dir / "training_config.json")
    bundles = read_jsonl(args.input)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float32
    )

    tokenizer_source = args.run_dir / "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source if tokenizer_source.is_dir() else config["model_id"],
        trust_remote_code=bool(config.get("trust_remote_code", True)),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dataset = DecisionDataset(
        bundles,
        tokenizer,
        max_length=int(config["max_length"]),
        max_choices=int(config["max_choices"]),
        max_questions=int(config["max_questions"]),
        compact_attention_topology=(
            bool(config.get("compact_attention_topology", False))
            or config.get("attn_implementation", "eager") == "flex_attention"
        ),
        shuffle=False,
        seed=int(config["seed"]),
        require_targets=False,
    )
    collator = DecisionCollator(
        tokenizer,
        max_choices=int(config["max_choices"]),
        max_questions=int(config["max_questions"]),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    model = DecisionModel.load_components(
        args.run_dir / "final",
        model_id=config["model_id"],
        dtype=dtype,
        trust_remote_code=bool(config.get("trust_remote_code", True)),
        attn_implementation=config.get("attn_implementation", "eager"),
        flex_block_size=int(config.get("flex_block_size", 128)),
    ).to(device)
    model.eval()

    output, should_close = _open_output(args.output)
    bundle_offset = 0
    try:
        with torch.inference_mode():
            for batch in dataloader:
                logits = model(**decision_model_inputs(batch, device))
                probabilities = torch.softmax(
                    mask_invalid_choices(logits.float(), batch["num_choices"].to(device)),
                    dim=-1,
                ).cpu()

                batch_size = len(batch["ids"])
                for row, bundle in enumerate(
                    bundles[bundle_offset : bundle_offset + batch_size]
                ):
                    rows = [
                        probabilities[row, index, : len(question.options)].tolist()
                        for index, question in enumerate(bundle.questions)
                    ]
                    prediction = format_bundle_prediction(bundle, rows)
                    output.write(json.dumps(prediction, ensure_ascii=False) + "\n")
                bundle_offset += batch_size
    finally:
        if should_close:
            output.close()


if __name__ == "__main__":
    main()
