from __future__ import annotations

import argparse
from pathlib import Path

from .schema import DecisionBundle
from .train import load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify packed questions match isolated Qwen evaluations"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--atol", type=float, default=0.005)
    parser.add_argument("--rtol", type=float, default=0.005)
    args = parser.parse_args()
    config = load_config(args.config)

    import torch
    from transformers import AutoTokenizer

    from .data import DecisionCollator, DecisionDataset, read_jsonl
    from .model import DecisionModel

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the real-model parallelism check")

    torch.manual_seed(int(config["seed"]))
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_id"], trust_remote_code=bool(config.get("trust_remote_code", True))
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = DecisionModel.from_pretrained(
        config["model_id"],
        max_choices=int(config["max_choices"]),
        trust_remote_code=bool(config.get("trust_remote_code", True)),
        attn_implementation=config.get("attn_implementation", "eager"),
    )
    lora = config.get("lora", {})
    if lora.get("enabled", True):
        model.attach_lora(
            rank=int(lora["rank"]),
            alpha=int(lora["alpha"]),
            dropout=float(lora["dropout"]),
            target_modules=lora["target_modules"],
        )
    model = model.cuda()
    model.eval()

    collator = DecisionCollator(
        tokenizer,
        max_choices=int(config["max_choices"]),
        max_questions=int(config["max_questions"]),
    )

    def run(bundles: list[DecisionBundle]) -> tuple[torch.Tensor, list[list[str | None]]]:
        dataset = DecisionDataset(
            bundles,
            tokenizer,
            max_length=int(config["max_length"]),
            max_choices=int(config["max_choices"]),
            max_questions=int(config["max_questions"]),
            shuffle=False,
            seed=int(config["seed"]),
        )
        batch = collator([dataset[index] for index in range(len(dataset))])
        with torch.inference_mode():
            logits = model(
                input_ids=batch["input_ids"].cuda(),
                position_ids=batch["position_ids"].cuda(),
                tree_attention_mask=batch["tree_attention_mask"].cuda(),
                decision_indices=batch["decision_indices"].cuda(),
            )
        return logits.float().cpu(), batch["question_ids"]

    bundle = read_jsonl(config["validation_file"])[0]
    if len(bundle.questions) < 2:
        raise SystemExit("parallelism check requires a bundle with at least two questions")

    bundled_logits, bundled_ids = run([bundle])
    singletons = [
        DecisionBundle(f"{bundle.id}-{question.id}", bundle.state, (question,))
        for question in bundle.questions
    ]
    singleton_logits, singleton_ids = run(singletons)
    reversed_bundle = DecisionBundle(bundle.id, bundle.state, tuple(reversed(bundle.questions)))
    reversed_logits, reversed_ids = run([reversed_bundle])

    bundled_by_id = {
        question_id: bundled_logits[0, index]
        for index, question_id in enumerate(bundled_ids[0])
        if question_id is not None
    }
    singleton_by_id = {
        ids[0]: singleton_logits[index, 0]
        for index, ids in enumerate(singleton_ids)
        if ids[0] is not None
    }
    reversed_by_id = {
        question_id: reversed_logits[0, index]
        for index, question_id in enumerate(reversed_ids[0])
        if question_id is not None
    }

    max_singleton_delta = 0.0
    max_reordered_delta = 0.0
    for question in bundle.questions:
        size = len(question.options)
        bundled = bundled_by_id[question.id][:size]
        singleton = singleton_by_id[question.id][:size]
        reordered = reversed_by_id[question.id][:size]
        torch.testing.assert_close(singleton, bundled, atol=args.atol, rtol=args.rtol)
        torch.testing.assert_close(reordered, bundled, atol=args.atol, rtol=args.rtol)
        max_singleton_delta = max(
            max_singleton_delta, float((singleton - bundled).abs().max())
        )
        max_reordered_delta = max(
            max_reordered_delta, float((reordered - bundled).abs().max())
        )

    print("parallel invariance passed")
    print(f"questions={len(bundle.questions)}")
    print(f"max_singleton_logit_delta={max_singleton_delta:.6f}")
    print(f"max_reordered_logit_delta={max_reordered_delta:.6f}")


if __name__ == "__main__":
    main()
