from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .choice_set_probe import build_choice_set_probe, summarize_choice_set_probe
from .diagnose_choice import predict_decision_rows
from .diagnose_score import select_source_bundles, summarize_score_predictions
from .schema import QuestionType


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only Score and dynamic Choice development diagnostics"
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--mode", choices=("all", "score", "choice-set"), default="all")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--score-split", choices=("train", "validation"), default="validation")
    parser.add_argument("--score-source")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    if args.score_source is not None and not re.fullmatch(r"[a-z0-9_]+", args.score_source):
        parser.error("score-source must contain only lowercase letters, digits, and underscores")
    if not (args.run_dir / "final" / "decision_head.pt").is_file():
        parser.error("run directory needs a saved final model")

    import torch
    from transformers import AutoTokenizer

    from .hf_data import load_configured_bundles, select_question_type
    from .model import DecisionModel
    from .train import load_config

    config = load_config(args.run_dir / "training_config.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float32
    )
    tokenizer_dir = args.run_dir / "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir if tokenizer_dir.is_dir() else config["model_id"],
        trust_remote_code=bool(config.get("trust_remote_code", True)),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = DecisionModel.load_components(
        args.run_dir / "final",
        model_id=config["model_id"],
        dtype=dtype,
        trust_remote_code=bool(config.get("trust_remote_code", True)),
        attn_implementation=config.get("attn_implementation", "eager"),
    ).to(device)
    model.eval()

    if args.mode in ("all", "score"):
        # This command deliberately supports only configured train/development data.
        source_bundles = load_configured_bundles(config, args.score_split)
        if args.score_source is not None:
            source_bundles = select_source_bundles(source_bundles, args.score_source)
        score_bundles = select_question_type(source_bundles, QuestionType.SCORE)
        score_rows = predict_decision_rows(
            score_bundles, model=model, tokenizer=tokenizer, config=config,
            batch_size=args.batch_size, device=device, question_type=QuestionType.SCORE,
        )
        score_report = summarize_score_predictions(score_rows)
        score_report["view"] = {
            "split": args.score_split,
            "source": args.score_source,
            "score_branches_only": True,
            "configured_bundle_limit": config.get(f"{args.score_split}_bundle_limit"),
        }
        source_suffix = f"_{args.score_source}" if args.score_source else ""
        score_output = (
            args.run_dir / f"score_{args.score_split}{source_suffix}_diagnostic.json"
        )
        score_output.write_text(
            json.dumps(score_report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        for name, summary in (
            ("overall", score_report["overall"]),
            *score_report["by_dimension"].items(),
        ):
            print(
                f"score {args.score_split} {name} n={summary['count']} "
                f"bias={summary['model']['expected_score_bias']:+.4f} "
                f"expected_mae={summary['model']['expected_score_mae']:.4f} "
                f"uniform_mae={summary['uniform']['expected_score_mae']:.4f} "
                f"rps={summary['model']['ranked_probability_score']:.4f} "
                f"uniform_rps={summary['uniform']['ranked_probability_score']:.4f} "
                f"mode_misses=near:{summary['adjacent_mode_miss']} "
                f"far:{summary['far_mode_miss']} "
                f"under:{summary['underpredicted_mode']} over:{summary['overpredicted_mode']} "
                f"ties:{summary['target_mode_ties_excluded']}"
            )
        for example in score_report["worst_examples_by_dimension"]:
            if example["group"] not in (
                "helpsteer2/correctness", "helpsteer2/helpfulness"
            ):
                continue
            print(
                f"score {args.score_split} worst {example['group']} "
                f"bundle={example['bundle_id']} "
                f"expected={example['target_expected_score']:.3f} "
                f"predicted={example['predicted_expected_score']:.3f} "
                f"error={example['expected_score_error']:+.3f} "
                f"target_modes={example['target_modes']} "
                f"predicted_mode={example['predicted_mode']} "
                f"rps={example['ranked_probability_score']:.3f}"
            )
            print(f"  state: {example['state_excerpt']}")
            print(f"  question: {example['instructions']}")
        print(f"score diagnostic saved: {score_output}")

    if args.mode in ("all", "choice-set"):
        probe_bundles = build_choice_set_probe()
        probe_rows = predict_decision_rows(
            probe_bundles, model=model, tokenizer=tokenizer, config=config,
            batch_size=args.batch_size, device=device, question_type=QuestionType.CHOICE,
        )
        probe_report = summarize_choice_set_probe(probe_rows)
        probe_output = args.run_dir / "choice_set_probe_diagnostic.json"
        probe_output.write_text(
            json.dumps(probe_report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        for name, summary in probe_report["by_variant"].items():
            print(
                f"choice-set {name} n={summary['count']} "
                f"brier={summary['model']['brier']:.4f} "
                f"uniform_brier={summary['uniform']['brier']:.4f} "
                f"kl={summary['model']['kl_divergence']:.4f}"
            )
        print(
            "choice-set remainder absolute error "
            f"none={probe_report['mean_none_absolute_error']:.4f} "
            f"explicit={probe_report['mean_explicit_absolute_error']:.4f}"
        )
        print(f"choice-set probe saved: {probe_output}")


if __name__ == "__main__":
    main()
