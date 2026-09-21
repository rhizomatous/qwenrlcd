from __future__ import annotations

import argparse
import json
from pathlib import Path

from .choice_set_probe import build_choice_set_probe, summarize_choice_set_probe
from .diagnose_choice import predict_decision_rows
from .diagnose_score import summarize_score_predictions
from .schema import QuestionType


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only Score and dynamic Choice development diagnostics"
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--mode", choices=("all", "score", "choice-set"), default="all")
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
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
        # Only the configured development-validation sample is read. Never load test here.
        source_bundles = load_configured_bundles(config, "validation")
        score_bundles = select_question_type(source_bundles, QuestionType.SCORE)
        score_rows = predict_decision_rows(
            score_bundles, model=model, tokenizer=tokenizer, config=config,
            batch_size=args.batch_size, device=device, question_type=QuestionType.SCORE,
        )
        score_report = summarize_score_predictions(score_rows)
        score_report["view"] = {
            "split": "validation",
            "score_branches_only": True,
            "configured_validation_bundle_limit": config.get("validation_bundle_limit"),
        }
        score_output = args.run_dir / "score_validation_diagnostic.json"
        score_output.write_text(
            json.dumps(score_report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        for name, summary in (
            ("overall", score_report["overall"]),
            *score_report["by_dimension"].items(),
        ):
            print(
                f"score {name} n={summary['count']} "
                f"expected_mae={summary['model']['expected_score_mae']:.4f} "
                f"uniform_mae={summary['uniform']['expected_score_mae']:.4f} "
                f"rps={summary['model']['ranked_probability_score']:.4f} "
                f"uniform_rps={summary['uniform']['ranked_probability_score']:.4f} "
                f"mode_misses=near:{summary['adjacent_mode_miss']} "
                f"far:{summary['far_mode_miss']} ties:{summary['target_mode_ties_excluded']}"
            )
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
