from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

COHORT_KEYS = (
    "dataset_path",
    "dataset_config",
    "train_bundle_limit",
    "validation_bundle_limit",
    "seed",
    "max_length",
    "max_questions",
    "max_choices",
    "require_no_state_truncation",
    "permute_training",
    "epochs",
    "micro_batch_size",
    "gradient_accumulation_steps",
    "head_learning_rate",
    "weight_decay",
    "warmup_ratio",
    "max_grad_norm",
    "gradient_checkpointing",
    "ce_weight",
    "brier_weight",
    "ordinal_rps_weight",
)

COHORT_DEFAULTS = {
    "ordinal_rps_weight": 0.0,
}


def normalized_cohort(config: dict[str, Any]) -> dict[str, Any]:
    return {
        key: config.get(key, COHORT_DEFAULTS.get(key))
        for key in COHORT_KEYS
    }


def load_run(run_dir: str | Path) -> dict[str, Any]:
    path = Path(run_dir)
    with (path / "training_config.json").open(encoding="utf-8") as handle:
        config = json.load(handle)
    with (path / "validation_metrics.json").open(encoding="utf-8") as handle:
        history = json.load(handle)
    if not history:
        raise ValueError(f"validation history is empty: {path}")
    metrics = history[-1]
    lora = config.get("lora", {})
    adaptation = config.get(
        "adaptation", "lora" if lora.get("enabled", False) else "full_finetune"
    )
    return {
        "run_dir": str(path),
        "model_id": config["model_id"],
        "adaptation": adaptation,
        "attention_backend": config.get("attn_implementation", "eager"),
        "learning_rate": config["learning_rate"],
        "head_learning_rate": config["head_learning_rate"],
        "step": metrics["step"],
        "config": config,
        "metrics": metrics,
    }


def validate_same_cohort(runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    labels = list(runs)
    cohort = normalized_cohort(runs[labels[0]]["config"])
    for label in labels[1:]:
        candidate = normalized_cohort(runs[label]["config"])
        differences = {
            key: (cohort[key], candidate[key])
            for key in COHORT_KEYS
            if cohort[key] != candidate[key]
        }
        if differences:
            raise ValueError(
                f"{label} does not use the control cohort/schedule: {differences}"
            )
    return cohort


def metric_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: metrics[key]
        for key in (
            "bundles",
            "questions",
            "accuracy",
            "expected_accuracy",
            "brier",
            "uniform_brier",
            "kl",
            "uniform_kl",
            "ece",
        )
    } | {
        "by_source": metrics["slices"]["by_source"],
        "by_type": metrics["slices"]["by_type"],
    }


def build_report(runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    cohort = validate_same_cohort(runs)
    control = runs["control"]["metrics"]
    summarized: dict[str, Any] = {}
    deltas: dict[str, Any] = {}
    for label, run in runs.items():
        summarized[label] = {
            key: run[key]
            for key in (
                "run_dir",
                "model_id",
                "adaptation",
                "attention_backend",
                "learning_rate",
                "head_learning_rate",
                "step",
            )
        } | {"development_validation": metric_summary(run["metrics"])}
        if label != "control":
            metrics = run["metrics"]
            deltas[label] = {
                "brier": metrics["brier"] - control["brier"],
                "kl": metrics["kl"] - control["kl"],
                "accuracy": metrics["accuracy"] - control["accuracy"],
                "by_type": {
                    question_type: {
                        "brier": values["brier"]
                        - control["slices"]["by_type"][question_type]["brier"],
                        "kl": values["kl_divergence"]
                        - control["slices"]["by_type"][question_type]["kl_divergence"],
                    }
                    for question_type, values in metrics["slices"]["by_type"].items()
                },
            }
    return {
        "purpose": "development-only adaptation/capacity comparison",
        "test_splits_opened": False,
        "cohort": cohort,
        "runs": summarized,
        "deltas_vs_control": deltas,
    }


def print_report(report: dict[str, Any]) -> None:
    print("capacity comparison: identical 10k train / 500 development cohort")
    for label, run in report["runs"].items():
        metrics = run["development_validation"]
        print(
            f"{label}: model={run['model_id']} adaptation={run['adaptation']} "
            f"backend={run['attention_backend']} step={run['step']} "
            f"brier={metrics['brier']:.4f} kl={metrics['kl']:.4f} "
            f"accuracy={metrics['accuracy']:.4f} ece={metrics['ece']:.4f}"
        )
        for question_type, values in metrics["by_type"].items():
            print(
                f"  {question_type}: n={values['count']} "
                f"brier={values['brier']:.4f} kl={values['kl_divergence']:.4f}"
            )
        for source, values in metrics["by_source"].items():
            print(
                f"  {source}: n={values.get('questions', values.get('count'))} "
                f"brier={values['brier']:.4f} kl={values['kl_divergence']:.4f}"
            )
    print("deltas versus 1.7B LoRA control (negative Brier/KL is better)")
    for label, values in report["deltas_vs_control"].items():
        print(
            f"{label}: brier={values['brier']:+.4f} kl={values['kl']:+.4f} "
            f"accuracy={values['accuracy']:+.4f}"
        )
        for question_type, type_values in values["by_type"].items():
            print(
                f"  {question_type}: brier={type_values['brier']:+.4f} "
                f"kl={type_values['kl']:+.4f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare model capacity on one sealed-test-safe development cohort"
    )
    parser.add_argument("--control-run", required=True, type=Path)
    parser.add_argument("--full-run", required=True, type=Path)
    parser.add_argument("--large-lora-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    runs = {
        "control": load_run(args.control_run),
        "full_finetune": load_run(args.full_run),
        "large_lora": load_run(args.large_lora_run),
    }
    report = build_report(runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print_report(report)
    print(f"comparison saved: {args.output}")


if __name__ == "__main__":
    main()
