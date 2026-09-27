from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from .compare_capacity import load_run, metric_summary, normalized_cohort


def select_best_full_run(runs: list[dict[str, Any]]) -> dict[str, Any]:
    if not runs:
        raise ValueError("full-fine-tune sweep is empty")
    for run in runs:
        if run["adaptation"] != "full_finetune":
            raise ValueError("full sweep contains a non-full-fine-tune run")
    return min(runs, key=lambda run: (run["metrics"]["brier"], run["metrics"]["kl"]))


def select_best_lora_run(runs: list[dict[str, Any]]) -> dict[str, Any]:
    if not runs:
        raise ValueError("LoRA sweep is empty")
    for run in runs:
        if run["adaptation"] != "lora":
            raise ValueError("LoRA sweep contains a non-LoRA run")
    return min(runs, key=lambda run: (run["metrics"]["brier"], run["metrics"]["kl"]))


def fixed_sample_cohort(config: dict[str, Any]) -> dict[str, Any]:
    cohort = normalized_cohort(config)
    cohort.pop("seed")
    return cohort


def _index_by_seed(
    runs: list[dict[str, Any]], *, adaptation: str
) -> dict[int, dict[str, Any]]:
    indexed: dict[int, dict[str, Any]] = {}
    for run in runs:
        if run["adaptation"] != adaptation:
            raise ValueError(f"expected {adaptation}, found {run['adaptation']}")
        seed = int(run["config"]["seed"])
        if seed in indexed:
            raise ValueError(f"duplicate {adaptation} seed: {seed}")
        indexed[seed] = run
    return indexed


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def _aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    overall = {
        name: _stats([float(run["metrics"][name]) for run in runs])
        for name in ("brier", "kl", "accuracy", "ece")
    }
    first = runs[0]["metrics"]["slices"]
    by_type = {
        question_type: {
            "brier": _stats(
                [
                    float(run["metrics"]["slices"]["by_type"][question_type]["brier"])
                    for run in runs
                ]
            ),
            "kl": _stats(
                [
                    float(
                        run["metrics"]["slices"]["by_type"][question_type][
                            "kl_divergence"
                        ]
                    )
                    for run in runs
                ]
            ),
        }
        for question_type in first["by_type"]
    }
    by_source = {
        source: {
            "brier": _stats(
                [
                    float(run["metrics"]["slices"]["by_source"][source]["brier"])
                    for run in runs
                ]
            ),
            "kl": _stats(
                [
                    float(
                        run["metrics"]["slices"]["by_source"][source][
                            "kl_divergence"
                        ]
                    )
                    for run in runs
                ]
            ),
        }
        for source in first["by_source"]
    }
    return {
        "seeds": [int(run["config"]["seed"]) for run in runs],
        "overall": overall,
        "by_type": by_type,
        "by_source": by_source,
    }


def build_adaptation_report(
    *,
    lora_runs: list[dict[str, Any]],
    full_runs: list[dict[str, Any]],
    lora_sweep_runs: list[dict[str, Any]],
    full_sweep_runs: list[dict[str, Any]],
) -> dict[str, Any]:
    lora_by_seed = _index_by_seed(lora_runs, adaptation="lora")
    full_by_seed = _index_by_seed(full_runs, adaptation="full_finetune")
    if set(lora_by_seed) != set(full_by_seed):
        raise ValueError("LoRA and full fine-tune seeds do not match")
    seeds = sorted(lora_by_seed)
    if len(seeds) < 3 or 17 not in seeds:
        raise ValueError("comparison requires selection seed 17 plus at least two repeats")

    all_runs = [*lora_runs, *full_runs, *lora_sweep_runs, *full_sweep_runs]
    expected_cohort = fixed_sample_cohort(all_runs[0]["config"])
    for run in all_runs[1:]:
        cohort = fixed_sample_cohort(run["config"])
        if cohort != expected_cohort:
            raise ValueError(
                f"run does not use the fixed data cohort/schedule: {run['run_dir']}"
            )
    if any(run["model_id"] != "Qwen/Qwen3-1.7B-Base" for run in all_runs):
        raise ValueError("adaptation comparison must use Qwen3-1.7B-Base")
    if any(run["attention_backend"] != "sdpa" for run in all_runs):
        raise ValueError("adaptation runs must all use SDPA")
    if {int(run["config"]["seed"]) for run in full_sweep_runs} != {17}:
        raise ValueError("full learning-rate sweep must use selection seed 17")
    if {int(run["config"]["seed"]) for run in lora_sweep_runs} != {17}:
        raise ValueError("LoRA learning-rate sweep must use selection seed 17")
    if len({float(run["learning_rate"]) for run in full_sweep_runs}) != len(
        full_sweep_runs
    ):
        raise ValueError("full learning-rate sweep contains duplicate rates")
    if len({float(run["learning_rate"]) for run in lora_sweep_runs}) != len(
        lora_sweep_runs
    ):
        raise ValueError("LoRA learning-rate sweep contains duplicate rates")

    best_lora = select_best_lora_run(lora_sweep_runs)
    best_lora_learning_rate = float(best_lora["learning_rate"])
    if {float(run["learning_rate"]) for run in lora_runs} != {best_lora_learning_rate}:
        raise ValueError("LoRA repeats do not use the selected learning rate")
    if lora_by_seed[17]["run_dir"] != best_lora["run_dir"]:
        raise ValueError("selection-seed LoRA run is not the sweep winner")
    best_full = select_best_full_run(full_sweep_runs)
    best_full_learning_rate = float(best_full["learning_rate"])
    if {float(run["learning_rate"]) for run in full_runs} != {best_full_learning_rate}:
        raise ValueError("full repeats do not use the selected learning rate")
    if full_by_seed[17]["run_dir"] != best_full["run_dir"]:
        raise ValueError("selection-seed full run is not the sweep winner")

    confirmation_seeds = [seed for seed in seeds if seed != 17]
    paired = {}
    for seed in seeds:
        lora_metrics = lora_by_seed[seed]["metrics"]
        full_metrics = full_by_seed[seed]["metrics"]
        paired[str(seed)] = {
            "lora": metric_summary(lora_metrics),
            "full_finetune": metric_summary(full_metrics),
            "full_minus_lora": {
                "brier": full_metrics["brier"] - lora_metrics["brier"],
                "kl": full_metrics["kl"] - lora_metrics["kl"],
                "accuracy": full_metrics["accuracy"] - lora_metrics["accuracy"],
                "by_type": {
                    question_type: {
                        "brier": full_metrics["slices"]["by_type"][question_type][
                            "brier"
                        ]
                        - values["brier"],
                        "kl": full_metrics["slices"]["by_type"][question_type][
                            "kl_divergence"
                        ]
                        - values["kl_divergence"],
                    }
                    for question_type, values in lora_metrics["slices"]["by_type"].items()
                },
            },
        }

    return {
        "purpose": "development-only LoRA versus full-fine-tune comparison",
        "test_splits_opened": False,
        "selection_rule": "lowest development Brier, then KL",
        "fixed_sample_cohort": expected_cohort,
        "lora_learning_rate_sweep": [
            {
                "run_dir": run["run_dir"],
                "learning_rate": run["learning_rate"],
                "brier": run["metrics"]["brier"],
                "kl": run["metrics"]["kl"],
                "accuracy": run["metrics"]["accuracy"],
            }
            for run in sorted(lora_sweep_runs, key=lambda run: run["learning_rate"])
        ],
        "full_learning_rate_sweep": [
            {
                "run_dir": run["run_dir"],
                "learning_rate": run["learning_rate"],
                "brier": run["metrics"]["brier"],
                "kl": run["metrics"]["kl"],
                "accuracy": run["metrics"]["accuracy"],
            }
            for run in sorted(full_sweep_runs, key=lambda run: run["learning_rate"])
        ],
        "selected": {
            "lora_learning_rate": best_lora_learning_rate,
            "lora_run_dir": best_lora["run_dir"],
            "full_learning_rate": best_full_learning_rate,
            "full_run_dir": best_full["run_dir"],
        },
        "paired_seeds": seeds,
        "confirmation_seeds": confirmation_seeds,
        "paired_results": paired,
        "all_seed_aggregate": {
            "lora": _aggregate([lora_by_seed[seed] for seed in seeds]),
            "full_finetune": _aggregate([full_by_seed[seed] for seed in seeds]),
        },
        "confirmation_aggregate": {
            "lora": _aggregate([lora_by_seed[seed] for seed in confirmation_seeds]),
            "full_finetune": _aggregate(
                [full_by_seed[seed] for seed in confirmation_seeds]
            ),
        },
    }


def print_adaptation_report(report: dict[str, Any]) -> None:
    print("LoRA learning-rate sweep (selection seed 17)")
    for row in report["lora_learning_rate_sweep"]:
        selected = " selected" if row["run_dir"] == report["selected"]["lora_run_dir"] else ""
        print(
            f"  lr={row['learning_rate']:.1e} brier={row['brier']:.4f} "
            f"kl={row['kl']:.4f} accuracy={row['accuracy']:.4f}{selected}"
        )
    print("full-fine-tune learning-rate sweep (selection seed 17)")
    for row in report["full_learning_rate_sweep"]:
        selected = " selected" if row["run_dir"] == report["selected"]["full_run_dir"] else ""
        print(
            f"  lr={row['learning_rate']:.1e} brier={row['brier']:.4f} "
            f"kl={row['kl']:.4f} accuracy={row['accuracy']:.4f}{selected}"
        )
    print("paired development results (full minus LoRA; positive Brier/KL favors LoRA)")
    for seed, row in report["paired_results"].items():
        delta = row["full_minus_lora"]
        print(
            f"  seed={seed} brier={delta['brier']:+.4f} kl={delta['kl']:+.4f} "
            f"accuracy={delta['accuracy']:+.4f}"
        )
    print("confirmation aggregate (training seeds 29 and 43)")
    for adaptation, aggregate in report["confirmation_aggregate"].items():
        overall = aggregate["overall"]
        print(
            f"  {adaptation}: brier={overall['brier']['mean']:.4f} "
            f"±{overall['brier']['sample_std']:.4f} "
            f"kl={overall['kl']['mean']:.4f} ±{overall['kl']['sample_std']:.4f} "
            f"accuracy={overall['accuracy']['mean']:.4f}"
        )
        for question_type, values in aggregate["by_type"].items():
            print(
                f"    {question_type}: brier={values['brier']['mean']:.4f} "
                f"kl={values['kl']['mean']:.4f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare tuned 1.7B LoRA and full fine-tuning across seeds"
    )
    parser.add_argument("--lora-run", action="append", required=True, type=Path)
    parser.add_argument("--full-run", action="append", required=True, type=Path)
    parser.add_argument("--lora-sweep-run", action="append", required=True, type=Path)
    parser.add_argument("--full-sweep-run", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = build_adaptation_report(
        lora_runs=[load_run(path) for path in args.lora_run],
        full_runs=[load_run(path) for path in args.full_run],
        lora_sweep_runs=[load_run(path) for path in args.lora_sweep_run],
        full_sweep_runs=[load_run(path) for path in args.full_sweep_run],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print_adaptation_report(report)
    print(f"adaptation comparison saved: {args.output}")


if __name__ == "__main__":
    main()
