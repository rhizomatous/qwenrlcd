from __future__ import annotations

from copy import deepcopy

import pytest

from qwenrlcd.compare_adaptation import (
    build_adaptation_report,
    select_best_full_run,
    select_best_lora_run,
)


def example_run(
    *, adaptation: str, seed: int, learning_rate: float, brier: float, run_dir: str
) -> dict:
    config = {
        "model_id": "Qwen/Qwen3-1.7B-Base",
        "adaptation": adaptation,
        "dataset_path": "../qwenrlcd-data",
        "dataset_config": "core",
        "train_bundle_limit": 10000,
        "validation_bundle_limit": 500,
        "seed": seed,
        "dataset_seed": 17,
        "max_length": 4096,
        "max_questions": 32,
        "max_choices": 255,
        "require_no_state_truncation": True,
        "permute_training": True,
        "epochs": 1,
        "micro_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "head_learning_rate": 0.001,
        "weight_decay": 0.01,
        "warmup_ratio": 0.05,
        "max_grad_norm": 1.0,
        "gradient_checkpointing": True,
        "ce_weight": 1.0,
        "brier_weight": 0.25,
        "ordinal_rps_weight": 0.0,
    }
    slice_metrics = {
        "count": 8,
        "questions": 8,
        "brier": brier,
        "kl_divergence": brier * 2,
    }
    metrics = {
        "bundles": 4,
        "questions": 8,
        "accuracy": 1 - brier,
        "expected_accuracy": 1 - brier,
        "brier": brier,
        "uniform_brier": 0.4,
        "kl": brier * 2,
        "uniform_kl": 0.8,
        "ece": brier / 10,
        "slices": {
            "by_type": {"choice": deepcopy(slice_metrics)},
            "by_source": {"fixture": deepcopy(slice_metrics)},
        },
    }
    return {
        "run_dir": run_dir,
        "model_id": config["model_id"],
        "adaptation": adaptation,
        "attention_backend": "sdpa",
        "learning_rate": learning_rate,
        "head_learning_rate": 0.001,
        "step": 1250,
        "config": config,
        "metrics": metrics,
    }


def test_adaptation_report_selects_lr_then_compares_fixed_data_seeds() -> None:
    full_seed17 = example_run(
        adaptation="full_finetune", seed=17, learning_rate=1e-5,
        brier=0.08, run_dir="full-best-17",
    )
    sweep = [
        example_run(
            adaptation="full_finetune", seed=17, learning_rate=5e-6,
            brier=0.09, run_dir="full-5e-6-17",
        ),
        full_seed17,
        example_run(
            adaptation="full_finetune", seed=17, learning_rate=2e-5,
            brier=0.10, run_dir="full-2e-5-17",
        ),
    ]
    lora_runs = [
        example_run(
            adaptation="lora", seed=seed, learning_rate=2e-4,
            brier=brier, run_dir=f"lora-{seed}",
        )
        for seed, brier in ((17, 0.07), (29, 0.06), (43, 0.08))
    ]
    lora_sweep = [
        example_run(
            adaptation="lora", seed=17, learning_rate=1e-4,
            brier=0.08, run_dir="lora-1e-4-17",
        ),
        lora_runs[0],
        example_run(
            adaptation="lora", seed=17, learning_rate=4e-4,
            brier=0.09, run_dir="lora-4e-4-17",
        ),
    ]
    full_runs = [
        full_seed17,
        example_run(
            adaptation="full_finetune", seed=29, learning_rate=1e-5,
            brier=0.07, run_dir="full-best-29",
        ),
        example_run(
            adaptation="full_finetune", seed=43, learning_rate=1e-5,
            brier=0.09, run_dir="full-best-43",
        ),
    ]

    report = build_adaptation_report(
        lora_runs=lora_runs,
        full_runs=full_runs,
        lora_sweep_runs=lora_sweep,
        full_sweep_runs=sweep,
    )

    assert report["test_splits_opened"] is False
    assert report["selected"]["full_learning_rate"] == pytest.approx(1e-5)
    assert report["selected"]["lora_learning_rate"] == pytest.approx(2e-4)
    assert report["confirmation_seeds"] == [29, 43]
    assert report["paired_results"]["29"]["full_minus_lora"]["brier"] == pytest.approx(
        0.01
    )
    assert report["confirmation_aggregate"]["lora"]["overall"]["brier"][
        "mean"
    ] == pytest.approx(0.07)


def test_adaptation_report_rejects_seed_dependent_dataset_sample() -> None:
    lora_runs = [
        example_run(
            adaptation="lora", seed=seed, learning_rate=2e-4,
            brier=0.07, run_dir=f"lora-{seed}",
        )
        for seed in (17, 29, 43)
    ]
    full_runs = [
        example_run(
            adaptation="full_finetune", seed=seed, learning_rate=1e-5,
            brier=0.08, run_dir=f"full-{seed}",
        )
        for seed in (17, 29, 43)
    ]
    del lora_runs[1]["config"]["dataset_seed"]
    with pytest.raises(ValueError, match="fixed data cohort"):
        build_adaptation_report(
            lora_runs=lora_runs,
            full_runs=full_runs,
            lora_sweep_runs=[lora_runs[0]],
            full_sweep_runs=[full_runs[0]],
        )


def test_full_selection_uses_kl_as_brier_tiebreaker() -> None:
    first = example_run(
        adaptation="full_finetune", seed=17, learning_rate=1e-5,
        brier=0.08, run_dir="first",
    )
    second = example_run(
        adaptation="full_finetune", seed=17, learning_rate=2e-5,
        brier=0.08, run_dir="second",
    )
    second["metrics"]["kl"] = first["metrics"]["kl"] - 0.01
    assert select_best_full_run([first, second]) is second


def test_lora_selection_uses_kl_as_brier_tiebreaker() -> None:
    first = example_run(
        adaptation="lora", seed=17, learning_rate=1e-4,
        brier=0.08, run_dir="first",
    )
    second = example_run(
        adaptation="lora", seed=17, learning_rate=2e-4,
        brier=0.08, run_dir="second",
    )
    second["metrics"]["kl"] = first["metrics"]["kl"] - 0.01
    assert select_best_lora_run([first, second]) is second
