from __future__ import annotations

from copy import deepcopy

import pytest

from qwenrlcd.compare_capacity import build_report


def example_run(*, model_id: str, adaptation: str, brier: float) -> dict:
    config = {
        "model_id": model_id,
        "dataset_path": "../qwenrlcd-data",
        "dataset_config": "core",
        "train_bundle_limit": 10000,
        "validation_bundle_limit": 500,
        "seed": 17,
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
    type_metrics = {
        "count": 7,
        "brier": brier,
        "kl_divergence": brier * 2,
    }
    metrics = {
        "bundles": 500,
        "questions": 7,
        "accuracy": 0.8,
        "expected_accuracy": 0.75,
        "brier": brier,
        "uniform_brier": 0.4,
        "kl": brier * 2,
        "uniform_kl": 0.8,
        "ece": 0.02,
        "slices": {
            "by_source": {"fixture": deepcopy(type_metrics)},
            "by_type": {"choice": deepcopy(type_metrics)},
        },
    }
    return {
        "run_dir": f"outputs/{adaptation}",
        "model_id": model_id,
        "adaptation": adaptation,
        "attention_backend": "sdpa",
        "learning_rate": 2e-5,
        "head_learning_rate": 1e-3,
        "step": 1250,
        "config": config,
        "metrics": metrics,
    }


def test_capacity_report_compares_one_cohort_without_test_splits() -> None:
    control = example_run(
        model_id="Qwen/Qwen3-1.7B-Base", adaptation="lora", brier=0.10
    )
    del control["config"]["ordinal_rps_weight"]
    report = build_report(
        {
            "control": control,
            "full_finetune": example_run(
                model_id="Qwen/Qwen3-1.7B-Base",
                adaptation="full_finetune",
                brier=0.08,
            ),
            "large_lora": example_run(
                model_id="Qwen/Qwen3-4B-Base", adaptation="lora", brier=0.06
            ),
        }
    )

    assert report["test_splits_opened"] is False
    assert report["deltas_vs_control"]["full_finetune"]["brier"] == pytest.approx(
        -0.02
    )
    assert report["deltas_vs_control"]["large_lora"]["by_type"]["choice"][
        "kl"
    ] == pytest.approx(-0.08)


def test_capacity_report_rejects_different_sample() -> None:
    control = example_run(
        model_id="Qwen/Qwen3-1.7B-Base", adaptation="lora", brier=0.10
    )
    candidate = example_run(
        model_id="Qwen/Qwen3-1.7B-Base", adaptation="full_finetune", brier=0.08
    )
    candidate["config"]["seed"] = 99
    with pytest.raises(ValueError, match="cohort/schedule"):
        build_report(
            {
                "control": control,
                "full_finetune": candidate,
                "large_lora": example_run(
                    model_id="Qwen/Qwen3-4B-Base", adaptation="lora", brier=0.06
                ),
            }
        )
