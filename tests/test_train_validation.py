from __future__ import annotations

import json
from pathlib import Path

import pytest

from qwenrlcd.train import validation_metrics_with_reference


def test_reference_validation_reuses_subset_predictions() -> None:
    rows = [
        {
            "id": "original",
            "source": "source_a",
            "loss": 0.1,
            "questions": [{
                "type": "choice", "probabilities": [0.8, 0.2], "target": [1.0, 0.0],
            }],
        },
        {
            "id": "additional",
            "source": "source_b",
            "loss": 2.0,
            "questions": [{
                "type": "noul", "probabilities": [0.1, 0.9], "target": [1.0, 0.0],
            }],
        },
    ]
    metrics = validation_metrics_with_reference(rows, {"original"})
    reference = metrics["reference_validation"]
    assert metrics["bundles"] == 2
    assert metrics["questions"] == 2
    assert reference["bundles"] == 1
    assert reference["questions"] == 1
    assert reference["loss"] == pytest.approx(0.1)
    assert reference["brier"] == pytest.approx(0.08)
    assert reference["slices"]["by_type"].keys() == {"choice"}
    assert metrics["brier"] > reference["brier"]


def test_reference_validation_rejects_missing_bundle() -> None:
    row = {
        "id": "present", "source": "fixture", "loss": 0.0,
        "questions": [{
            "type": "noul", "probabilities": [1.0, 0.0], "target": [1.0, 0.0],
        }],
    }
    with pytest.raises(ValueError, match="reference validation bundle IDs"):
        validation_metrics_with_reference([row], {"missing"})


def test_50k_config_only_changes_sample_and_operational_fields() -> None:
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    with (config_dir / "qwen3_1_7b_core_option_10k.json").open(encoding="utf-8") as handle:
        previous = json.load(handle)
    with (config_dir / "qwen3_1_7b_core_option_50k.json").open(encoding="utf-8") as handle:
        expanded = json.load(handle)
    permitted_changes = {
        "train_bundle_limit", "validation_bundle_limit", "validation_reference_bundle_limit",
        "output_dir", "log_every", "save_every",
    }
    assert {key: value for key, value in previous.items() if key not in permitted_changes} == {
        key: value for key, value in expanded.items() if key not in permitted_changes
    }
    assert expanded["train_bundle_limit"] == 50_000
    assert expanded["validation_bundle_limit"] == 2_000
    assert expanded["validation_reference_bundle_limit"] == previous["validation_bundle_limit"]
