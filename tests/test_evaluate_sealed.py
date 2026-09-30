from __future__ import annotations

import json
from pathlib import Path

import pytest

from qwenrlcd.evaluate_sealed import initialize_or_resume_report


def _development_metrics() -> dict:
    slice_metrics = {
        "count": 1,
        "questions": 1,
        "brier": 0.1,
        "kl_divergence": 0.2,
    }
    return {
        "bundles": 1,
        "questions": 1,
        "accuracy": 1.0,
        "expected_accuracy": 0.9,
        "brier": 0.1,
        "uniform_brier": 0.5,
        "kl": 0.2,
        "uniform_kl": 0.7,
        "ece": 0.1,
        "slices": {
            "by_source": {"fixture": slice_metrics},
            "by_type": {"choice": slice_metrics},
        },
    }


def _make_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    final_dir = run_dir / "final"
    final_dir.mkdir(parents=True)
    (final_dir / "adapter.bin").write_bytes(b"frozen model")
    return run_dir


def test_report_freezes_artifact_before_splits(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path)
    output = tmp_path / "sealed.json"
    report = initialize_or_resume_report(
        output=output,
        run_dir=run_dir,
        config={"model_id": "fixture/model"},
        development_metrics=_development_metrics(),
        attention_backend="sdpa",
    )

    assert report["selection_frozen_before_evaluation"] is True
    assert report["splits"] == {}
    assert report["complete"] is False
    assert len(report["selected_model"]["final_artifact_sha256"]) == 64
    assert json.loads(output.read_text(encoding="utf-8")) == report


def test_report_resumes_only_same_frozen_artifact(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path)
    output = tmp_path / "sealed.json"
    kwargs = {
        "output": output,
        "run_dir": run_dir,
        "config": {"model_id": "fixture/model"},
        "development_metrics": _development_metrics(),
        "attention_backend": "sdpa",
    }
    original = initialize_or_resume_report(**kwargs)
    assert initialize_or_resume_report(**kwargs) == original

    (run_dir / "final" / "adapter.bin").write_bytes(b"changed model")
    with pytest.raises(ValueError, match="different model"):
        initialize_or_resume_report(**kwargs)
