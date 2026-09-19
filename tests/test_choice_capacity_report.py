from __future__ import annotations

from qwenrlcd.choice_capacity_report import capacity_gate_failures


def passing_metrics() -> dict:
    return {
        "step": 400,
        "bundles": 32,
        "kl": 0.001,
        "brier": 0.001,
        "slices": {
            "by_source": {
                source: {"kl_divergence": 0.001, "brier": 0.001}
                for source in ("exact_probability_experiments", "goemotions", "snli",
                               "procedural_workflows")
            }
        },
    }


def test_capacity_gate_checks_completion_and_fit() -> None:
    metrics = passing_metrics()
    assert capacity_gate_failures(metrics, expected_steps=400, has_final_model=True) == []
    assert "final model export is missing" in capacity_gate_failures(
        metrics, expected_steps=400, has_final_model=False
    )
    metrics["step"] = 380
    metrics["slices"]["by_source"]["snli"]["kl_divergence"] = 0.2
    failures = capacity_gate_failures(metrics, expected_steps=400, has_final_model=True)
    assert "step=380 != 400" in failures
    assert any("snli kl_divergence" in failure for failure in failures)
