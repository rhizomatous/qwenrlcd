from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def capacity_gate_failures(
    metrics: dict[str, Any], *, expected_steps: int, has_final_model: bool
) -> list[str]:
    failures = []
    if metrics["step"] != expected_steps:
        failures.append(f"step={metrics['step']} != {expected_steps}")
    if metrics["bundles"] != 32:
        failures.append(f"expected 32 bundles, found {metrics['bundles']}")
    if not has_final_model:
        failures.append("final model export is missing")
    for name, limit in (("kl", 0.05), ("brier", 0.02)):
        if metrics[name] > limit:
            failures.append(f"overall {name}={metrics[name]:.4f} > {limit:.2f}")
    by_source = metrics["slices"]["by_source"]
    if len(by_source) != 4:
        failures.append(f"expected four sources, found {len(by_source)}")
    for source, source_metrics in by_source.items():
        for name, limit in (("kl_divergence", 0.10), ("brier", 0.04)):
            if source_metrics[name] > limit:
                failures.append(f"{source} {name}={source_metrics[name]:.4f} > {limit:.2f}")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Report the fixed-order Choice fit gate")
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    with (args.run_dir / "training_config.json").open(encoding="utf-8") as handle:
        config = json.load(handle)
    with (args.run_dir / "validation_metrics.json").open(encoding="utf-8") as handle:
        history = json.load(handle)
    if not history:
        raise SystemExit("[capacity] no validation metrics found")

    for metrics in history:
        print(
            "[capacity] fit",
            f"epoch={metrics['epoch']}",
            f"step={metrics['step']}",
            f"accuracy={metrics['accuracy']:.4f}",
            f"brier={metrics['brier']:.6f}",
            f"kl={metrics['kl']:.6f}",
        )
    final = history[-1]
    print("[capacity] final by source")
    for source, metrics in final["slices"]["by_source"].items():
        print(
            f"  {source}: questions={metrics['questions']} "
            f"brier={metrics['brier']:.6f} kl={metrics['kl_divergence']:.6f}"
        )
    batches = math.ceil(32 / int(config["micro_batch_size"]))
    steps_per_epoch = math.ceil(batches / int(config["gradient_accumulation_steps"]))
    failures = capacity_gate_failures(
        final,
        expected_steps=steps_per_epoch * int(config["epochs"]),
        has_final_model=(args.run_dir / "final" / "decision_head.pt").is_file(),
    )
    if failures:
        raise SystemExit("[capacity] fit gate failed: " + "; ".join(failures))
    print("[capacity] fit gate passed; these are fit metrics, not holdout validation")


if __name__ == "__main__":
    main()
