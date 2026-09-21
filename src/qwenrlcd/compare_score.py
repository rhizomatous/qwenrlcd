from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def compare_score_reports(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> list[dict[str, Any]]:
    baseline_groups = {"overall": baseline["overall"], **baseline["by_dimension"]}
    candidate_groups = {"overall": candidate["overall"], **candidate["by_dimension"]}
    if baseline_groups.keys() != candidate_groups.keys():
        raise ValueError("score reports contain different dimensions")

    comparisons = []
    for name, before in baseline_groups.items():
        after = candidate_groups[name]
        if before["count"] != after["count"]:
            raise ValueError(f"score report count changed for {name}")
        before_unique = before["unique_target_modes"]
        after_unique = after["unique_target_modes"]
        if before_unique != after_unique:
            raise ValueError(f"unique target-mode count changed for {name}")

        before_rps = before["model"]["ranked_probability_score"]
        after_rps = after["model"]["ranked_probability_score"]
        before_mae = before["model"]["expected_score_mae"]
        after_mae = after["model"]["expected_score_mae"]
        before_far_rate = before["far_mode_miss"] / before_unique if before_unique else 0.0
        after_far_rate = after["far_mode_miss"] / after_unique if after_unique else 0.0
        comparisons.append({
            "name": name,
            "count": before["count"],
            "baseline_rps": before_rps,
            "candidate_rps": after_rps,
            "rps_delta": after_rps - before_rps,
            "baseline_expected_score_mae": before_mae,
            "candidate_expected_score_mae": after_mae,
            "expected_score_mae_delta": after_mae - before_mae,
            "baseline_far_misses": before["far_mode_miss"],
            "candidate_far_misses": after["far_mode_miss"],
            "far_miss_rate_delta": after_far_rate - before_far_rate,
            "baseline_bias": before["model"]["expected_score_bias"],
            "candidate_bias": after["model"]["expected_score_bias"],
        })
    return comparisons


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two Score development reports")
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    args = parser.parse_args()
    with args.baseline.open(encoding="utf-8") as handle:
        baseline = json.load(handle)
    with args.candidate.open(encoding="utf-8") as handle:
        candidate = json.load(handle)
    comparisons = compare_score_reports(baseline, candidate)
    for row in comparisons:
        if row["name"] != "overall" and not row["name"].startswith("helpsteer2/"):
            continue
        print(
            f"{row['name']}: "
            f"RPS {row['baseline_rps']:.4f} -> {row['candidate_rps']:.4f} "
            f"({row['rps_delta']:+.4f}); "
            f"MAE {row['baseline_expected_score_mae']:.4f} -> "
            f"{row['candidate_expected_score_mae']:.4f} "
            f"({row['expected_score_mae_delta']:+.4f}); "
            f"far {row['baseline_far_misses']} -> {row['candidate_far_misses']}; "
            f"bias {row['baseline_bias']:+.4f} -> {row['candidate_bias']:+.4f}"
        )


if __name__ == "__main__":
    main()
