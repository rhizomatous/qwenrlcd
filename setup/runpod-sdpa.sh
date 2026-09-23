#!/usr/bin/env bash
# Prove SDPA correctness, then compare it with eager attention on the saved full run.
set -euo pipefail

QWENRLCD_SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
QWENRLCD_RUN_DIR="${1:-outputs/qwen3-1.7b-core-option-full-v0}"
QWENRLCD_EAGER_REPORT="$QWENRLCD_RUN_DIR/parallel_speed_benchmark_eager.json"
QWENRLCD_SDPA_REPORT="$QWENRLCD_RUN_DIR/parallel_speed_benchmark_sdpa.json"
QWENRLCD_COMPARISON="$QWENRLCD_RUN_DIR/attention_backend_comparison.json"

# shellcheck source=/dev/null
source "$QWENRLCD_SETUP_DIR/runpod-activate.sh"
test -f "$QWENRLCD_RUN_DIR/final/decision_head.pt"
uv sync --extra train --extra dev
uv run pytest -q tests/test_data.py tests/test_model.py tests/test_benchmark.py

uv run qwenrlcd-check-parallel \
  --run-dir "$QWENRLCD_RUN_DIR" \
  --attn-implementation sdpa \
  --compare-attn-implementation eager

uv run qwenrlcd-benchmark \
  --run-dir "$QWENRLCD_RUN_DIR" \
  --attn-implementation eager \
  --output "$QWENRLCD_EAGER_REPORT"
uv run qwenrlcd-benchmark \
  --run-dir "$QWENRLCD_RUN_DIR" \
  --attn-implementation sdpa \
  --output "$QWENRLCD_SDPA_REPORT"

uv run python - \
  "$QWENRLCD_EAGER_REPORT" \
  "$QWENRLCD_SDPA_REPORT" \
  "$QWENRLCD_COMPARISON" <<'PY'
import json
import sys
from pathlib import Path

reference_path, candidate_path, output_path = map(Path, sys.argv[1:])
with reference_path.open(encoding="utf-8") as handle:
    reference = json.load(handle)
with candidate_path.open(encoding="utf-8") as handle:
    candidate = json.load(handle)

if reference["attn_implementation"] != "eager":
    raise SystemExit("reference report is not eager attention")
if candidate["attn_implementation"] != "sdpa":
    raise SystemExit("candidate report is not SDPA")

def keyed(report):
    return {
        (case["actual_state_tokens"], case["questions"]): case
        for case in report["cases"]
    }

reference_cases = keyed(reference)
candidate_cases = keyed(candidate)
if reference_cases.keys() != candidate_cases.keys():
    raise SystemExit("attention reports contain different benchmark cases")

comparisons = []
for identity in sorted(reference_cases):
    eager = reference_cases[identity]
    sdpa = candidate_cases[identity]
    eager_mode = eager["modes"]["bundled"]
    sdpa_mode = sdpa["modes"]["bundled"]
    comparison = {
        "state_tokens": identity[0],
        "questions": identity[1],
        "eager_p50_ms": eager_mode["p50_ms"],
        "sdpa_p50_ms": sdpa_mode["p50_ms"],
        "eager_over_sdpa_speedup": eager_mode["p50_ms"] / sdpa_mode["p50_ms"],
        "eager_peak_extra_mib": eager_mode["peak_above_loaded_model_mib"],
        "sdpa_peak_extra_mib": sdpa_mode["peak_above_loaded_model_mib"],
        "sdpa_peak_extra_mib_delta": (
            sdpa_mode["peak_above_loaded_model_mib"]
            - eager_mode["peak_above_loaded_model_mib"]
        ),
    }
    comparisons.append(comparison)
    print(
        f"state={identity[0]} questions={identity[1]} "
        f"eager={comparison['eager_p50_ms']:.1f}ms "
        f"sdpa={comparison['sdpa_p50_ms']:.1f}ms "
        f"speedup={comparison['eager_over_sdpa_speedup']:.2f}x "
        f"sdpa_peak_delta={comparison['sdpa_peak_extra_mib_delta']:+.1f}MiB"
    )

output_path.write_text(
    json.dumps(
        {
            "reference": str(reference_path),
            "candidate": str(candidate_path),
            "gpu": candidate["gpu"],
            "torch": candidate["torch"],
            "cases": comparisons,
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
print(f"attention comparison saved: {output_path}")
PY
