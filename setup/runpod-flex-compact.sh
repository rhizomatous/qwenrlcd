#!/usr/bin/env bash
# Run compact-topology correctness, block-size selection, and matched calibrations.
set -euo pipefail

QWENRLCD_SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
QWENRLCD_RUN_DIR="${1:-outputs/qwen3-1.7b-core-option-full-v0}"
QWENRLCD_EXPERIMENT_DIR="${QWENRLCD_EXPERIMENT_DIR:-outputs/qwen3-1.7b-flex-compact-experiment-v0}"
QWENRLCD_CALIBRATION_STEPS="${QWENRLCD_CALIBRATION_STEPS:-250}"
QWENRLCD_WARMUPS="${QWENRLCD_WARMUPS:-1}"
QWENRLCD_REPEATS="${QWENRLCD_REPEATS:-3}"
QWENRLCD_SELECTION_REPORT="$QWENRLCD_RUN_DIR/sdpa_training_probability_benchmark.json"
if [ ! -f "$QWENRLCD_SELECTION_REPORT" ]; then
  QWENRLCD_SELECTION_REPORT="$QWENRLCD_RUN_DIR/sdpa_training_benchmark.json"
fi
QWENRLCD_SELECTION_ARGS=()
if [ -f "$QWENRLCD_SELECTION_REPORT" ]; then
  QWENRLCD_SELECTION_ARGS=(--selection-report "$QWENRLCD_SELECTION_REPORT")
fi

# shellcheck source=/dev/null
source "$QWENRLCD_SETUP_DIR/runpod-activate.sh"
test -f "$QWENRLCD_RUN_DIR/final/decision_head.pt"
test -f "$QWENRLCD_RUN_DIR/training_config.json"
mkdir -p "$QWENRLCD_EXPERIMENT_DIR"
uv sync --extra train --extra dev

echo "[compact-flex] stage 1/4: exact compact-topology tests"
uv run pytest -q \
  tests/test_data.py \
  tests/test_model.py \
  tests/test_benchmark_training.py

for QWENRLCD_BLOCK_SIZE in 64 128; do
  QWENRLCD_TOPOLOGY_MARKER="$QWENRLCD_EXPERIMENT_DIR/topology-${QWENRLCD_BLOCK_SIZE}.passed"
  QWENRLCD_TOPOLOGY_LOG="$QWENRLCD_EXPERIMENT_DIR/topology-${QWENRLCD_BLOCK_SIZE}.log"
  if [ -f "$QWENRLCD_TOPOLOGY_MARKER" ]; then
    echo "[compact-flex] topology block_size=$QWENRLCD_BLOCK_SIZE already passed"
  else
    uv run qwenrlcd-check-parallel \
      --run-dir "$QWENRLCD_RUN_DIR" \
      --attn-implementation flex_attention \
      --compare-attn-implementation sdpa \
      --flex-block-size "$QWENRLCD_BLOCK_SIZE" \
      2>&1 | tee "$QWENRLCD_TOPOLOGY_LOG"
    touch "$QWENRLCD_TOPOLOGY_MARKER"
  fi
done

echo "[compact-flex] stage 2/4: compare 64- and 128-token Flex blocks"
for QWENRLCD_BLOCK_SIZE in 64 128; do
  QWENRLCD_BLOCK_REPORT="$QWENRLCD_EXPERIMENT_DIR/block-${QWENRLCD_BLOCK_SIZE}-tiled.json"
  if [ -f "$QWENRLCD_BLOCK_REPORT" ]; then
    echo "[compact-flex] reusing $QWENRLCD_BLOCK_REPORT"
  else
    uv run qwenrlcd-benchmark-training \
      --run-dir "$QWENRLCD_RUN_DIR" \
      --reference-backend sdpa \
      --candidate-backend flex_attention \
      --flex-block-size "$QWENRLCD_BLOCK_SIZE" \
      --warmups "$QWENRLCD_WARMUPS" \
      --repeats "$QWENRLCD_REPEATS" \
      --allow-performance-failure \
      --output "$QWENRLCD_BLOCK_REPORT" \
      "${QWENRLCD_SELECTION_ARGS[@]}"
  fi
  if [ "${#QWENRLCD_SELECTION_ARGS[@]}" -eq 0 ]; then
    QWENRLCD_SELECTION_ARGS=(--selection-report "$QWENRLCD_BLOCK_REPORT")
  fi
done

QWENRLCD_WINNER="$(uv run python - \
  "$QWENRLCD_EXPERIMENT_DIR/block-64-tiled.json" \
  "$QWENRLCD_EXPERIMENT_DIR/block-128-tiled.json" \
  "$QWENRLCD_EXPERIMENT_DIR/block-selection.json" <<'PY'
import json
import math
import sys
from pathlib import Path

reports = []
for path_text in sys.argv[1:3]:
    path = Path(path_text)
    with path.open(encoding="utf-8") as handle:
        report = json.load(handle)
    if not report["equivalence_passed"]:
        raise SystemExit(f"correctness failed for {path}")
    candidate_p50 = [
        float(report["backends"]["flex_attention"]["cases"][name]["p50_ms"])
        for name in ("median", "p95", "maximum")
    ]
    reports.append({
        "block_size": int(report["flex_block_size"]),
        "geometric_mean_speedup": float(report["geometric_mean_speedup"]),
        "geometric_mean_flex_p50_ms": math.prod(candidate_p50) ** (1 / len(candidate_p50)),
        "performance_passed": bool(report["performance_passed"]),
        "report": str(path),
        "cases": [
            {
                "name": case["name"],
                "speedup": case["reference_over_candidate_speedup"],
                "peak_mib_delta": case["candidate_peak_extra_mib_delta"],
            }
            for case in report["comparisons"]
        ],
    })
winner = min(reports, key=lambda item: item["geometric_mean_flex_p50_ms"])
selection = {
    "criterion": "smallest geometric mean Flex p50 latency across median, p95, and maximum real batches",
    "candidates": reports,
    "winner": winner["block_size"],
}
Path(sys.argv[3]).write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
print(winner["block_size"])
PY
)"
echo "[compact-flex] selected block_size=$QWENRLCD_WINNER"

echo "[compact-flex] stage 3/4: generate matched calibration configs"
uv run python - \
  "$QWENRLCD_RUN_DIR/training_config.json" \
  "$QWENRLCD_EXPERIMENT_DIR" \
  "$QWENRLCD_WINNER" <<'PY'
import json
import re
import sys
from pathlib import Path

source_path = Path(sys.argv[1])
experiment_dir = Path(sys.argv[2])
block_size = int(sys.argv[3])
with source_path.open(encoding="utf-8") as handle:
    source = json.load(handle)
for backend in ("sdpa", "flex_attention"):
    name = "flex" if backend == "flex_attention" else backend
    config = {
        **source,
        "output_dir": str(experiment_dir / f"calibration-{name}"),
        "attn_implementation": backend,
        "compact_attention_topology": True,
        "flex_block_size": block_size,
        "log_every": 25,
        "save_every": 1000000,
        "keep_last_checkpoints": 1,
    }
    path = experiment_dir / f"calibration-{name}.json"
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
PY

run_calibration() {
  local name="$1"
  local config="$QWENRLCD_EXPERIMENT_DIR/calibration-${name}.json"
  local output="$QWENRLCD_EXPERIMENT_DIR/calibration-${name}"
  local timing="$output/training_timing.json"
  if [ -f "$timing" ] \
    && [ -f "$output/checkpoint-$QWENRLCD_CALIBRATION_STEPS/trainer_state.json" ] \
    && uv run python - "$timing" "$QWENRLCD_CALIBRATION_STEPS" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    timing = json.load(handle)
raise SystemExit(0 if int(timing["current_step"]) >= int(sys.argv[2]) else 1)
PY
  then
    echo "[compact-flex] calibration-$name already reached step $QWENRLCD_CALIBRATION_STEPS"
    return
  fi

  local args=(--config "$config" --stop-after-step "$QWENRLCD_CALIBRATION_STEPS")
  if [ -d "$output" ]; then
    if compgen -G "$output/checkpoint-*" >/dev/null; then
      args+=(--resume-from latest)
    else
      local failed_output="${output}.failed-$(date -u +%Y%m%dT%H%M%SZ)"
      echo "[compact-flex] archiving non-resumable $output as $failed_output"
      mv "$output" "$failed_output"
    fi
  fi
  uv run qwenrlcd-train "${args[@]}" 2>&1 \
    | tee -a "$QWENRLCD_EXPERIMENT_DIR/calibration-${name}.log"
}

echo "[compact-flex] stage 4/4: matched $QWENRLCD_CALIBRATION_STEPS-step calibrations"
run_calibration sdpa
run_calibration flex

uv run python - \
  "$QWENRLCD_EXPERIMENT_DIR" \
  "$QWENRLCD_CALIBRATION_STEPS" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
steps = int(sys.argv[2])
with (root / "block-selection.json").open(encoding="utf-8") as handle:
    selection = json.load(handle)
timings = {}
tail_timings = {}
for name in ("sdpa", "flex"):
    with (root / f"calibration-{name}" / "training_timing.json").open(encoding="utf-8") as handle:
        timings[name] = json.load(handle)
    if int(timings[name]["current_step"]) < steps:
        raise SystemExit(f"calibration-{name} did not reach step {steps}")
    snapshots = {}
    pattern = re.compile(r"step=(\d+)/\d+.*step_s=([0-9.]+)")
    for line in (root / f"calibration-{name}.log").read_text(encoding="utf-8").splitlines():
        if match := pattern.search(line):
            snapshots[int(match.group(1))] = float(match.group(2))
    end_step = max(step for step in snapshots if step <= steps)
    start_step = max(step for step in snapshots if step <= end_step // 2)
    end_elapsed = end_step * snapshots[end_step]
    start_elapsed = start_step * snapshots[start_step]
    tail_timings[name] = {
        "start_step": start_step,
        "end_step": end_step,
        "seconds_per_optimizer_step": (
            (end_elapsed - start_elapsed) / (end_step - start_step)
        ),
    }
sdpa_step = float(timings["sdpa"]["seconds_per_optimizer_step"])
flex_step = float(timings["flex"]["seconds_per_optimizer_step"])
sdpa_tail = tail_timings["sdpa"]["seconds_per_optimizer_step"]
flex_tail = tail_timings["flex"]["seconds_per_optimizer_step"]
recommended_backend = "flex_attention" if flex_tail < 0.97 * sdpa_tail else "sdpa"
report = {
    "calibration_steps": steps,
    "selected_flex_block_size": selection["winner"],
    "sdpa_seconds_per_optimizer_step": sdpa_step,
    "flex_seconds_per_optimizer_step": flex_step,
    "sdpa_over_flex_speedup": sdpa_step / flex_step,
    "tail_window": tail_timings,
    "tail_sdpa_over_flex_speedup": sdpa_tail / flex_tail,
    "recommended_training_backend": recommended_backend,
    "recommendation_rule": (
        "Prefer Flex only when its second-half marginal step time is at least 3% "
        "faster; otherwise prefer simpler, lower-startup SDPA"
    ),
    "projected_full_training_hours": {
        "sdpa": sdpa_step * int(timings["sdpa"]["total_steps"]) / 3600,
        "flex": flex_step * int(timings["flex"]["total_steps"]) / 3600,
    },
    "block_selection": selection,
}
(root / "final-report.json").write_text(
    json.dumps(report, indent=2) + "\n", encoding="utf-8"
)
print(
    "[compact-flex] complete",
    f"block_size={report['selected_flex_block_size']}",
    f"sdpa_step_s={sdpa_step:.3f}",
    f"flex_step_s={flex_step:.3f}",
    f"speedup={report['sdpa_over_flex_speedup']:.2f}x",
    f"tail_speedup={report['tail_sdpa_over_flex_speedup']:.2f}x",
    f"recommended={recommended_backend}",
)
print(f"[compact-flex] report: {root / 'final-report.json'}")
PY
