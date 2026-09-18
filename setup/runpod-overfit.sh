#!/usr/bin/env bash
# Prove that the complete training path can memorize two mixed-question bundles.
set -euo pipefail

QWENRLCD_SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
QWENRLCD_CONFIG="${1:-configs/qwen3_1_7b_overfit.json}"

# shellcheck source=/dev/null
source "$QWENRLCD_SETUP_DIR/runpod-activate.sh"

echo "[overfit] config=$QWENRLCD_CONFIG"
uv run qwenrlcd-train --config "$QWENRLCD_CONFIG"

QWENRLCD_OUTPUT_DIR="$(uv run python - "$QWENRLCD_CONFIG" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    print(json.load(handle)["output_dir"])
PY
)"

uv run qwenrlcd-check-parallel --run-dir "$QWENRLCD_OUTPUT_DIR"

uv run python - "$QWENRLCD_OUTPUT_DIR/validation_metrics.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    final = json.load(handle)[-1]

requirements = {
    "step": (final["step"] == 100, "must equal 100"),
    "accuracy": (final["accuracy"] >= 0.999, "must be at least 0.999"),
    "loss": (final["loss"] <= 0.25, "must be at most 0.25"),
    "nll": (final["nll"] <= 0.20, "must be at most 0.20"),
    "brier": (final["brier"] <= 0.05, "must be at most 0.05"),
    "ece": (final["ece"] <= 0.20, "must be at most 0.20"),
}
failures = [
    f"{name}={final[name]} {message}"
    for name, (passed, message) in requirements.items()
    if not passed
]

print("overfit final metrics", json.dumps(final, sort_keys=True))
if failures:
    raise SystemExit("overfit thresholds failed: " + "; ".join(failures))
print("overfit thresholds passed")
PY

echo "[overfit] passed"
echo "[overfit] metrics: $QWENRLCD_OUTPUT_DIR/validation_metrics.json"
