#!/usr/bin/env bash
# watch_multiseed_then_tier1b.sh — wait for Tier 1a, then start Tier 1b.
#
# This is intended for unattended overnight use. It waits until the active
# run_multiseed_full_grid.sh process exits, verifies that the seed43/44 full
# grid is complete, then launches the next priority task: Tier 1b C1
# randomized-weights seeds 1 and 2.
set -euo pipefail

cd "$(dirname "$0")"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

PY="${PY:-./.venv/bin/python}"
LOCK=".tier1b_after_multiseed.lock"

if ! ( set -o noclobber; echo "$$" > "${LOCK}" ) 2>/dev/null; then
  echo "Another watcher appears to be active (${LOCK} exists); exiting."
  exit 1
fi
trap 'rm -f "${LOCK}"' EXIT

echo "=========================================================================="
echo "  WATCHER start $(date)"
echo "  Waiting for run_multiseed_full_grid.sh to finish before Tier 1b"
echo "=========================================================================="

while pgrep -f "[r]un_multiseed_full_grid.sh" >/dev/null; do
  echo "  $(date): Tier 1a still running; sleeping 120s"
  sleep 120
done

echo ""
echo "=========================================================================="
echo "  Tier 1a process ended $(date); verifying full-grid outputs"
echo "=========================================================================="

"${PY}" - <<'PY'
from pathlib import Path
import sys

layers = {
    "esm2": [0, 4, 8, 12, 16, 20, 24, 28, 32],
    "rita": [0, 3, 6, 9, 12, 15, 18, 21, 23],
    "prott5_enc": [0, 3, 6, 9, 12, 15, 18, 21, 23],
    "prott5_dec": [0, 3, 6, 9, 12, 15, 18, 21, 23],
    "protgpt2": [0, 4, 9, 13, 18, 22, 27, 31, 35],
}

missing = []
for seed in (43, 44):
    root = Path(f"outputs_layerwise_seed{seed}")
    for model, layer_list in layers.items():
        for layer in layer_list:
            d = root / model / f"layer_{layer}"
            for fname in ("META.json", "struct_seq_metrics.csv", "feature_interpretability.csv"):
                if not (d / fname).exists():
                    missing.append(str(d / fname))

required_aggregate = Path("analysis_results_multiseed/cross_seed_summary.csv")
if not required_aggregate.exists():
    missing.append(str(required_aggregate))

if missing:
    print("Tier 1a verification FAILED; not launching Tier 1b.")
    print("Missing:")
    for item in missing[:80]:
        print(f"  {item}")
    if len(missing) > 80:
        print(f"  ... and {len(missing) - 80} more")
    sys.exit(1)

print("Tier 1a verification passed; launching Tier 1b.")
PY

echo ""
echo "=========================================================================="
echo "  Launching Tier 1b $(date)"
echo "=========================================================================="

./run_tier1b_random_weight_seeds.sh 2>&1 | tee run_tier1b_random_weight_seeds.log

echo ""
echo "=========================================================================="
echo "  WATCHER complete $(date)"
echo "=========================================================================="
