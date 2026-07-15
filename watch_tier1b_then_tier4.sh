#!/usr/bin/env bash
# watch_tier1b_then_tier4.sh — extend unattended queue after Tier 1b.
#
# Waits for watch_multiseed_then_tier1b.sh / run_tier1b_random_weight_seeds.sh
# to finish, verifies Tier 1b outputs, then launches Tier 4 fold-level CI jobs.
set -euo pipefail

cd "$(dirname "$0")"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

PY="${PY:-./.venv/bin/python}"
LOCK=".tier4_after_tier1b.lock"

if ! ( set -o noclobber; echo "$$" > "${LOCK}" ) 2>/dev/null; then
  echo "Another Tier 4 watcher appears to be active (${LOCK} exists); exiting."
  exit 1
fi
trap 'rm -f "${LOCK}"' EXIT

echo "=========================================================================="
echo "  TIER 4 WATCHER start $(date)"
echo "  Waiting for Tier 1b watcher/runner to finish"
echo "=========================================================================="

while pgrep -f "[w]atch_multiseed_then_tier1b.sh" >/dev/null || \
      pgrep -f "[r]un_tier1b_random_weight_seeds.sh" >/dev/null; do
  echo "  $(date): upstream Tier 1a/1b chain still running; sleeping 180s"
  sleep 180
done

echo ""
echo "=========================================================================="
echo "  Tier 1b chain ended $(date); verifying Tier 1b outputs"
echo "=========================================================================="

"${PY}" - <<'PY'
from pathlib import Path
import sys

missing = []
for weight_seed in (1, 2):
    layer = 16
    checks = [
        Path(f"outputs_random_weightseed{weight_seed}/esm2/layer_{layer}/META.json"),
        Path(f"results_concept_f1_random_weightseed{weight_seed}/esm2_l{layer}/summary.json"),
        Path(f"results_null_random_weightseed{weight_seed}/esm2_l{layer}/summary.json"),
    ]
    missing.extend(str(p) for p in checks if not p.exists())

if missing:
    print("Tier 1b verification FAILED; not launching Tier 4.")
    print("Missing:")
    for item in missing:
        print(f"  {item}")
    sys.exit(1)

print("Tier 1b verification passed; launching Tier 4 fold CI jobs.")
PY

echo ""
echo "=========================================================================="
echo "  Launching Tier 4 $(date)"
echo "=========================================================================="

./run_tier4_fold_cis.sh 2>&1 | tee run_tier4_fold_cis.log

echo ""
echo "=========================================================================="
echo "  TIER 4 WATCHER complete $(date)"
echo "=========================================================================="
