#!/usr/bin/env bash
# run_multiseed_extensions_cpu.sh
# Expand single-seed CPU extension experiments to SAE seeds {43,44} at the
# HEADLINE cells (ESM-2 L16, RITA L18). Seed 42 already exists in the canonical
# results_*/ dirs; this fills the other two points of each triad.
# Idempotent (skip-if-summary-exists), FAIL-guarded, prints a DONE marker.
set -u
cd "$(dirname "$0")"
source env_local_caches.sh 2>/dev/null || true
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=.venv/bin/python

run_if_absent () {  # $1 = sentinel file, rest = command
  local sentinel="$1"; shift
  if [ -e "$sentinel" ]; then echo "SKIP (exists): $sentinel"; return 0; fi
  echo ">>> $*"
  "$@" || echo "FAIL: $*"
}

for SEED in 43 44; do
  LW="outputs_layerwise_seed${SEED}"
  for CELL in "esm2 16" "rita 18"; do
    set -- $CELL; M=$1; D=$2; C="${M}_l${D}"

    # E4 null calibration
    OUT="results_null_seed${SEED}/${C}"
    run_if_absent "$OUT/summary.json" \
      $PY experiment_null_calibration.py --layer-dir "$LW/$M/layer_$D" --save-dir "$OUT"

    # SAE diagnostics
    OUT="results_sae_diagnostics_seed${SEED}/${C}"
    run_if_absent "$OUT/diagnostics.json" \
      $PY experiment_sae_diagnostics.py --layer-dir "$LW/$M/layer_$D" --save-dir "$OUT"

    # Metric-independence / interp comparison (seed-matched concept csv)
    OUT="results_interp_comparison_seed${SEED}/${C}"
    CC="results_concept_f1_seed${SEED}/${C}/feature_concept_best.csv"
    run_if_absent "$OUT/summary.json" \
      $PY experiment_interp_comparison.py --layer-dir "$LW/$M/layer_$D" --concept-csv "$CC" --save-dir "$OUT"

    # E1 linear probe (raw vs SAE)
    OUT="results_probe_seed${SEED}/${C}"
    run_if_absent "$OUT/summary.json" \
      $PY experiment_probe_baseline.py --layer-dir "$LW/$M/layer_$D" --model "$M" --layer "$D" --save-dir "$OUT"
  done
done

echo "MULTISEED_CPU DONE $(date)"
