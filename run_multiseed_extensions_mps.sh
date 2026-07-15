#!/usr/bin/env bash
# run_multiseed_extensions_mps.sh
# Expand single-seed MPS forward-pass extension experiments to SAE seeds {43,44}
# at the ESM-2 L16 headline cell (E2/E3/faithfulness are ESM-2-only by design).
# Matches the canonical seed-42 invocations exactly. Idempotent, FAIL-guarded.
set -u
cd "$(dirname "$0")"
source env_local_caches.sh 2>/dev/null || true
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=.venv/bin/python
D=16

run_if_absent () {  # $1 = sentinel, rest = command
  local sentinel="$1"; shift
  if [ -e "$sentinel" ]; then echo "SKIP (exists): $sentinel"; return 0; fi
  echo ">>> $*"
  "$@" || echo "FAIL: $*"
}

for SEED in 43 44; do
  LW="outputs_layerwise_seed${SEED}/esm2/layer_${D}"

  # E2 per-feature causal ablation
  OUT="results_causal_seed${SEED}/esm2_l${D}"
  run_if_absent "$OUT/summary.json" \
    $PY experiment_causal_features.py --layer-dir "$LW" \
      --top-k 15 --n-control 15 --max-proteins 120 --save-dir "$OUT"

  # E3 steering dose-response sweep (seed-matched concept csv)
  OUT="results_steering_sweep_seed${SEED}/esm2_l${D}"
  CC="results_concept_f1_seed${SEED}/esm2_l${D}/feature_concept_best.csv"
  run_if_absent "$OUT/summary.json" \
    $PY experiment_steering_sweep.py --layer-dir "$LW" \
      --concept-csv "$CC" --top-k 8 --n-random 8 --max-proteins 50 \
      --scales 0,0.5,1,2,4 --n-boot 1000 --save-dir "$OUT"

  # LM faithfulness (ESM-2 MLM substitution)
  OUT="results_faithfulness_seed${SEED}/esm2_l${D}"
  run_if_absent "$OUT/summary.json" \
    $PY experiment_lm_faithfulness.py --layer-dir "$LW" \
      --layer "$D" --max-proteins 60 --n-boot 1000 --save-dir "$OUT"
done

echo "MULTISEED_MPS DONE $(date)"
