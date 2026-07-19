#!/usr/bin/env bash
# run_lstruct_predictor.sh — convergent-validity test: does L_struct SELECT features
# that predict long-range contacts on HELD-OUT proteins?
#
# 3 SAE seeds {42,43,44} (project convention; contacts are deterministic so seeds
# apply only to the SAE/L_struct side) x 2 headline cells (ESM-2 L16, RITA L18).
# Idempotent: skips a cell whose summary.json already exists.
# The expensive step (train-only L_struct) caches to struct_seq_metrics_train.csv
# inside each layer dir, so a re-run is cheap.
set -uo pipefail
cd "$(dirname "$0")"
source env_local_caches.sh 2>/dev/null || true
PY=.venv/bin/python
OUT=results_lstruct_predictor
mkdir -p "$OUT"

run () {  # $1=layer-dir  $2=tag
  if [ -f "$OUT/$2/summary.json" ]; then
    echo "[skip] $2 (summary.json exists)"; return 0
  fi
  if [ ! -f "$1/Z.npy" ]; then
    echo "[MISS] $2 — no Z.npy at $1"; return 0
  fi
  echo "=============== $2 ==============="
  $PY -u eval_lstruct_predictor.py --layer-dir "$1" --out "$OUT/$2" \
      --ks 64,256,1024 --n-random 10 --n-shuffles 5 --boot 1000 \
      || echo "[FAIL] $2 (continuing)"
}

for seed in 42 43 44; do
  case $seed in
    42) base="outputs_layerwise" ;;
    *)  base="outputs_layerwise_seed${seed}" ;;
  esac
  run "$base/esm2/layer_16" "esm2_l16_seed${seed}"
  run "$base/rita/layer_18" "rita_l18_seed${seed}"
done

echo
echo "=============== DONE ==============="
ls -1 "$OUT"
