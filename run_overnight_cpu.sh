#!/usr/bin/env bash
# OVERNIGHT (CPU-bound): model-agnostic residue experiments across ALL residue
# models and ALL layers, plus token-agnostic diagnostics for protgpt2.
# Idempotent: skips any (experiment, model, layer) whose output already exists.
set -u
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=.venv/bin/python
RESIDUE="esm2 rita prott5_enc prott5_dec"

layers_for(){ case "$1" in
  esm2) echo "0 4 8 12 16 20 24 28 32";;
  rita|prott5_enc|prott5_dec) echo "0 3 6 9 12 15 18 21 23";;
  protgpt2) echo "0 4 9 13 18 22 27 31 35";;
esac; }

# wait (up to 90 min) for the currently-running metrics suite to finish
w=0; while ! grep -q "METRICS SUITE DONE" metrics_suite.log 2>/dev/null; do
  sleep 60; w=$((w+1)); [ $w -ge 90 ] && { echo "[cpu] wait timeout"; break; }
done
echo "[cpu] starting $(date)"

echo "######## diagnostics (all residue models + protgpt2) ########"
for m in $RESIDUE protgpt2; do for d in $(layers_for $m); do
  out=results_sae_diagnostics/${m}_l${d}; [ -f "$out/diagnostics.json" ] && continue
  $PY experiment_sae_diagnostics.py --layer-dir outputs_layerwise/$m/layer_$d --save-dir "$out" || echo "FAIL diag $m $d"
done; done

echo "######## E0 concept-F1 (residue models) ########"
for m in $RESIDUE; do for d in $(layers_for $m); do
  out=results_concept_f1/${m}_l${d}; [ -f "$out/summary.json" ] && continue
  $PY experiment_concept_f1.py --layer-dir outputs_layerwise/$m/layer_$d --save-dir "$out" || echo "FAIL cf1 $m $d"
done; done

echo "######## E4 null calibration (residue models) ########"
for m in $RESIDUE; do for d in $(layers_for $m); do
  out=results_null/${m}_l${d}; [ -f "$out/summary.json" ] && continue
  $PY experiment_null_calibration.py --layer-dir outputs_layerwise/$m/layer_$d --save-dir "$out" || echo "FAIL null $m $d"
done; done

echo "######## interp comparison (residue models) ########"
for m in $RESIDUE; do for d in $(layers_for $m); do
  out=results_interp_comparison/${m}_l${d}; [ -f "$out/summary.json" ] && continue
  cc=results_concept_f1/${m}_l${d}/feature_concept_best.csv
  $PY experiment_interp_comparison.py --layer-dir outputs_layerwise/$m/layer_$d --concept-csv "$cc" --save-dir "$out" || echo "FAIL interp $m $d"
done; done

echo "######## E6 feature viz (mid-depth per model) ########"
for pair in "prott5_enc 18" "prott5_dec 9" "rita 12" "esm2 16"; do
  set -- $pair; m=$1; d=$2; out=results_feature_viz/${m}_l${d}; [ -d "$out" ] && continue
  cc=results_concept_f1/${m}_l${d}/feature_concept_best.csv
  $PY experiment_feature_viz.py --layer-dir outputs_layerwise/$m/layer_$d --concept-csv "$cc" --top-k 6 --save-dir "$out" || echo "FAIL viz $m $d"
done
echo "[cpu] OVERNIGHT CPU DONE $(date)"
