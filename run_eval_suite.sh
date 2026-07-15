#!/usr/bin/env bash
# Runs the already-built CPU experiments (E0 concept-F1, E4 null calibration)
# across all matched ESM-2 and RITA depths. CPU-bound; safe to run alongside
# MPS extraction jobs.
set -u
PY=.venv/bin/python
ESM_LAYERS="0 4 8 12 16 20 24 28 32"
RITA_LAYERS="0 3 6 9 12 15 18 21 23"

echo "=== E0 concept-F1: ESM-2 ==="
for d in $ESM_LAYERS; do
  echo "--- esm2 layer $d ---"
  $PY experiment_concept_f1.py --layer-dir outputs_layerwise/esm2/layer_$d \
    --save-dir results_concept_f1/esm2_l$d || echo "FAILED esm2 $d"
done
echo "=== E0 concept-F1: RITA ==="
for d in $RITA_LAYERS; do
  echo "--- rita layer $d ---"
  $PY experiment_concept_f1.py --layer-dir outputs_layerwise/rita/layer_$d \
    --save-dir results_concept_f1/rita_l$d || echo "FAILED rita $d"
done

echo "=== E4 null calibration: ESM-2 ==="
for d in $ESM_LAYERS; do
  echo "--- esm2 layer $d ---"
  $PY experiment_null_calibration.py --layer-dir outputs_layerwise/esm2/layer_$d \
    --save-dir results_null/esm2_l$d || echo "FAILED esm2 $d"
done
echo "=== E4 null calibration: RITA ==="
for d in $RITA_LAYERS; do
  echo "--- rita layer $d ---"
  $PY experiment_null_calibration.py --layer-dir outputs_layerwise/rita/layer_$d \
    --save-dir results_null/rita_l$d || echo "FAILED rita $d"
done
echo "=== EVAL SUITE DONE ==="
