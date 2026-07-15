#!/usr/bin/env bash
# MPS-based experiments (E1 probe, E2 causal, E3 steering, E6 viz, C1 control).
# Run sequentially so only one PLM is on the GPU at a time. Safe to run
# alongside the CPU-bound run_eval_suite.sh.
set -u
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
PY=.venv/bin/python

echo "######## E1 SAE-vs-raw probe ########"
for d in 0 16 32; do
  $PY experiment_probe_baseline.py --layer-dir outputs_layerwise/esm2/layer_$d \
    --model esm2 --layer $d --save-dir results_probe/esm2_l$d || echo "FAIL E1 esm2 $d"
done
for d in 0 12 23; do
  $PY experiment_probe_baseline.py --layer-dir outputs_layerwise/rita/layer_$d \
    --model rita --layer $d --save-dir results_probe/rita_l$d || echo "FAIL E1 rita $d"
done

echo "######## E2 causal features ########"
for d in 16 0; do
  $PY experiment_causal_features.py --layer-dir outputs_layerwise/esm2/layer_$d \
    --top-k 15 --n-control 15 --max-proteins 120 --save-dir results_causal/esm2_l$d || echo "FAIL E2 esm2 $d"
done

echo "######## E3 steering ########"
$PY experiment_steering.py --layer-dir outputs_layerwise/esm2/layer_16 \
  --top-k 8 --n-control 8 --max-proteins 60 --scales 1,2,4 \
  --save-dir results_steering/esm2_l16 || echo "FAIL E3 esm2 16"

echo "######## E6 feature viz ########"
$PY experiment_feature_viz.py --layer-dir outputs_layerwise/esm2/layer_16 \
  --top-k 6 --concept-csv results_concept_f1/esm2_l16/feature_concept_best.csv \
  --save-dir results_feature_viz/esm2_l16 || echo "FAIL E6 esm2 16"
$PY experiment_feature_viz.py --layer-dir outputs_layerwise/rita/layer_12 \
  --top-k 6 --save-dir results_feature_viz/rita_l12 || echo "FAIL E6 rita 12"

echo "######## C1 randomized-weights control (+ concept-F1 + null on control) ########"
for d in 16 0; do
  $PY experiment_random_control.py --ref-layer-dir outputs_layerwise/esm2/layer_$d \
    --model esm2 --layer $d --epochs 60 --out-layer-dir outputs_random/esm2/layer_$d || { echo "FAIL C1 esm2 $d"; continue; }
  $PY experiment_concept_f1.py --layer-dir outputs_random/esm2/layer_$d \
    --save-dir results_concept_f1_random/esm2_l$d || echo "FAIL C1-cf1 esm2 $d"
  $PY experiment_null_calibration.py --layer-dir outputs_random/esm2/layer_$d \
    --save-dir results_null_random/esm2_l$d || echo "FAIL C1-null esm2 $d"
done
echo "######## CAUSAL SUITE DONE ########"
