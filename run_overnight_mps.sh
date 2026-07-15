#!/usr/bin/env bash
# OVERNIGHT (MPS-bound): ESM-2 forward-pass experiments across ALL layers, plus
# the ESM-2 + RITA probe across all layers. Causal/steering/faithfulness/control
# use ESM-2's MLM machinery (the principled main model for interventions).
# Idempotent: skips already-completed (experiment, layer). Heaviest (C1) is last.
set -u
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
PY=.venv/bin/python
ESM="0 4 8 12 16 20 24 28 32"
RITA="0 3 6 9 12 15 18 21 23"

w=0; while ! grep -q "METRICS SUITE DONE" metrics_suite.log 2>/dev/null; do
  sleep 60; w=$((w+1)); [ $w -ge 90 ] && { echo "[mps] wait timeout"; break; }
done
echo "[mps] starting $(date)"

echo "######## E2 causal features (ESM-2, all layers) ########"
for d in $ESM; do out=results_causal/esm2_l$d; [ -f "$out/summary.json" ] && continue
  $PY experiment_causal_features.py --layer-dir outputs_layerwise/esm2/layer_$d \
    --top-k 15 --n-control 15 --max-proteins 120 --save-dir "$out" || echo "FAIL causal $d"
done

echo "######## E3 steering sweep (ESM-2, all layers) ########"
for d in $ESM; do out=results_steering_sweep/esm2_l$d; [ -f "$out/summary.json" ] && continue
  $PY experiment_steering_sweep.py --layer-dir outputs_layerwise/esm2/layer_$d \
    --concept-csv results_concept_f1/esm2_l$d/feature_concept_best.csv \
    --top-k 8 --n-random 8 --max-proteins 50 --scales 0,0.5,1,2,4 --n-boot 1000 \
    --save-dir "$out" || echo "FAIL steer $d"
done

echo "######## E1 probe (ESM-2 + RITA, all layers) ########"
for d in $ESM; do out=results_probe/esm2_l$d; [ -f "$out/summary.json" ] && continue
  $PY experiment_probe_baseline.py --layer-dir outputs_layerwise/esm2/layer_$d --model esm2 --layer $d --save-dir "$out" || echo "FAIL probe esm2 $d"
done
for d in $RITA; do out=results_probe/rita_l$d; [ -f "$out/summary.json" ] && continue
  $PY experiment_probe_baseline.py --layer-dir outputs_layerwise/rita/layer_$d --model rita --layer $d --save-dir "$out" || echo "FAIL probe rita $d"
done

echo "######## C1 randomized-weights control (ESM-2, all layers) — heaviest, LAST ########"
for d in $ESM; do
  od=outputs_random/esm2/layer_$d
  if [ ! -f "$od/META.json" ]; then
    $PY experiment_random_control.py --ref-layer-dir outputs_layerwise/esm2/layer_$d \
      --model esm2 --layer $d --epochs 60 --out-layer-dir "$od" || { echo "FAIL c1 $d"; continue; }
  fi
  [ -f results_concept_f1_random/esm2_l$d/summary.json ] || \
    $PY experiment_concept_f1.py --layer-dir "$od" --save-dir results_concept_f1_random/esm2_l$d || echo "FAIL c1cf1 $d"
  [ -f results_null_random/esm2_l$d/summary.json ] || \
    $PY experiment_null_calibration.py --layer-dir "$od" --save-dir results_null_random/esm2_l$d || echo "FAIL c1null $d"
done
echo "[mps] OVERNIGHT MPS DONE $(date)"
