#!/usr/bin/env bash
# Follow-up experiments from review feedback: steering dose-response sweep (E3-v2)
# and the interpretability-metric comparison (Spearman-continuous vs F1-categorical
# vs L_struct).
set -u
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
PY=.venv/bin/python

echo "######## E3-v2 steering dose-response sweep ########"
for d in 16 0; do
  $PY experiment_steering_sweep.py --layer-dir outputs_layerwise/esm2/layer_$d \
    --top-k 8 --n-control 8 --max-proteins 50 --scales 0,0.5,1,2,4 \
    --save-dir results_steering_sweep/esm2_l$d || echo "FAIL sweep esm2 $d"
done

echo "######## interpretability-metric comparison ########"
for d in 0 16 32; do
  $PY experiment_interp_comparison.py --layer-dir outputs_layerwise/esm2/layer_$d \
    --concept-csv results_concept_f1/esm2_l$d/feature_concept_best.csv \
    --save-dir results_interp_comparison/esm2_l$d || echo "FAIL cmp esm2 $d"
done
for d in 0 12 23; do
  $PY experiment_interp_comparison.py --layer-dir outputs_layerwise/rita/layer_$d \
    --concept-csv results_concept_f1/rita_l$d/feature_concept_best.csv \
    --save-dir results_interp_comparison/rita_l$d || echo "FAIL cmp rita $d"
done
echo "######## FOLLOWUPS DONE ########"
