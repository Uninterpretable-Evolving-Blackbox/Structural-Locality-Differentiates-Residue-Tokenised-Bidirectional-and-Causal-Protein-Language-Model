#!/usr/bin/env bash
# Structural diagnostics (all layers, both models) + LM faithfulness (ESM-2 depths).
set -u
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
PY=.venv/bin/python
ESM="0 4 8 12 16 20 24 28 32"
RITA="0 3 6 9 12 15 18 21 23"

echo "######## SAE structural diagnostics ########"
for d in $ESM; do
  $PY experiment_sae_diagnostics.py --layer-dir outputs_layerwise/esm2/layer_$d \
    --save-dir results_sae_diagnostics/esm2_l$d || echo "FAIL diag esm2 $d"
done
for d in $RITA; do
  $PY experiment_sae_diagnostics.py --layer-dir outputs_layerwise/rita/layer_$d \
    --save-dir results_sae_diagnostics/rita_l$d || echo "FAIL diag rita $d"
done

echo "######## LM faithfulness (ESM-2 MLM, depth profile) ########"
for d in $ESM; do
  $PY experiment_lm_faithfulness.py --layer-dir outputs_layerwise/esm2/layer_$d \
    --layer $d --max-proteins 60 --n-boot 1000 \
    --save-dir results_faithfulness/esm2_l$d || echo "FAIL faith esm2 $d"
done
echo "######## METRICS SUITE DONE ########"
