#!/usr/bin/env bash
# run_eval_ctrl_causal.sh — gradient-attribution computational lens across depths, both models.
set -u
cd "$(dirname "$0")"
source env_local_caches.sh 2>/dev/null || true
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=.venv/bin/python
CKDIR="$HOME/own_sae_data/uniref50_pilot"
mkdir -p results_ctrl_causal
for M in mlm clm; do
  CAUS=""; [ "$M" = clm ] && CAUS="--causal"
  for L in 0 3 6 9 11; do
    echo "=== $M L$L $(date) ==="
    $PY eval_ctrl_causal.py --ckpt "$CKDIR/ckpt_${M}/model_final.pt" \
      --layer-dir "outputs_ctrl/ctrl_${M}_A/layer_${L}" --layer "$L" $CAUS \
      --max-proteins 200 --device cpu > "results_ctrl_causal/${M}_l${L}.json" 2>>results_ctrl_causal/err.log \
      || echo "FAIL $M $L"
  done
done
echo "CTRL_CAUSAL DONE $(date)"
