#!/usr/bin/env bash
# run_eval_ctrlA_depths.sh — full-A (660M) depth profile: blocks 0/3/9/11 (6 already done).
set -u
cd "$(dirname "$0")"
source env_local_caches.sh 2>/dev/null || true
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=.venv/bin/python
CKDIR="$HOME/own_sae_data/uniref50_pilot"
LAYERS="0 3 9 11"

for M in mlm clm; do
  NAME="ctrl_${M}_A"; CK="$CKDIR/ckpt_${M}/model_final.pt"
  for L in $LAYERS; do
    echo "===== $NAME L$L $(date) ====="
    $PY eval_ctrl_plm.py --ckpt "$CK" --name "$NAME" --layer "$L" \
        --out-root outputs_ctrl --device cpu --sae-device cpu --sae-epochs 60 \
        || { echo "FAIL eval $NAME $L"; continue; }
    $PY cpu_stage.py --layer-dir "outputs_ctrl/$NAME/layer_$L" --model-type residue \
        --pdb-dir cache/pdb_files --fasta-path cache/scope_40.fa --n-shuffles 3 \
        || echo "FAIL cpu_stage $NAME $L"
    $PY experiment_concept_f1.py --layer-dir "outputs_ctrl/$NAME/layer_$L" \
        --save-dir "results_concept_f1_ctrl/${NAME}_l${L}" --fasta-path cache/scope_40.fa \
        || echo "FAIL conceptf1 $NAME $L"
  done
done
echo "EVAL_CTRL_A_DEPTHS DONE $(date)"
