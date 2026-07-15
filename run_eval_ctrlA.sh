#!/usr/bin/env bash
# run_eval_ctrlA.sh — full-A (660M) L_struct + concept-F1 on the FINAL controlled
# MLM vs CLM checkpoints. CPU-only. Writes to *_A names so it does NOT clobber the
# 300M (B) results.
set -u
cd "$(dirname "$0")"
source env_local_caches.sh 2>/dev/null || true
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=.venv/bin/python
CKDIR="$HOME/own_sae_data/uniref50_pilot"
LAYERS="6"

for M in mlm clm; do
  NAME="ctrl_${M}_A"
  CK="$CKDIR/ckpt_${M}/model_final.pt"
  for L in $LAYERS; do
    echo "===== $NAME L$L : extract + SAE $(date) ====="
    $PY eval_ctrl_plm.py --ckpt "$CK" --name "$NAME" --layer "$L" \
        --out-root outputs_ctrl --device cpu --sae-device cpu --sae-epochs 60 \
        || { echo "FAIL eval $NAME $L"; continue; }
    echo "===== $NAME L$L : L_struct $(date) ====="
    $PY cpu_stage.py --layer-dir "outputs_ctrl/$NAME/layer_$L" --model-type residue \
        --pdb-dir cache/pdb_files --fasta-path cache/scope_40.fa --n-shuffles 3 \
        || echo "FAIL cpu_stage $NAME $L"
    echo "===== $NAME L$L : concept-F1 $(date) ====="
    $PY experiment_concept_f1.py --layer-dir "outputs_ctrl/$NAME/layer_$L" \
        --save-dir "results_concept_f1_ctrl/${NAME}_l${L}" --fasta-path cache/scope_40.fa \
        || echo "FAIL conceptf1 $NAME $L"
  done
done
echo "EVAL_CTRL_A DONE $(date)"
