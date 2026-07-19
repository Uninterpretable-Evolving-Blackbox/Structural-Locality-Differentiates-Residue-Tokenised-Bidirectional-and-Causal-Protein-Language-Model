#!/usr/bin/env bash
# run_ctrl_k96.sh — REDO the controlled pilot's SAE stage at a defensible sparsity.
#
# WHY: the original ctrl SAEs used k_sparse=256, which was ablated on ESM-2 L16
# (1280-dim). On the 480-dim controlled models that is k/embed_dim = 53.3% (vs 20.0%
# ESM-2, 16.7% RITA) -> barely a sparse decomposition. Measured val_EV: MLM
# 0.945-0.997, CLM 0.847-0.982; ctrl_mlm_A L0 = 0.9968 is ABOVE the 0.99 at which
# ProGen2 was dropped for a degenerate basis. The two arms also differ systematically
# in SAE fit (MLM better at every depth), so L_struct was being compared across
# bases of unequal quality.
#
# FIX: k_sparse=96. With expansion fixed at 8x, this reproduces ESM-2 L16's regime
# on BOTH axes at once: k/embed_dim 96/480 = 20.0% (== ESM-2's 256/1280) and
# k/hidden 96/3840 = 2.50% (== ESM-2's 256/10240). One anchor, one deviation (k),
# stated.
#
# Writes to outputs_ctrl_k96/ so the k=256 run is preserved for comparison.
# Idempotent: skips a cell whose Z.npy already exists.
set -u
cd "$(dirname "$0")"
source env_local_caches.sh 2>/dev/null || true
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=.venv/bin/python
CKDIR="$HOME/own_sae_data/uniref50_pilot"
LAYERS="0 3 6 9 11"
K=96
ROOT=outputs_ctrl_k96

for M in mlm clm; do
  NAME="ctrl_${M}_A"; CK="$CKDIR/ckpt_${M}/model_final.pt"
  for L in $LAYERS; do
    LD="$ROOT/$NAME/layer_$L"
    if [ -f "$LD/Z.npy" ]; then echo "[skip] $NAME L$L"; continue; fi
    echo "===== $NAME L$L : extract + SAE (k=$K) $(date) ====="
    $PY eval_ctrl_plm.py --ckpt "$CK" --name "$NAME" --layer "$L" \
        --out-root "$ROOT" --device cpu --sae-device cpu --sae-epochs 60 \
        --expansion 8 --k-sparse "$K" \
        || { echo "FAIL eval $NAME $L"; continue; }
    echo "===== $NAME L$L : L_struct $(date) ====="
    $PY cpu_stage.py --layer-dir "$LD" --model-type residue \
        --pdb-dir cache/pdb_files --fasta-path cache/scope_40.fa --n-shuffles 3 \
        || echo "FAIL cpu_stage $NAME $L"
    echo "===== $NAME L$L : concept-F1 $(date) ====="
    $PY experiment_concept_f1.py --layer-dir "$LD" \
        --save-dir "results_concept_f1_ctrl_k96/${NAME}_l${L}" --fasta-path cache/scope_40.fa \
        || echo "FAIL conceptf1 $NAME $L"
  done
done

echo "===== val_EV check on the NEW SAEs $(date) ====="
$PY -u eval_ctrl_sae_ev.py --roots "$ROOT/ctrl_mlm_A,$ROOT/ctrl_clm_A" \
    --layers 0,3,6,9,11 --out results_ctrl_sae_ev/summary_k96.json \
    || echo "FAIL val_EV"

echo "===== causal attribution on the NEW SAEs $(date) ====="
mkdir -p results_ctrl_causal_k96
for M in mlm clm; do
  CAUS=""; [ "$M" = clm ] && CAUS="--causal"
  for L in $LAYERS; do
    $PY eval_ctrl_causal.py --ckpt "$CKDIR/ckpt_${M}/model_final.pt" \
      --layer-dir "$ROOT/ctrl_${M}_A/layer_${L}" --layer "$L" $CAUS \
      --max-proteins 200 --device cpu > "results_ctrl_causal_k96/${M}_l${L}.json" \
      2>>results_ctrl_causal_k96/err.log || echo "FAIL causal $M $L"
  done
done

echo "===== fold/protein bootstrap $(date) ====="
$PY outputs_robustness/compute_h1_bootstrap.py --preset ctrl_k96 || echo "FAIL bootstrap"

echo "CTRL_K96 DONE $(date)"
