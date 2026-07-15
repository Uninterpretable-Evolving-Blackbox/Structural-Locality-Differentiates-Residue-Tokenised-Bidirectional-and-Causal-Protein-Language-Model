#!/usr/bin/env bash
# run_pilot_B.sh — controlled MLM-vs-CLM pilot, "B" pass (~300M tokens each).
# LR schedule targets the full compute-optimal budget (660M) so --resume finishes "A".
# Checkpoints every ~82M tokens -> crash-safe / resumable. Writes only to ~/own_sae_data.
set -u
cd "$(dirname "$0")"
source env_local_caches.sh 2>/dev/null || true
PY=.venv/bin/python
DATA="$HOME/own_sae_data/uniref50_pilot"
TARGET=660e6      # compute-optimal ~20 tok/param for 33.2M
STOP=300e6        # this "B" run stops here; resume (no --stop-at-tokens) finishes to TARGET

for OBJ in mlm clm; do
  echo "=================== $OBJ (B: ${STOP} tok) $(date) ==================="
  $PY -u train_ctrl_plm.py --objective "$OBJ" --data-dir "$DATA" \
      --target-tokens "$TARGET" --stop-at-tokens "$STOP" \
      --batch-size 32 --seq-len 512 --lr 6e-4 --warmup 500 \
      --ckpt-every 5000 --val-every 1000 \
      --out-dir "$DATA/ckpt_${OBJ}" || echo "FAIL $OBJ"
done
echo "PILOT_B DONE $(date)"
