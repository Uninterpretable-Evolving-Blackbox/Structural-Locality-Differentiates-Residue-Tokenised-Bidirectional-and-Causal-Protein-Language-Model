#!/usr/bin/env bash
# run_pilot_A.sh — overnight AUTONOMOUS run: train MLM and CLM to the full
# compute-optimal budget (A = 660M tokens each). Self-healing: on restart/crash it
# resumes from the latest checkpoint. Saves ~300M checkpoints en route (the "B" read).
# Idempotent: skips an objective whose model_final.pt already exists.
# Writes ONLY to ~/own_sae_data.
set -u
cd "$(dirname "$0")"
source env_local_caches.sh 2>/dev/null || true
PY=.venv/bin/python
DATA="$HOME/own_sae_data/uniref50_pilot"
TARGET=660e6
MAXTRIES=5

latest_ckpt () {   # echo "DONE", a checkpoint path, or empty
  local d="$1"
  [ -f "$d/model_final.pt" ] && { echo "DONE"; return; }
  ls -t "$d"/model_step*.pt "$d"/model_partial.pt 2>/dev/null | head -1
}

for OBJ in mlm clm; do
  D="$DATA/ckpt_${OBJ}"; mkdir -p "$D"
  for try in $(seq 1 $MAXTRIES); do
    lc=$(latest_ckpt "$D")
    if [ "$lc" = "DONE" ]; then echo "[$OBJ] already final — skip"; break; fi
    RES=""; [ -n "$lc" ] && RES="--resume $lc"
    echo "=========== [$OBJ] try $try  resume='${lc:-none}'  $(date) ==========="
    $PY -u train_ctrl_plm.py --objective "$OBJ" --data-dir "$DATA" \
        --target-tokens "$TARGET" \
        --batch-size 32 --seq-len 512 --lr 6e-4 --warmup 500 \
        --ckpt-every 2500 --val-every 1000 \
        --out-dir "$D" $RES && break
    echo "[$OBJ] try $try FAILED $(date) — will resume from latest checkpoint"
    sleep 15
  done
done
echo "PILOT_A DONE $(date)"
