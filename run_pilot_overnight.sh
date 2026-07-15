#!/usr/bin/env bash
# run_pilot_overnight.sh — AUTONOMOUS staged controlled MLM-vs-CLM run.
#
# Stage 1: bring BOTH objectives to 300M tokens ("B" — a MATCHED budget for a fair
#          comparison). MLM is already at 300M (instant no-op); CLM trains (~5h).
# Stage 2: bring BOTH to 660M tokens ("A" — full compute-optimal).
#
# Self-healing: resumes from the latest checkpoint on crash/restart (up to MAXTRIES).
# Idempotent: skips an objective/stage whose model_final.pt already exists.
# Writes ONLY to ~/own_sae_data. LR schedule always targets 660M so resume is clean.
set -u
cd "$(dirname "$0")"
source env_local_caches.sh 2>/dev/null || true
PY=.venv/bin/python
DATA="$HOME/own_sae_data/uniref50_pilot"
TARGET=660e6
MAXTRIES=5

latest_ckpt () {
  ls -t "$1"/model_step*.pt "$1"/model_partial.pt "$1"/model_final.pt 2>/dev/null | head -1
}

train_to () {  # $1=objective  $2=stop_tokens (0=full)  $3=stage tag
  local OBJ="$1" STOP="$2" TAG="$3" D="$DATA/ckpt_$1"
  mkdir -p "$D"
  [ -f "$D/model_final.pt" ] && { echo "[$OBJ/$TAG] model_final exists — skip"; return 0; }
  local stoparg=""; [ "$STOP" != "0" ] && stoparg="--stop-at-tokens $STOP"
  for try in $(seq 1 $MAXTRIES); do
    local lc; lc=$(latest_ckpt "$D"); local res=""; [ -n "$lc" ] && res="--resume $lc"
    echo "=== [$OBJ/$TAG] try $try stop=$STOP resume='${lc:-none}' $(date) ==="
    $PY -u train_ctrl_plm.py --objective "$OBJ" --data-dir "$DATA" \
        --target-tokens "$TARGET" $stoparg \
        --batch-size 32 --seq-len 512 --lr 6e-4 --warmup 500 \
        --ckpt-every 2500 --val-every 1000 --out-dir "$D" $res && return 0
    echo "[$OBJ/$TAG] try $try FAILED $(date) — resuming from latest checkpoint"; sleep 15
  done
  echo "[$OBJ/$TAG] GAVE UP after $MAXTRIES tries"; return 1
}

echo "########## STAGE 1: both -> 300M (matched B) $(date) ##########"
train_to mlm 300e6 B
train_to clm 300e6 B
echo "########## STAGE 2: both -> 660M (full A) $(date) ##########"
train_to mlm 0 A
train_to clm 0 A
echo "PILOT_OVERNIGHT DONE $(date)"
