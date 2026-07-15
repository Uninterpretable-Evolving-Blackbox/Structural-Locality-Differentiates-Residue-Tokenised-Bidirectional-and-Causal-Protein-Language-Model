#!/usr/bin/env bash
# Queued JOINT k x expansion sweep. Waits for the overnight MPS runner to finish
# (so only one training job uses the GPU at a time), then runs the full 2-D grid
# on the 6 layers with cached raw embeddings — main H1 layers (esm2 L16, rita
# L12) FIRST so the headline result lands even if later layers run long.
# Idempotent: skips any layer whose summary.json already exists.
set -u
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
PY=.venv/bin/python
KS="64,128,256"
EXPS="4,8,16,32"

# wait (up to 16h) for the MPS overnight runner to finish, to avoid GPU contention
w=0
while ! grep -q "OVERNIGHT MPS DONE" overnight_mps.log 2>/dev/null; do
  sleep 120; w=$((w+1)); [ $w -ge 480 ] && { echo "[joint] wait timeout, starting anyway"; break; }
done
echo "[joint] starting $(date)"

# (model layer) pairs — main H1 layers first, then early/late for depth generality
for pair in "esm2 16" "rita 12" "esm2 0" "esm2 32" "rita 0" "rita 23"; do
  set -- $pair; m=$1; d=$2
  out=results_joint_sweep/${m}_l${d}
  [ -f "$out/summary.json" ] && { echo "[joint] skip $m l$d (done)"; continue; }
  echo "######## joint sweep: $m layer $d ########"
  $PY experiment_joint_sweep.py --layer-dir outputs_layerwise/$m/layer_$d \
    --ks "$KS" --expansions "$EXPS" --save-dir "$out" || echo "FAIL joint $m $d"
done
echo "[joint] JOINT SWEEP DONE $(date)"
