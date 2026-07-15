#!/usr/bin/env bash
# run_plms_then_resume_tier4.sh — priority chain for the overnight run.
#
# 1. Run the new bidirectional/causal PLM pair full grid (ProtBert-BFD + ProGen2),
#    which is the deadline-bound task.
# 2. Then resume the paused Tier 4 fold-level CI jobs. compute_cis_v2.py skips
#    already-completed depths (ESM/RITA 90/90, ProtT5 L0/L3/L6 done), so it picks
#    up where it left off; compute_cis_v3_optimized.py then runs.
#
# Both stages are wrapped so a non-zero exit in one does not prevent the other
# from being attempted (we still want the machine busy overnight).
set -uo pipefail

cd "$(dirname "$0")"

echo "=========================================================================="
echo "  CHAIN START $(date)"
echo "=========================================================================="

echo ""
echo ">>> STAGE 1: new PLM pair full grid  $(date)"
if ./run_new_plm_pair_full_grid.sh; then
    echo ">>> STAGE 1 complete (new PLM pair)  $(date)"
else
    echo ">>> WARNING: STAGE 1 (new PLM pair) exited non-zero  $(date)"
fi

echo ""
echo ">>> STAGE 2: resume Tier 4 fold-level CIs  $(date)"
if ./run_tier4_fold_cis.sh; then
    echo ">>> STAGE 2 complete (Tier 4)  $(date)"
else
    echo ">>> WARNING: STAGE 2 (Tier 4) exited non-zero  $(date)"
fi

echo ""
echo "=========================================================================="
echo "  CHAIN DONE $(date)"
echo "=========================================================================="
