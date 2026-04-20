#!/usr/bin/env bash
# Overnight chain:  split99-full  →  BPE-crossing-split99  →  metric-sweep
# Each stage stops the chain on failure (pipefail + exit codes).
set -euo pipefail

cd "$(dirname "$0")"
PY=./.venv/bin/python

# Offline mode — earlier ProtGPT2 run hit a transient HF SSL issue mid-pipeline
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "=========================================================================="
echo "  OVERNIGHT RUN  —  start $(date)"
echo "  Stage 1: split99 full pipeline (all 4 models) ~7-8 hr"
echo "  Stage 2: BPE-crossing on split99 ProtGPT2        ~15 min"
echo "  Stage 3: metric sweep on main seed=42             ~2-3 hr"
echo "=========================================================================="
echo ""

# -------------------------------------------------------------------
#  Stage 1: Data-split SPLIT_SEED=99, all 4 models
# -------------------------------------------------------------------
echo "=== STAGE 1  start $(date) ==="
SAE_SEED=42 SPLIT_SEED=99 RUN_SUFFIX=_split99 \
    ./run_all.sh all
echo "=== STAGE 1  done  $(date) ==="

# -------------------------------------------------------------------
#  Stage 2: BPE-crossing on split99 ProtGPT2 outputs
# -------------------------------------------------------------------
echo ""
echo "=== STAGE 2  start $(date) ==="
$PY experiment_bpe_crossing_all_depths.py \
    --outputs-dir outputs_layerwise_split99 \
    --out        results_bpe_crossing_split99 \
    --n-shuffles 5
echo "=== STAGE 2  done  $(date) ==="

# -------------------------------------------------------------------
#  Stage 3: Metric sweep on main seed=42 run
# -------------------------------------------------------------------
echo ""
echo "=== STAGE 3  start $(date) ==="
$PY experiment_metric_sweep.py \
    --outputs-dir outputs_layerwise \
    --out        results_metric_sweep \
    --n-shuffles 5
echo "=== STAGE 3  done  $(date) ==="

echo ""
echo "=========================================================================="
echo "  OVERNIGHT RUN COMPLETE  —  end $(date)"
echo "=========================================================================="
