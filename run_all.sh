#!/usr/bin/env bash
# run_all.sh — Complete SAE-PLM pipeline
# ========================================
#
# Runs the full pipeline: dataset build → embedding extraction + SAE training
# → CPU structural analysis → hypothesis testing.
#
# Usage:
#   ./run_all.sh              # run everything (auto-detect device)
#   ./run_all.sh esm2         # single model only
#   DEVICE=cpu ./run_all.sh   # force CPU
#
# Prerequisites:
#   pip install torch transformers biopython scipy scikit-learn
#                 pandas matplotlib seaborn umap-learn tqdm joblib requests
#   conda install -c salilab dssp   # or: brew install dssp

set -euo pipefail

# Pin python interpreter to the project's .venv so the script never depends
# on which python3 happens to be on PATH (e.g. system python vs anaconda
# base vs venv).  Override with: PY=/some/other/python ./run_all.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-$SCRIPT_DIR/.venv/bin/python}"
if [ ! -x "$PY" ]; then
    echo "❌ Python interpreter not found at $PY" >&2
    echo "   Either create the venv (python3 -m venv .venv && .venv/bin/pip install -r requirements.txt)" >&2
    echo "   or override with: PY=/path/to/python ./run_all.sh" >&2
    exit 1
fi
echo "Using python: $PY ($($PY --version))"

MODEL="${1:-all}"

# Multi-variant support: lets us run different (seed, k_sparse, dataset)
# combos to separate output dirs without clobbering each other.
#   RUN_SUFFIX=""        → outputs_layerwise/         (default)
#   RUN_SUFFIX="_seed43" → outputs_layerwise_seed43/
#   RUN_SUFFIX="_k128"   → outputs_layerwise_k128/
RUN_SUFFIX="${RUN_SUFFIX:-}"
OUTPUT_ROOT="outputs_layerwise${RUN_SUFFIX}"
ANALYSIS_DIR="analysis_results${RUN_SUFFIX}"
echo "  Output root: $OUTPUT_ROOT"
echo "  Analysis dir: $ANALYSIS_DIR"

# Auto-detect device
if "$PY" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    DEVICE="${DEVICE:-cuda}"
elif "$PY" -c "import torch; assert torch.backends.mps.is_available()" 2>/dev/null; then
    DEVICE="${DEVICE:-mps}"
else
    DEVICE="${DEVICE:-cpu}"
fi

echo "============================================"
echo "  SAE-PLM Pipeline"
echo "  Device: $DEVICE"
echo "  Model:  $MODEL"
echo "============================================"

# ── Step 0: Build dataset (if not already done) ──
if [ ! -f "cache/sequences.json" ]; then
    echo ""
    echo "──── Step 0: Building dataset from SCOPe ────"
    "$PY" build_dataset.py
else
    echo ""
    echo "──── Step 0: Dataset already built (cache/sequences.json exists) ────"
fi

# ── Step 1: GPU stage (extract embeddings + train SAEs) ──
echo ""
echo "──── Step 1: Embedding extraction + SAE training ────"
DEVICE=$DEVICE MODEL=$MODEL RUN_SUFFIX=$RUN_SUFFIX \
  SAE_SEED=${SAE_SEED:-42} K_SPARSE=${K_SPARSE:-256} EXPANSION=${EXPANSION:-8} \
  "$PY" run_unsupervised.py

# ── Step 2: CPU stage (structural analysis per layer) ──
echo ""
echo "──── Step 2: Structural analysis (CPU stage) ────"

# Determine which models to process
if [ "$MODEL" = "all" ]; then
    MODELS=("esm2" "protgpt2" "prott5_enc" "prott5_dec" "rita")
else
    MODELS=("$MODEL")
fi

for M in "${MODELS[@]}"; do
    MODEL_DIR="${OUTPUT_ROOT}/$M"
    if [ ! -d "$MODEL_DIR" ]; then
        echo "  Skipping $M (no output directory)"
        continue
    fi

    # Determine model type for cpu_stage
    if [ "$M" = "protgpt2" ]; then
        MTYPE="protgpt2"
    else
        MTYPE="residue"
    fi

    for LAYER_DIR in "$MODEL_DIR"/layer_*; do
        if [ ! -d "$LAYER_DIR" ]; then
            continue
        fi

        # Skip if already analyzed
        if [ -f "$LAYER_DIR/struct_seq_metrics.csv" ] && [ -f "$LAYER_DIR/feature_interpretability.csv" ]; then
            echo "  Skipping $LAYER_DIR (already analyzed)"
            continue
        fi

        echo "  Analyzing $LAYER_DIR..."
        "$PY" cpu_stage.py \
            --layer-dir "$LAYER_DIR" \
            --model-type "$MTYPE" \
            --features-csv cache/residue_features.csv \
            --pdb-dir cache/pdb_files \
            --fasta-path cache/scope_40.fa \
            --n-shuffles 5 \
            --sweep-topk
    done
done

# ── Step 3: Hypothesis testing ──
echo ""
echo "──── Step 3: Hypothesis testing ────"
"$PY" analyze_hypotheses.py \
    --root "$OUTPUT_ROOT" \
    --save-dir "$ANALYSIS_DIR"

echo ""
echo "============================================"
echo "  Pipeline complete!"
echo "  Results: ${ANALYSIS_DIR}/"
echo "============================================"
