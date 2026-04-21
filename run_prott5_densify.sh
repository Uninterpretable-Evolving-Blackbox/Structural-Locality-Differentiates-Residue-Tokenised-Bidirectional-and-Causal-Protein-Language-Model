#!/usr/bin/env bash
# run_prott5_densify.sh — densify ProtT5 enc + dec to 9 depths per side.
#
# Stages:
#   1. DE-RISK  — run prott5_dec with layers [0,6,9,12,18,23].  Existing
#                 layers 0/6/12/18/23 are skipped by per-layer META.json
#                 check, so only layer 9 trains + cpu_stages.
#   2. GATE     — experiment_prott5_densify_check.py verifies that
#                   (a) val_EV at L9 fits the existing range (< 0.99), and
#                   (b) d_struct at L9 sits between L6 and L12 values.
#                 On failure: STOP, exit 1.
#   3. FULL DEC — rerun prott5_dec with the full 9-layer list; layer 9
#                 already done, so only 3/15/21 train + cpu_stage.
#   4. FULL ENC — prott5_enc with the full 9-layer list; existing
#                 0/6/12/18/23 skipped; 3/9/15/21 train + cpu_stage.
#   5. ANALYSIS — experiment_prott5_densify_analysis.py writes
#                 H3_enc_vs_dec_dense.{csv,png,pdf,txt}.
#
# Hyperparameters match existing ProtT5 runs exactly: k_sparse=256,
# expansion=8, seed=42, split=42.  Only new layers are added; no existing
# layer is retrained or overwritten.
set -euo pipefail

cd "$(dirname "$0")"
PY=./.venv/bin/python

# HF offline OK — ProtT5 weights cached from earlier runs.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "=========================================================================="
echo "  ProtT5 DENSIFICATION  —  start $(date)"
echo "=========================================================================="
echo ""

# ──────────── STAGE 1 — DE-RISK ────────────
echo "=== STAGE 1 (de-risk: prott5_dec layer 9) start $(date) ==="
PROTT5_DEC_LAYERS="0,6,9,12,18,23" MODEL=prott5_dec ./run_all.sh prott5_dec
echo "=== STAGE 1 (de-risk) done $(date) ==="
echo ""

# ──────────── STAGE 2 — GATE ────────────
echo "=== STAGE 2 (gate check) start $(date) ==="
if ! $PY experiment_prott5_densify_check.py; then
    echo ""
    echo "  ❌  De-risk gate FAILED — aborting before full expansion."
    echo "  Review outputs_layerwise/prott5_dec/layer_9/META.json and"
    echo "  struct_seq_metrics.csv at layers 6, 9, 12, then decide whether"
    echo "  to rerun with different hyperparameters or abandon the densification."
    exit 1
fi
echo "=== STAGE 2 (gate) done $(date) ==="
echo ""

# ──────────── STAGE 3 — FULL DEC ────────────
echo "=== STAGE 3 (full prott5_dec 9-layer) start $(date) ==="
PROTT5_DEC_LAYERS="0,3,6,9,12,15,18,21,23" MODEL=prott5_dec ./run_all.sh prott5_dec
echo "=== STAGE 3 (full dec) done $(date) ==="
echo ""

# ──────────── STAGE 4 — FULL ENC ────────────
echo "=== STAGE 4 (full prott5_enc 9-layer) start $(date) ==="
PROTT5_ENC_LAYERS="0,3,6,9,12,15,18,21,23" MODEL=prott5_enc ./run_all.sh prott5_enc
echo "=== STAGE 4 (full enc) done $(date) ==="
echo ""

# ──────────── STAGE 5 — ANALYSIS ────────────
echo "=== STAGE 5 (densified H3 analysis + plot) start $(date) ==="
$PY experiment_prott5_densify_analysis.py
echo "=== STAGE 5 (analysis) done $(date) ==="

echo ""
echo "=========================================================================="
echo "  ProtT5 DENSIFICATION COMPLETE  —  end $(date)"
echo "=========================================================================="
