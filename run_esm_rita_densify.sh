#!/usr/bin/env bash
# run_esm_rita_densify.sh — ESM-2 and RITA within-model H5 densification.
#
# Stages:
#   1. DE-RISK  — ESM-2 with layers [0,8,12,16,24,32].  Existing layers
#                 0/8/16/24/32 skip via per-layer META.json check, so only
#                 L12 trains + cpu_stages.
#   2. GATE     — experiment_esm_rita_densify_check.py verifies:
#                   (a) val_EV(L12) fits existing ESM-2 range (< 0.99);
#                   (b) mean struct_delta at L12 sits between L8 and L16.
#                 On failure: STOP.
#   3. FULL ESM-2 — rerun ESM-2 with full 9-layer list; L12 already done,
#                 so only 4/20/28 train + cpu_stage.
#   4. FULL RITA  — RITA with 9-layer list [0,3,6,9,12,15,18,21,23];
#                 existing 0/6/12/18/23 skipped; 3/9/15/21 train + cpu_stage.
#   5. ANALYSIS — experiment_esm_rita_densify_analysis.py writes
#                 H5_within_model_dense.{csv,png,pdf,txt}.
#
# Hyperparameters match existing runs: k_sparse=256, expansion=8, seed=42,
# split=42.  No existing layer is retrained.
set -euo pipefail

cd "$(dirname "$0")"
PY=./.venv/bin/python

# HF offline OK — ESM-2 and RITA weights cached from earlier runs.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "=========================================================================="
echo "  ESM-2 + RITA H5 DENSIFICATION  —  start $(date)"
echo "=========================================================================="
echo ""

# ──────────── STAGE 1 — DE-RISK (ESM-2 layer 12) ────────────
echo "=== STAGE 1 (de-risk: esm2 layer 12) start $(date) ==="
ESM2_LAYERS="0,8,12,16,24,32" MODEL=esm2 ./run_all.sh esm2
echo "=== STAGE 1 (de-risk) done $(date) ==="
echo ""

# ──────────── STAGE 2 — GATE ────────────
echo "=== STAGE 2 (gate check) start $(date) ==="
if ! $PY experiment_esm_rita_densify_check.py; then
    echo ""
    echo "  ❌  De-risk gate FAILED — aborting before remaining 7 SAEs."
    exit 1
fi
echo "=== STAGE 2 (gate) done $(date) ==="
echo ""

# ──────────── STAGE 3 — FULL ESM-2 ────────────
echo "=== STAGE 3 (full esm2 9-layer) start $(date) ==="
ESM2_LAYERS="0,4,8,12,16,20,24,28,32" MODEL=esm2 ./run_all.sh esm2
echo "=== STAGE 3 (full esm2) done $(date) ==="
echo ""

# ──────────── STAGE 4 — FULL RITA ────────────
echo "=== STAGE 4 (full rita 9-layer) start $(date) ==="
RITA_LAYERS="0,3,6,9,12,15,18,21,23" MODEL=rita ./run_all.sh rita
echo "=== STAGE 4 (full rita) done $(date) ==="
echo ""

# ──────────── STAGE 5 — ANALYSIS ────────────
echo "=== STAGE 5 (H5 within-model densified analysis + plot) start $(date) ==="
$PY experiment_esm_rita_densify_analysis.py
echo "=== STAGE 5 (analysis) done $(date) ==="

echo ""
echo "=========================================================================="
echo "  ESM-2 + RITA H5 DENSIFICATION COMPLETE  —  end $(date)"
echo "=========================================================================="
