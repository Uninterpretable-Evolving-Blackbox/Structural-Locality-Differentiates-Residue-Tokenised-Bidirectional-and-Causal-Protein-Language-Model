#!/usr/bin/env bash
# run_robustness_overnight.sh — overnight chain for v6 robustness + densification.
#
# Stages:
#   1. DENSIFY PROTGPT2    — add layers {4, 13, 22, 31} to ProtGPT2; existing
#                            5 layers skipped by per-layer META.json check.
#   2. BPE-CROSS NEW PROTGPT2 LAYERS — run experiment_bpe_correction.py
#                            for each new layer so H2' has 9-depth coverage.
#   3. Cα CUTOFF SWEEP     — contact_cutoff ∈ {6, 8, 10} Å × 4 struct-applicable
#                            models × 9 depths.
#   4. SEQ-WINDOW SWEEP    — window ∈ {1, 2, 4} × 3 seq-applicable models × 9
#                            depths; ProtGPT2 sweeps raw AND inter-token.
#   5. EXTEND METRIC SWEEP TO 9 DEPTHS FOR ESM vs PROTGPT2  (bonus, cheap)
#
# Hyperparameters match existing runs; no retraining of existing SAEs.
set -euo pipefail

cd "$(dirname "$0")"
PY=./.venv/bin/python
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "=========================================================================="
echo "  ROBUSTNESS + PROTGPT2 DENSIFICATION OVERNIGHT  —  start $(date)"
echo "=========================================================================="

STAGE=0
mark_start() { STAGE=$((STAGE+1)); echo ""; echo "=== STAGE $STAGE ($1) start $(date) ==="; }
mark_end()   {                     echo "=== STAGE $STAGE ($1) done  $(date) ==="; }

# ─────────── 1. ProtGPT2 densification ───────────
mark_start "ProtGPT2 densify (layers 0,4,9,13,18,22,27,31,35)"
PROTGPT2_LAYERS="0,4,9,13,18,22,27,31,35" MODEL=protgpt2 ./run_all.sh protgpt2
mark_end "ProtGPT2 densify"

# ─────────── 2. BPE-cross the 4 new ProtGPT2 layers for H2' ───────────
mark_start "BPE-correction for new ProtGPT2 layers {4,13,22,31}"
pg2_to_esm() {
    case "$1" in
        4)  echo 4 ;;
        13) echo 12 ;;
        22) echo 20 ;;
        31) echo 28 ;;
    esac
}
for L in 4 13 22 31; do
    ESM_L="$(pg2_to_esm "$L")"
    echo "  -- ProtGPT2 L$L  vs  ESM-2 L$ESM_L --"
    $PY experiment_bpe_correction.py \
        --layer-dir      "outputs_layerwise/protgpt2/layer_$L" \
        --esm2-layer-dir "outputs_layerwise/esm2/layer_$ESM_L" \
        --save-dir       "results_bpe_crossing_new/l$L" \
        --n-shuffles 5 || echo "    (skipped: layer $L BPE crossing failed)"
done
mark_end "BPE-correction new ProtGPT2 layers"

# ─────────── 3. Cα cutoff sweep ───────────
mark_start "Cα cutoff sweep (ESM-2, RITA, ProtT5-enc, ProtT5-dec × 9 depths × 3 cutoffs)"
$PY experiment_cutoff_sweep.py \
    --out results_cutoff_sweep \
    --n-shuffles 5
mark_end "Cα cutoff sweep"

# ─────────── 4. Seq-window sweep ───────────
mark_start "Seq-window sweep (ESM-2, RITA, ProtGPT2 × 9 depths × 3 windows)"
$PY experiment_seqwindow_sweep.py \
    --out results_seqwindow_sweep \
    --n-shuffles 5
mark_end "Seq-window sweep"

echo ""
echo "=========================================================================="
echo "  ROBUSTNESS OVERNIGHT COMPLETE  —  end $(date)"
echo "=========================================================================="
echo ""
echo "  Artefacts:"
echo "    outputs_layerwise/protgpt2/layer_{4,13,22,31}/   (new SAEs + cpu_stage)"
echo "    results_bpe_crossing_new/l{4,13,22,31}/          (H2' at new depths)"
echo "    results_cutoff_sweep/                            (Cα sweep tables)"
echo "    results_seqwindow_sweep/                         (seq-window sweep tables)"
