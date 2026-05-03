#!/usr/bin/env bash
# run_scale_and_resample_overnight.sh — two robustness axes in one overnight:
#
#   (A) PLM-scale ablation  : ESM-2 t12 (35M) vs RITA-s (85M) at 5 matched
#                              depths. Tests H1 at ~1/20× the parameter count.
#
#   (B) Protein resample    : fresh SCOPe 1500-protein subsample
#                              (--seed 43), rerun the 5 paper models + H1/H2
#                              tests. Tests H1/H2/H3 aren't specific to the
#                              original subsample.
#
# Cache handling: subsample_dataset.py overwrites cache/sequences.json and
# cache/residue_features.csv. full_sequences.json / full_residue_features.csv
# preserve the complete ~11k SCOPe population, so any seed re-subsamples from
# the full cache. We restore at the end so the default cache is back to seed=42.
set -euo pipefail

cd "$(dirname "$0")"
PY=./.venv/bin/python
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

STAGE=0
mark_start() { STAGE=$((STAGE+1)); echo ""; echo "=== STAGE $STAGE ($1) start $(date) ==="; }
mark_end()   {                     echo "=== STAGE $STAGE ($1) done  $(date) ==="; }

echo "=========================================================================="
echo "  SCALE + RESAMPLE OVERNIGHT  —  start $(date)"
echo "=========================================================================="

# ─────────── (A) PLM-scale ablation ───────────
mark_start "ESM-2 t12 (35M) pipeline — MODEL=esm2_small"
MODEL=esm2_small ./run_all.sh esm2_small
mark_end   "ESM-2 t12 pipeline"

mark_start "RITA-s (85M) pipeline — MODEL=rita_small"
# HF must be online the first time RITA_s is fetched (~350 MB)
unset HF_HUB_OFFLINE
unset TRANSFORMERS_OFFLINE
MODEL=rita_small ./run_all.sh rita_small
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
mark_end   "RITA-s pipeline"

mark_start "H1 at smaller scale — experiment_h1_scale.py"
$PY experiment_h1_scale.py
mark_end   "H1 at smaller scale"

# ─────────── (B) Protein resample (SCOPe seed=43) ───────────
mark_start "Resample SCOPe subsample with --seed 43"
$PY subsample_dataset.py --seed 43 --min-coverage 0.80
mark_end   "Resample subsample"

# Run only the 5 paper models (skip progen2 which was paper-dropped).
for M in esm2 protgpt2 prott5_enc prott5_dec rita; do
    mark_start "dseed=43 pipeline — MODEL=$M"
    MODEL=$M RUN_SUFFIX=_dseed43 ./run_all.sh $M
    mark_end   "dseed=43 pipeline $M"
done

mark_start "dseed=43 H1 ESM-2 vs RITA"
$PY experiment_h1h2_rita.py \
    --outputs-dir outputs_layerwise_dseed43 \
    --out        analysis_results_rita_dseed43
mark_end "dseed=43 H1"

# ─────────── Restore default cache (seed=42) ───────────
mark_start "Restore default SCOPe subsample (seed=42)"
$PY subsample_dataset.py --restore
mark_end "Restore"

echo ""
echo "=========================================================================="
echo "  SCALE + RESAMPLE OVERNIGHT COMPLETE  —  end $(date)"
echo "=========================================================================="
echo ""
echo "  Artefacts:"
echo "    outputs_layerwise/esm2_small/layer_{0,3,6,9,11}/"
echo "    outputs_layerwise/rita_small/layer_{0,3,6,9,11}/"
echo "    analysis_results/comparison/H1_scale_small.{csv,txt}"
echo "    outputs_layerwise_dseed43/{esm2,protgpt2,prott5_enc,prott5_dec,rita}/"
echo "    analysis_results_dseed43/comparison/H1_H2_all_depths.csv"
echo "    analysis_results_rita_dseed43/comparison/H1_H2_esm2_vs_rita.csv"
