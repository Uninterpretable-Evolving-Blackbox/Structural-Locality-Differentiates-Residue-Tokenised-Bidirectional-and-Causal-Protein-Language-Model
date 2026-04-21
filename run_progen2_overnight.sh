#!/usr/bin/env bash
# Overnight chain: full ProGen2 robustness package.
#   5 pipeline variants  →  val-only  →  metric sweep  →  aggregation
# Each stage stops the chain on failure (pipefail + &&).
set -euo pipefail

cd "$(dirname "$0")"
PY=./.venv/bin/python

# HF must be ONLINE for first-run model download (~3 GB cached thereafter).
unset HF_HUB_OFFLINE
unset TRANSFORMERS_OFFLINE

echo "=========================================================================="
echo "  PROGEN2 OVERNIGHT  —  start $(date)"
echo "=========================================================================="
echo ""

STAGE=1

run_pipeline() {
    # $1 = suffix tag, $2 = env-var preamble, $3 = human label
    local tag="$1"; local prefix="$2"; local label="$3"
    echo ""
    echo "=== STAGE $STAGE ($label) start $(date) ==="
    # shellcheck disable=SC2086
    env $prefix MODEL=progen2 ./run_all.sh progen2
    echo "=== STAGE $STAGE ($label) done $(date) ==="
    STAGE=$((STAGE+1))
}

run_h1h2() {
    # $1 = outputs dir, $2 = analysis dir
    local outdir="$1"; local ana="$2"
    echo "=== STAGE $STAGE (H1/H2 on $outdir) start $(date) ==="
    $PY experiment_h1h2_progen2.py --outputs-dir "$outdir" --out "$ana"
    echo "=== STAGE $STAGE (H1/H2 on $outdir) done $(date) ==="
    STAGE=$((STAGE+1))
}

# ---- 5 pipeline variants ----
run_pipeline "main"    "SAE_SEED=42"                                                     "main seed42 k256"
run_h1h2    "outputs_layerwise"         "analysis_results_progen2"

run_pipeline "seed43"  "SAE_SEED=43 RUN_SUFFIX=_seed43"                                  "seed43"
run_h1h2    "outputs_layerwise_seed43"  "analysis_results_progen2_seed43"

run_pipeline "seed44"  "SAE_SEED=44 RUN_SUFFIX=_seed44"                                  "seed44"
run_h1h2    "outputs_layerwise_seed44"  "analysis_results_progen2_seed44"

run_pipeline "k128"    "SAE_SEED=42 K_SPARSE=128 RUN_SUFFIX=_k128"                       "k=128"
run_h1h2    "outputs_layerwise_k128"    "analysis_results_progen2_k128"

run_pipeline "split99" "SAE_SEED=42 SPLIT_SEED=99 RUN_SUFFIX=_split99"                   "split99"
run_h1h2    "outputs_layerwise_split99" "analysis_results_progen2_split99"

# ---- val-only (ESM-2 already in cache from earlier val-only run; progen2 fresh) ----
echo ""
echo "=== STAGE $STAGE (val-only H1/H2 esm2+progen2) start $(date) ==="
$PY experiment_val_only_progen2.py --out analysis_results_valonly_progen2 --n-shuffles 5
echo "=== STAGE $STAGE (val-only) done $(date) ==="
STAGE=$((STAGE+1))

# ---- metric sweep (9 cells × 5 depths on esm2 + progen2) ----
echo ""
echo "=== STAGE $STAGE (metric sweep esm2+progen2) start $(date) ==="
$PY experiment_metric_sweep_progen2.py --out results_metric_sweep_progen2 --n-shuffles 5
echo "=== STAGE $STAGE (metric sweep) done $(date) ==="
STAGE=$((STAGE+1))

# ---- aggregation ----
echo ""
echo "=== STAGE $STAGE (aggregate 5-run × 5-depth ProGen2 robustness) start $(date) ==="
$PY aggregate_progen2_robustness.py
echo "=== STAGE $STAGE (aggregate) done $(date) ==="

echo ""
echo "=========================================================================="
echo "  PROGEN2 OVERNIGHT COMPLETE  —  end $(date)"
echo "=========================================================================="
