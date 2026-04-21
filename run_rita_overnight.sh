#!/usr/bin/env bash
# Overnight chain: full RITA robustness package, mirroring the ProGen2 chain.
#   5 pipeline variants  →  val-only  →  metric sweep  →  aggregation
# Each stage stops the chain on failure (pipefail + &&).
set -euo pipefail

cd "$(dirname "$0")"
PY=./.venv/bin/python

# HF must be ONLINE so the first-run RITA weights (~3 GB) can be cached.
unset HF_HUB_OFFLINE
unset TRANSFORMERS_OFFLINE

echo "=========================================================================="
echo "  RITA OVERNIGHT  —  start $(date)"
echo "=========================================================================="
echo ""

STAGE=1

run_pipeline() {
    # $1 = suffix tag, $2 = env-var preamble, $3 = human label
    local tag="$1"; local prefix="$2"; local label="$3"
    echo ""
    echo "=== STAGE $STAGE ($label) start $(date) ==="
    # shellcheck disable=SC2086
    env $prefix MODEL=rita ./run_all.sh rita
    echo "=== STAGE $STAGE ($label) done $(date) ==="
    STAGE=$((STAGE+1))
}

run_h1h2() {
    # $1 = outputs dir, $2 = analysis dir
    local outdir="$1"; local ana="$2"
    echo "=== STAGE $STAGE (H1/H2 on $outdir) start $(date) ==="
    $PY experiment_h1h2_rita.py --outputs-dir "$outdir" --out "$ana"
    echo "=== STAGE $STAGE (H1/H2 on $outdir) done $(date) ==="
    STAGE=$((STAGE+1))
}

# ---- 5 pipeline variants ----
run_pipeline "main"    "SAE_SEED=42"                                                      "main seed42 k256"
run_h1h2    "outputs_layerwise"         "analysis_results_rita"

run_pipeline "seed43"  "SAE_SEED=43 RUN_SUFFIX=_seed43"                                   "seed43"
run_h1h2    "outputs_layerwise_seed43"  "analysis_results_rita_seed43"

run_pipeline "seed44"  "SAE_SEED=44 RUN_SUFFIX=_seed44"                                   "seed44"
run_h1h2    "outputs_layerwise_seed44"  "analysis_results_rita_seed44"

run_pipeline "k128"    "SAE_SEED=42 K_SPARSE=128 RUN_SUFFIX=_k128"                        "k=128"
run_h1h2    "outputs_layerwise_k128"    "analysis_results_rita_k128"

run_pipeline "split99" "SAE_SEED=42 SPLIT_SEED=99 RUN_SUFFIX=_split99"                    "split99"
run_h1h2    "outputs_layerwise_split99" "analysis_results_rita_split99"

# ---- val-only ----
echo ""
echo "=== STAGE $STAGE (val-only H1/H2 esm2+rita) start $(date) ==="
$PY experiment_val_only_rita.py --out analysis_results_valonly_rita --n-shuffles 5
echo "=== STAGE $STAGE (val-only) done $(date) ==="
STAGE=$((STAGE+1))

# ---- metric sweep (9 cells × 5 depths on esm2 + rita) ----
echo ""
echo "=== STAGE $STAGE (metric sweep esm2+rita) start $(date) ==="
$PY experiment_metric_sweep_rita.py --out results_metric_sweep_rita --n-shuffles 5
echo "=== STAGE $STAGE (metric sweep) done $(date) ==="
STAGE=$((STAGE+1))

# ---- aggregation ----
echo ""
echo "=== STAGE $STAGE (aggregate 5-run × 5-depth RITA robustness) start $(date) ==="
$PY aggregate_rita_robustness.py
echo "=== STAGE $STAGE (aggregate) done $(date) ==="

echo ""
echo "=========================================================================="
echo "  RITA OVERNIGHT COMPLETE  —  end $(date)"
echo "=========================================================================="
