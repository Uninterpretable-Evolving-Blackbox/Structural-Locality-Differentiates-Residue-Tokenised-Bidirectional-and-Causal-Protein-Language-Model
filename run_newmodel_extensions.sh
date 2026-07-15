#!/usr/bin/env bash
# run_newmodel_extensions.sh
# ---------------------------------------------------------------------------
# Runs the CPU-bound extension experiments for the NEW model pair
# (ProtBert-BFD bidirectional, ProGen2 causal) so we can compare them to
# ESM-2 / RITA on the same lenses — most importantly Concept-F1 (annotation
# alignment), plus null calibration, SAE diagnostics, and metric-agreement.
#
# Design goals:
#   * AUTO-CHAIN: waits for the Tier-4 fold-CI job to finish before starting.
#   * RESUMABLE: every (experiment, model, layer) is skipped if its output
#     already exists, so it is safe to kill and re-launch (e.g. while travelling).
#   * iCloud-SAFE: sources env_local_caches.sh (pycache/numba off the synced tree).
#
# Launch detached (survives terminal close):
#   nohup ./run_newmodel_extensions.sh \
#       >> results_new_plm_pair/new_model_extensions.detached.log 2>&1 < /dev/null &
#   echo $! > results_new_plm_pair/new_model_extensions.pid
# ---------------------------------------------------------------------------
set -u

cd "$(dirname "$0")"
# shellcheck disable=SC1091
source env_local_caches.sh
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

PY=.venv/bin/python
OUTROOT=outputs_layerwise_newpair
LOGDIR=results_new_plm_pair
mkdir -p "$LOGDIR"
LOG="$LOGDIR/new_model_extensions.log"

log(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

# Pre-registered 9-depth grids (must match training).
layers_for(){ case "$1" in
  protbert_bfd) echo "0 4 7 11 14 18 22 25 29";;
  progen2)      echo "0 3 6 10 13 16 20 23 26";;
esac; }

MODELS="protbert_bfd progen2"

# ---------------------------------------------------------------------------
# 1. Wait for Tier-4 fold CIs to finish (auto-chain).
#    Proceeds immediately if already complete. Times out after 24 h.
# ---------------------------------------------------------------------------
TIER4_LOG=outputs_robustness/run_tier4_fold_cis.detached.log
TIER4_DONE_RE="TIER 4 FOLD-LEVEL CI JOBS complete"

log "=== new-model extensions runner start ==="
if grep -q "$TIER4_DONE_RE" "$TIER4_LOG" 2>/dev/null; then
  log "Tier 4 already complete — proceeding immediately."
else
  log "Waiting for Tier 4 to finish (checking every 2 min, max 24 h)..."
  waited=0
  while ! grep -q "$TIER4_DONE_RE" "$TIER4_LOG" 2>/dev/null; do
    # If no compute_cis process is alive AND no done marker, Tier 4 likely
    # isn't running — wait anyway in case it gets launched, but log it.
    sleep 120; waited=$((waited+2))
    if [ $((waited % 30)) -eq 0 ]; then log "  ...still waiting (${waited} min)"; fi
    [ "$waited" -ge 1440 ] && { log "Tier 4 wait timed out (24 h) — starting anyway."; break; }
  done
  log "Tier 4 wait finished — starting extensions."
fi

# ---------------------------------------------------------------------------
# 2. SAE diagnostics (model-agnostic; fast).
# ---------------------------------------------------------------------------
log "######## SAE diagnostics ########"
for m in $MODELS; do for d in $(layers_for "$m"); do
  out="results_sae_diagnostics/${m}_l${d}"
  [ -f "$out/diagnostics.json" ] && { log "skip diag $m L$d"; continue; }
  log "diag $m L$d"
  $PY experiment_sae_diagnostics.py \
      --layer-dir "$OUTROOT/$m/layer_$d" --save-dir "$out" \
      >>"$LOG" 2>&1 || log "FAIL diag $m L$d"
done; done

# ---------------------------------------------------------------------------
# 3. Concept-F1 (THE priority: annotation alignment, fold-disjoint split).
# ---------------------------------------------------------------------------
log "######## Concept-F1 (annotation alignment) ########"
for m in $MODELS; do for d in $(layers_for "$m"); do
  out="results_concept_f1/${m}_l${d}"
  [ -f "$out/summary.json" ] && { log "skip cf1 $m L$d"; continue; }
  log "cf1 $m L$d"
  $PY experiment_concept_f1.py \
      --layer-dir "$OUTROOT/$m/layer_$d" --save-dir "$out" \
      >>"$LOG" 2>&1 || log "FAIL cf1 $m L$d"
done; done

# ---------------------------------------------------------------------------
# 4. Null calibration (effect size vs shuffled-graph floor).
# ---------------------------------------------------------------------------
log "######## Null calibration ########"
for m in $MODELS; do for d in $(layers_for "$m"); do
  out="results_null/${m}_l${d}"
  [ -f "$out/summary.json" ] && { log "skip null $m L$d"; continue; }
  log "null $m L$d"
  $PY experiment_null_calibration.py \
      --layer-dir "$OUTROOT/$m/layer_$d" --save-dir "$out" \
      >>"$LOG" 2>&1 || log "FAIL null $m L$d"
done; done

# ---------------------------------------------------------------------------
# 5. Metric-agreement (needs Concept-F1 output above).
# ---------------------------------------------------------------------------
log "######## Metric agreement (interp comparison) ########"
for m in $MODELS; do for d in $(layers_for "$m"); do
  out="results_interp_comparison/${m}_l${d}"
  [ -f "$out/summary.json" ] && { log "skip interp $m L$d"; continue; }
  cc="results_concept_f1/${m}_l${d}/feature_concept_best.csv"
  [ -f "$cc" ] || { log "skip interp $m L$d (no concept csv)"; continue; }
  log "interp $m L$d"
  $PY experiment_interp_comparison.py \
      --layer-dir "$OUTROOT/$m/layer_$d" --concept-csv "$cc" --save-dir "$out" \
      >>"$LOG" 2>&1 || log "FAIL interp $m L$d"
done; done

log "=== ALL NEW-MODEL EXTENSIONS DONE ==="
