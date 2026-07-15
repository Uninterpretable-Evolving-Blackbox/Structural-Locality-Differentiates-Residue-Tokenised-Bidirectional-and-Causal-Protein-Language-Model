#!/usr/bin/env bash
# run_concept_f1_multiseed_headline.sh — Option A: symmetric Concept-F1 seeds
#
# Trained SAE seeds {42, 43, 44} on headline cells:
#   ESM-2 L16, RITA L18
# (seed 42 already in results_concept_f1/; this adds 43/44.)
#
# Random-weights control already has 3 PLM weight-init seeds at ESM-2 L16
# (results_concept_f1_random{,_weightseed1,_weightseed2}/).
#
# Prerequisite: seed43/44 layer dirs may be missing Z.npy (iCloud eviction).
# regen_z_from_sae.py rebuilds Z from sae_model.pt before Concept-F1.
#
# Launch detached:
#   nohup ./run_concept_f1_multiseed_headline.sh \
#       >> results_concept_f1_multiseed_headline/detached.log 2>&1 < /dev/null &
#   echo $! > results_concept_f1_multiseed_headline/run.pid
set -euo pipefail

cd "$(dirname "$0")"
source env_local_caches.sh

PY="${PY:-./.venv/bin/python}"
LOGDIR=results_concept_f1_multiseed_headline
mkdir -p "$LOGDIR"
LOG="$LOGDIR/run.log"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

# PLM embeddings are identical across SAE seeds; reuse cached raw_embeddings.npy.
maybe_copy_raw_cache() {
  local src="$1" dst="$2"
  if [ ! -f "${dst}/raw_embeddings.npy" ] && [ -f "${src}/raw_embeddings.npy" ]; then
    log "  copy raw_embeddings cache from ${src}"
    cp "${src}/raw_embeddings.npy" "${dst}/raw_embeddings.npy"
  fi
}

# sae_seed | outputs_root | model | layer
run_trained_cell() {
  local sae_seed="$1" root="$2" model="$3" layer="$4"
  local layer_dir="${root}/${model}/layer_${layer}"
  local save_dir="results_concept_f1_seed${sae_seed}/${model}_l${layer}"

  log "trained SAE seed=${sae_seed} ${model} L${layer}"

  if [ ! -f "${layer_dir}/META.json" ]; then
    log "  SKIP missing layer dir ${layer_dir}"
    return 1
  fi

  if [ ! -f "${layer_dir}/Z.npy" ]; then
    if [ "$sae_seed" = "44" ]; then
      maybe_copy_raw_cache "outputs_layerwise_seed43/${model}/layer_${layer}" "$layer_dir"
    fi
    log "  regen Z.npy from sae_model.pt"
    "$PY" regen_z_from_sae.py --layer-dir "$layer_dir" --model "$model" --layer "$layer"
  fi

  if [ -f "${save_dir}/summary.json" ]; then
    log "  SKIP concept-F1 (summary.json exists)"
    return 0
  fi

  # Protein-level concept val/test split matches the existing seed-42 and
  # random-weight headline runs (80 concepts scored). Fold-disjoint split
  # leaves too few cross-split concepts for this headline mean-F1 statistic.
  "$PY" experiment_concept_f1.py --layer-dir "$layer_dir" --save-dir "$save_dir" \
    --split-level protein
  log "  concept-F1 done -> ${save_dir}"
}

log "=== CONCEPT-F1 MULTI-SEED HEADLINE start $(date) ==="

for sae_seed in 43 44; do
  run_trained_cell "$sae_seed" "outputs_layerwise_seed${sae_seed}" esm2 16
  run_trained_cell "$sae_seed" "outputs_layerwise_seed${sae_seed}" rita 18
done

log "=== summarizing ==="
"$PY" summarize_concept_f1_multiseed_headline.py

log "=== CONCEPT-F1 MULTI-SEED HEADLINE DONE $(date) ==="