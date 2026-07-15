#!/usr/bin/env bash
# run_tier1b_random_weight_seeds.sh — C1 randomized-weights robustness.
#
# Adds weight seeds 1 and 2 for the headline ESM-2 layer-16 randomized-weights
# control. Seed 0 already lives under outputs_random/ and results_*_random/.
# These outputs use distinct roots so the existing seed-0 control is preserved.
set -euo pipefail

cd "$(dirname "$0")"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

PY="${PY:-./.venv/bin/python}"
LAYERS="${TIER1B_LAYERS:-16}"
WEIGHT_SEEDS="${TIER1B_WEIGHT_SEEDS:-1 2}"
EPOCHS="${TIER1B_EPOCHS:-60}"

echo "=========================================================================="
echo "  TIER 1b C1 RANDOMIZED-WEIGHTS MULTI-SEED start $(date)"
echo "  layers=${LAYERS}"
echo "  weight_seeds=${WEIGHT_SEEDS}"
echo "=========================================================================="

for weight_seed in ${WEIGHT_SEEDS}; do
  for layer in ${LAYERS}; do
    out_layer="outputs_random_weightseed${weight_seed}/esm2/layer_${layer}"
    concept_out="results_concept_f1_random_weightseed${weight_seed}/esm2_l${layer}"
    null_out="results_null_random_weightseed${weight_seed}/esm2_l${layer}"

    echo ""
    echo "=== C1 weight_seed=${weight_seed} layer=${layer} start $(date) ==="

    if [ ! -f "${out_layer}/META.json" ]; then
      "${PY}" experiment_random_control.py \
        --ref-layer-dir "outputs_layerwise/esm2/layer_${layer}" \
        --model esm2 \
        --layer "${layer}" \
        --epochs "${EPOCHS}" \
        --weight-seed "${weight_seed}" \
        --out-layer-dir "${out_layer}"
    else
      echo "  Skipping ${out_layer} (META.json exists)"
    fi

    if [ ! -f "${concept_out}/summary.json" ]; then
      "${PY}" experiment_concept_f1.py \
        --layer-dir "${out_layer}" \
        --save-dir "${concept_out}"
    else
      echo "  Skipping ${concept_out} (summary.json exists)"
    fi

    if [ ! -f "${null_out}/summary.json" ]; then
      "${PY}" experiment_null_calibration.py \
        --layer-dir "${out_layer}" \
        --save-dir "${null_out}"
    else
      echo "  Skipping ${null_out} (summary.json exists)"
    fi

    echo "=== C1 weight_seed=${weight_seed} layer=${layer} done $(date) ==="
  done
done

echo ""
echo "=========================================================================="
echo "  TIER 1b C1 RANDOMIZED-WEIGHTS MULTI-SEED complete $(date)"
echo "=========================================================================="
