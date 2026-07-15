#!/usr/bin/env bash
# run_multiseed_full_grid.sh — densify SAE seeds 43/44 to the 9-depth grid.
#
# This fills the missing intermediate layers for the existing multi-seed runs:
#   outputs_layerwise_seed43/
#   outputs_layerwise_seed44/
#
# Existing layer directories with META.json are skipped by run_unsupervised.py,
# and existing CPU-stage outputs are skipped by run_all.sh. The commands below
# therefore add only missing layers without overwriting the seed-42 run.
set -euo pipefail

cd "$(dirname "$0")"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

ESM2_LAYERS_FULL="0,4,8,12,16,20,24,28,32"
RITA_LAYERS_FULL="0,3,6,9,12,15,18,21,23"
PROTT5_LAYERS_FULL="0,3,6,9,12,15,18,21,23"
PROTGPT2_LAYERS_FULL="0,4,9,13,18,22,27,31,35"

run_model() {
    local seed="$1"
    local model="$2"
    local layer_env_name="$3"
    local layers="$4"
    local log_file="run_multiseed_full_grid_seed${seed}_${model}.log"

    echo ""
    echo "=========================================================================="
    echo "  seed=${seed} model=${model} start $(date)"
    echo "  layers=${layers}"
    echo "  log=${log_file}"
    echo "=========================================================================="

    env SAE_SEED="${seed}" RUN_SUFFIX="_seed${seed}" "${layer_env_name}=${layers}" \
        ./run_all.sh "${model}" 2>&1 | tee "${log_file}"

    echo "  seed=${seed} model=${model} done $(date)"
}

echo "=========================================================================="
echo "  MULTI-SEED FULL-GRID DENSIFICATION start $(date)"
echo "=========================================================================="

for seed in 43 44; do
    run_model "${seed}" esm2       ESM2_LAYERS       "${ESM2_LAYERS_FULL}"
    run_model "${seed}" rita       RITA_LAYERS       "${RITA_LAYERS_FULL}"
    run_model "${seed}" prott5_enc PROTT5_ENC_LAYERS "${PROTT5_LAYERS_FULL}"
    run_model "${seed}" prott5_dec PROTT5_DEC_LAYERS "${PROTT5_LAYERS_FULL}"
    run_model "${seed}" protgpt2   PROTGPT2_LAYERS   "${PROTGPT2_LAYERS_FULL}"
done

echo ""
echo "=========================================================================="
echo "  Aggregating seeds 42/43/44 start $(date)"
echo "=========================================================================="
.venv/bin/python aggregate_seeds.py --seeds 42 43 44 --out analysis_results_multiseed

echo ""
echo "=========================================================================="
echo "  MULTI-SEED FULL-GRID DENSIFICATION complete $(date)"
echo "=========================================================================="
