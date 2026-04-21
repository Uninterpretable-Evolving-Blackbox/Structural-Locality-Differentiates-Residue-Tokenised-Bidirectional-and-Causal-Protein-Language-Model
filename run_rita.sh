#!/usr/bin/env bash
# Launch RITA-only pipeline: extract + train SAEs + cpu_stage + H1/H2 test.
# HF offline OFF so the ~2.7 GB RITA weights can be fetched on first run.
set -euo pipefail

cd "$(dirname "$0")"
unset HF_HUB_OFFLINE
unset TRANSFORMERS_OFFLINE

echo "=== START $(date) ==="
echo "--- Stage 1: RITA extraction + SAE training + cpu_stage ---"
./run_all.sh rita

echo ""
echo "--- Stage 2: ESM-2 vs RITA clean H1/H2 test ---"
./.venv/bin/python experiment_h1h2_rita.py \
    --outputs-dir outputs_layerwise \
    --out        analysis_results_rita

echo "=== END $(date) ==="
