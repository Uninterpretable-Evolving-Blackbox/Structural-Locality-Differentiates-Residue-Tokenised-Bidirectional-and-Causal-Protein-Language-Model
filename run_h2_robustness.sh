#!/usr/bin/env bash
# Run BPE-crossing H2' correction on the 3 existing non-main runs.
# Sequential (&&) — no CPU thrash. Writes to distinct result dirs.
set -euo pipefail

cd "$(dirname "$0")"
PY=./.venv/bin/python

echo "=== START $(date) ==="

$PY experiment_bpe_crossing_all_depths.py \
    --outputs-dir outputs_layerwise_seed43 \
    --out        results_bpe_crossing_seed43 \
    --n-shuffles 5

$PY experiment_bpe_crossing_all_depths.py \
    --outputs-dir outputs_layerwise_seed44 \
    --out        results_bpe_crossing_seed44 \
    --n-shuffles 5

$PY experiment_bpe_crossing_all_depths.py \
    --outputs-dir outputs_layerwise_k128 \
    --out        results_bpe_crossing_k128 \
    --n-shuffles 5

echo "=== END $(date) ==="
