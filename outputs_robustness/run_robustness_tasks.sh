#!/usr/bin/env bash
# Run Tasks 2a + 2b, 3, 4 sequentially. Task 1 already done.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=./.venv/bin/python
unset HF_HUB_OFFLINE
unset TRANSFORMERS_OFFLINE

mark() { echo ""; echo "=== $1 $(date) ==="; }

mark "TASK 2 (embedding baseline) start"
$PY outputs_robustness/compute_embedding_baseline.py
mark "TASK 2 done"

mark "TASK 4 (threshold sensitivity) start"
$PY outputs_robustness/compute_threshold_sensitivity.py
mark "TASK 4 done"

# Re-enable offline for val-only (no PLM extraction needed there)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mark "TASK 3 (val-only recomputation) start"
$PY outputs_robustness/compute_val_only.py
mark "TASK 3 done"

mark "ALL TASKS COMPLETE"
