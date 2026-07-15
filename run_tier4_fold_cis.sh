#!/usr/bin/env bash
# run_tier4_fold_cis.sh — fold-level CI jobs for paper tables.
#
# Runs the existing robustness CI engines at SCOPe-fold cluster level:
#   - compute_cis_v2.py: L_seq, threshold variants, ProtT5 H2/H3-style CIs
#   - compute_cis_v3_optimized.py: cutoff/window sensitivity CIs
#
# Both scripts write *_fold.csv files, preserving the older protein-level files.
set -euo pipefail

cd "$(dirname "$0")"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export CLUSTER_LEVEL=fold
export CLUSTER_TWO_STAGE=1

PY="${PY:-./.venv/bin/python}"

csv_rows() {
  "${PY}" - "$1" <<'PY'
import sys
from pathlib import Path
import pandas as pd
p = Path(sys.argv[1])
print(len(pd.read_csv(p)) if p.exists() else -1)
PY
}

v2_complete() {
  [ "$(csv_rows outputs_robustness/v2_cis_pair_esm_rita_fold.csv)" -eq 90 ] &&
  [ "$(csv_rows outputs_robustness/v2_cis_pair_pt5_fold.csv)" -eq 36 ] &&
  [ "$(csv_rows outputs_robustness/v2_cis_trajectory_fold.csv)" -eq 36 ]
}

v3_complete() {
  [ "$(csv_rows outputs_robustness/v3opt_cis_val_sweeps_fold.csv)" -eq 72 ]
}

echo "=========================================================================="
echo "  TIER 4 FOLD-LEVEL CI JOBS start $(date)"
echo "=========================================================================="

if v2_complete; then
  echo "  Skipping compute_cis_v2.py: fold outputs already complete"
else
  echo "  Running compute_cis_v2.py at CLUSTER_LEVEL=fold"
  "${PY}" -u outputs_robustness/compute_cis_v2.py 2>&1 | tee outputs_robustness/compute_cis_v2_fold.log
fi

if v3_complete; then
  echo "  Skipping compute_cis_v3_optimized.py: fold outputs already complete"
else
  echo "  Running compute_cis_v3_optimized.py at CLUSTER_LEVEL=fold"
  "${PY}" -u outputs_robustness/compute_cis_v3_optimized.py 2>&1 | tee outputs_robustness/compute_cis_v3_optimized_fold.log
fi

echo ""
echo "=========================================================================="
echo "  TIER 4 FOLD-LEVEL CI JOBS complete $(date)"
echo "=========================================================================="
