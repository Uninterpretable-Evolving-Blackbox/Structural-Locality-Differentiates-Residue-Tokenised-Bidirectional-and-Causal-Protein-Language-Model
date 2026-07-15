#!/usr/bin/env bash
# run_bootstrap_h1_newpair.sh — fold-cluster bootstrap for ProtBert-BFD vs ProGen2.
# Identical protocol to the canonical ESM-2 vs RITA H1 bootstrap:
#   B=1000, two-stage SCOPe-fold cluster resampling, min_active=0, full+val splits.
set -euo pipefail
cd "$(dirname "$0")/.."
source env_local_caches.sh

PY=./.venv/bin/python
echo "=========================================================================="
echo "  H1 fold bootstrap: ProtBert-BFD vs ProGen2  $(date)"
echo "=========================================================================="

"${PY}" -u outputs_robustness/compute_h1_bootstrap.py \
  --preset protbert_progen2 \
  --cluster-levels fold,protein \
  --min-active 0 \
  --depths all \
  --n-boot 1000

echo ""
echo "=========================================================================="
echo "  H1 fold bootstrap DONE  $(date)"
echo "=========================================================================="
