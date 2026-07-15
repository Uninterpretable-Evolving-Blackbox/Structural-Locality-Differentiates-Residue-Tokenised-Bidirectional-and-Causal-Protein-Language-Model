#!/usr/bin/env bash
# run_new_plm_pair_full_grid.sh — additional bidirectional/causal PLM pair.
#
# Adds:
#   - protbert_bfd: Rostlab/prot_bert_bfd (BERT/MLM, residue-tokenized)
#   - progen2: ProGen2-base if feasible, otherwise recorded fallback
#
# The smoke test is correctness-only and must pass before full 9-depth runs.
set -euo pipefail

cd "$(dirname "$0")"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

PY="${PY:-./.venv/bin/python}"
RUN_SUFFIX="${RUN_SUFFIX:-_newpair}"
mkdir -p results_new_plm_pair

echo "=========================================================================="
echo "  NEW PLM PAIR SMOKE + FULL GRID start $(date)"
echo "=========================================================================="

"${PY}" smoke_new_plm_pair.py 2>&1 | tee results_new_plm_pair/smoke_new_plm_pair.log

"${PY}" - <<'PY'
import json
from pathlib import Path
report = json.loads(Path("results_new_plm_pair/smoke_alignment_report.json").read_text())
if report.get("status") != "passed":
    raise SystemExit("Smoke report did not pass; refusing full run")
PY

source results_new_plm_pair/new_pair_env.sh

echo ""
echo "=========================================================================="
echo "  ProtBert-BFD full 9-depth run start $(date)"
echo "  PROTBERT_LAYERS=${PROTBERT_LAYERS}"
echo "=========================================================================="
RUN_SUFFIX="${RUN_SUFFIX}" MODEL=protbert_bfd PROTBERT_LAYERS="${PROTBERT_LAYERS}" \
  ./run_all.sh protbert_bfd 2>&1 | tee results_new_plm_pair/run_protbert_bfd_full_grid.log

echo ""
echo "=========================================================================="
echo "  ProGen2 full 9-depth run start $(date)"
echo "  PROGEN2_MODEL_NAME=${PROGEN2_MODEL_NAME}"
echo "  PROGEN2_LAYERS=${PROGEN2_LAYERS}"
echo "=========================================================================="
RUN_SUFFIX="${RUN_SUFFIX}" MODEL=progen2 PROGEN2_MODEL_NAME="${PROGEN2_MODEL_NAME}" \
  PROGEN2_LAYERS="${PROGEN2_LAYERS}" \
  ./run_all.sh progen2 2>&1 | tee results_new_plm_pair/run_progen2_full_grid.log

echo ""
echo "=========================================================================="
echo "  NEW PLM PAIR FULL GRID complete $(date)"
echo "  Outputs: outputs_layerwise${RUN_SUFFIX}/{protbert_bfd,progen2}/"
echo "=========================================================================="
