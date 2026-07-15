#!/usr/bin/env bash
# regen_newpair.sh — regenerate the iCloud-evicted parts of the new PLM pair
# analysis and finish ProGen2, then run the H1 hypothesis analysis.
#
# Surgical (not a full re-run): all Z.npy are local, so we only recompute the
# layers whose required analysis CSVs were evicted, plus the ProGen2 layers that
# never got analyzed. analyze_hypotheses only reads struct_seq_metrics.csv,
# feature_interpretability.csv, fold_enrichment.csv per layer.
set -uo pipefail
cd "$(dirname "$0")"
source env_local_caches.sh

PY=./.venv/bin/python
ROOT=outputs_layerwise_newpair
CPU_ARGS=(--model-type residue
          --features-csv cache/residue_features.csv
          --pdb-dir cache/pdb_files
          --fasta-path cache/scope_40.fa
          --n-shuffles 5 --sweep-topk)

echo "=========================================================================="
echo "  REGEN NEW PLM PAIR start $(date)"
echo "=========================================================================="

echo ""
echo ">>> ProtBert-BFD: recompute evicted layers (7 18 29)  $(date)"
for L in 7 18 29; do
  echo "----- protbert_bfd/layer_$L  $(date) -----"
  "$PY" cpu_stage.py --layer-dir "$ROOT/protbert_bfd/layer_$L" "${CPU_ARGS[@]}"
done

echo ""
echo ">>> ProGen2: finish unanalyzed layers (3 6 23 26)  $(date)"
for L in 3 6 23 26; do
  echo "----- progen2/layer_$L  $(date) -----"
  "$PY" cpu_stage.py --layer-dir "$ROOT/progen2/layer_$L" "${CPU_ARGS[@]}"
done

echo ""
echo ">>> H1 hypothesis analysis on new PLM pair  $(date)"
"$PY" analyze_hypotheses.py --root "$ROOT" --save-dir analysis_results_newpair

echo ""
echo "=========================================================================="
echo "  REGEN NEW PLM PAIR DONE $(date)"
echo "=========================================================================="
