#!/usr/bin/env python3
"""
experiment_h1_valonly_9depth.py — val-only H1 (ESM-2 vs RITA) at 9 matched
depths.

The original 5 matched depths already have struct_seq_metrics_val.csv
written by experiment_val_only_h1h2.py (ESM-2 L{0,8,16,24,32}) and
experiment_val_only_rita.py (RITA L{0,6,12,18,23}). This script runs
`locality_val` for the 4 new ESM-2 layers {4,12,20,28} and 4 new RITA
layers {3,9,15,21}, then computes Cohen's d + MW p at all 9 matched pairs
on the 150-protein held-out validation set.

Outputs:
  analysis_results/comparison/H1_esm2_vs_rita_9depth_valonly.csv
  analysis_results/comparison/H1_esm2_vs_rita_9depth_combined.csv
    (joins d(all) from H1_esm2_vs_rita_9depth.csv with d(val) from this run)
  analysis_results/comparison/H1_esm2_vs_rita_9depth_valonly_summary.txt
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from experiment_val_only_h1h2 import locality_val

ROOT = Path(__file__).resolve().parent
OUT_LW = ROOT / "outputs_layerwise"
OUT_DIR = ROOT / "analysis_results" / "comparison"

# Same 9-pair matching as experiment_h1_9depth.py
MATCHED_9 = [
    (1,   0.0, 0,   0,   0.0),
    (2,  12.5, 4,   3,  13.0),
    (3,  25.0, 8,   6,  26.1),
    (4,  37.5, 12,  9,  39.1),
    (5,  50.0, 16, 12,  52.2),
    (6,  62.5, 20, 15,  65.2),
    (7,  75.0, 24, 18,  78.3),
    (8,  87.5, 28, 21,  91.3),
    (9, 100.0, 32, 23, 100.0),
]

# Layers added by the H5 densification that still need val-only CSVs
NEW_ESM  = [4, 12, 20, 28]
NEW_RITA = [3, 9, 15, 21]


def cohens_d(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    pooled = np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2.0 + 1e-12)
    return float((a.mean() - b.mean()) / pooled)


def main():
    t_start = time.time()

    # ── Stage 1: compute val-only CSVs for the 8 new layers ──
    for model, layers in [("esm2", NEW_ESM), ("rita", NEW_RITA)]:
        for L in layers:
            p = OUT_LW / model / f"layer_{L}" / "struct_seq_metrics_val.csv"
            if p.exists():
                print(f"  [skip] {p} (exists)")
                continue
            locality_val(model, L, n_shuffles=5, n_jobs=-1)

    # Also double-check that the existing 5 depths already have val csvs;
    # if not, run them (they should — from the earlier val-only runs).
    for model, layers in [("esm2", [0, 8, 16, 24, 32]), ("rita", [0, 6, 12, 18, 23])]:
        for L in layers:
            p = OUT_LW / model / f"layer_{L}" / "struct_seq_metrics_val.csv"
            if not p.exists():
                print(f"  [backfill] {p} missing — running locality_val")
                locality_val(model, L, n_shuffles=5, n_jobs=-1)

    print(f"\nLocality_val stage done in {(time.time()-t_start)/60:.1f} min")

    # ── Stage 2: compute H1 d(val) at 9 pairs ──
    val_rows = []
    for idx, eD, eL, rL, rD in MATCHED_9:
        esm_p  = OUT_LW / "esm2" / f"layer_{eL}"  / "struct_seq_metrics_val.csv"
        rita_p = OUT_LW / "rita" / f"layer_{rL}" / "struct_seq_metrics_val.csv"
        if not (esm_p.exists() and rita_p.exists()):
            print(f"  skip pair {idx} (missing val csv)")
            continue
        e = pd.read_csv(esm_p).struct_delta.values
        r = pd.read_csv(rita_p).struct_delta.values
        d = cohens_d(e, r)
        u, p_mw = stats.mannwhitneyu(e, r, alternative="greater")
        val_rows.append(dict(
            pair=idx, esm2_layer=eL, rita_layer=rL,
            esm2_rel_depth=eD, rita_rel_depth=rD,
            esm2_val_mean=float(e.mean()), rita_val_mean=float(r.mean()),
            cohens_d_val=d, MW_p_val=float(p_mw),
            n_esm_val=int(len(e)), n_rita_val=int(len(r)),
            significant_val=bool(p_mw < 0.05 and d > 0),
            new=bool(eL in NEW_ESM),
        ))
    val_df = pd.DataFrame(val_rows)
    val_df.to_csv(OUT_DIR / "H1_esm2_vs_rita_9depth_valonly.csv", index=False)

    # ── Stage 3: merge with d(all) from experiment_h1_9depth.py ──
    all_csv = OUT_DIR / "H1_esm2_vs_rita_9depth.csv"
    if all_csv.exists():
        all_df = pd.read_csv(all_csv)
        merged = all_df.merge(
            val_df[["pair", "cohens_d_val", "MW_p_val",
                    "significant_val", "esm2_val_mean", "rita_val_mean"]],
            on="pair", how="outer")
        merged.to_csv(OUT_DIR / "H1_esm2_vs_rita_9depth_combined.csv", index=False)
        print(f"\n=== H1 at 9 depths — full + val ===")
        print(merged[["pair", "esm2_layer", "rita_layer", "esm2_rel_depth",
                      "cohens_d", "cohens_d_val",
                      "significant", "significant_val"]]
              .to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

    # ── Text summary ──
    lines = [
        "H1 val-only — ESM-2 vs RITA across 9 matched depths\n",
        "=" * 72 + "\n\n",
        "150-protein held-out validation set.\n\n",
    ]
    lines.append(val_df[["pair", "esm2_layer", "rita_layer",
                         "esm2_rel_depth", "rita_rel_depth",
                         "cohens_d_val", "MW_p_val", "significant_val"]]
                 .to_string(index=False, float_format=lambda v: f"{v:+.4f}") + "\n\n")
    n_sig_val = int(val_df["significant_val"].sum())
    lines.append(f"H1 replicates on val in {n_sig_val}/{len(val_df)} pairs "
                 f"(one-sided MW p<0.05 & d>0).\n")
    (OUT_DIR / "H1_esm2_vs_rita_9depth_valonly_summary.txt").write_text("".join(lines))
    print(f"\nTotal wall: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
