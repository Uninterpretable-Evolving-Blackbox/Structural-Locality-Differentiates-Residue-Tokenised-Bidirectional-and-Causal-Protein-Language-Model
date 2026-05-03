#!/usr/bin/env python3
"""
experiment_h1_9depth.py — H1 (ESM-2 vs RITA structural locality) at the
densified 9-depth grid.

Uses the existing struct_seq_metrics.csv at each of the 9 ESM-2 layers
{0,4,8,12,16,20,24,28,32} and 9 RITA layers {0,3,6,9,12,15,18,21,23}.
Reports Cohen's d on `struct_delta` and a one-sided Mann-Whitney p
(ESM > RITA) at each of the 9 matched pairs.

Output:
  analysis_results/comparison/H1_esm2_vs_rita_9depth.csv
  analysis_results/comparison/H1_esm2_vs_rita_9depth_summary.txt
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent / "outputs_layerwise"
OUT_DIR = Path(__file__).resolve().parent / "analysis_results" / "comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# (pair index, ESM-2 rel_depth%, ESM-2 layer, RITA layer, RITA rel_depth%)
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


def cohens_d(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    pooled = np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2.0 + 1e-12)
    return float((a.mean() - b.mean()) / pooled)


def main():
    rows = []
    for idx, eD, eL, rL, rD in MATCHED_9:
        esm_csv  = ROOT / "esm2" / f"layer_{eL}"  / "struct_seq_metrics.csv"
        rita_csv = ROOT / "rita" / f"layer_{rL}" / "struct_seq_metrics.csv"
        if not (esm_csv.exists() and rita_csv.exists()):
            print(f"  skip pair {idx} (missing csv)")
            continue
        e = pd.read_csv(esm_csv).struct_delta.values
        r = pd.read_csv(rita_csv).struct_delta.values
        d = cohens_d(e, r)
        u, p = stats.mannwhitneyu(e, r, alternative="greater")
        rows.append(dict(
            pair=idx,
            esm2_layer=eL, esm2_rel_depth=eD,
            rita_layer=rL, rita_rel_depth=rD,
            esm2_mean=float(e.mean()), rita_mean=float(r.mean()),
            esm2_median=float(np.median(e)), rita_median=float(np.median(r)),
            cohens_d=d, MW_U=float(u), MW_p=float(p),
            n_esm=int(len(e)), n_rita=int(len(r)),
            significant=bool(p < 0.05 and d > 0),
            new=bool(eL in {4, 12, 20, 28}),
        ))

    df = pd.DataFrame(rows)
    csv_out = OUT_DIR / "H1_esm2_vs_rita_9depth.csv"
    df.to_csv(csv_out, index=False)
    print(f"\n✓ Wrote {csv_out}\n")
    print(df[["pair", "esm2_layer", "rita_layer", "esm2_rel_depth",
              "esm2_mean", "rita_mean", "cohens_d", "MW_p", "significant", "new"]]
          .to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

    # Text summary
    lines = [
        "H1 — ESM-2 vs RITA structural locality across 9 matched depths\n",
        "=" * 72 + "\n\n",
        "Matched pairs (ESM-2 has 33 blocks, RITA has 24, all pairs within ±4% rel. depth).\n\n",
    ]
    lines.append(df[["pair", "esm2_layer", "rita_layer",
                     "esm2_rel_depth", "rita_rel_depth",
                     "cohens_d", "MW_p", "significant"]]
                 .to_string(index=False,
                            float_format=lambda v: f"{v:+.4f}") + "\n\n")
    n_sig = int(df["significant"].sum())
    lines.append(f"H1 supported in {n_sig}/{len(df)} depth pairs "
                 f"(full 1,500-protein set; MW one-sided ESM-2 > RITA).\n")
    lines.append(f"\nCohen's d range: [{df.cohens_d.min():+.3f}, {df.cohens_d.max():+.3f}]\n")
    lines.append(f"Mean d across 9 pairs: {df.cohens_d.mean():+.3f}\n")
    summary_path = OUT_DIR / "H1_esm2_vs_rita_9depth_summary.txt"
    summary_path.write_text("".join(lines))
    print(f"\n✓ Summary → {summary_path}")


if __name__ == "__main__":
    main()
