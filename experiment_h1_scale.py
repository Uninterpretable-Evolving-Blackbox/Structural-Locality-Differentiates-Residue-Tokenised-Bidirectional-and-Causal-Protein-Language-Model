#!/usr/bin/env python3
"""
experiment_h1_scale.py — H1 at a smaller PLM scale.

Compares ESM-2 t12 (35M, 12 blocks) vs RITA-s (85M, 12 blocks) at 5 matched
relative depths.  Both models are residue-tokenised, so the sequential-
locality test is directly comparable.

This complements the paper's main H1 (ESM-2 t33 650M vs RITA-l 680M): if
H1 survives a 7-18× parameter-count reduction, the architectural effect
cannot be explained by scale alone.

Output:
  analysis_results/comparison/H1_scale_small.{csv,txt}
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent

MATCHED_PAIRS = [
    ("0%",     0,   0),
    ("27%",    3,   3),
    ("55%",    6,   6),
    ("82%",    9,   9),
    ("100%",  11,  11),
]


def cohens_d(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    pooled = np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2.0 + 1e-12)
    return float((a.mean() - b.mean()) / pooled)


def run_test(a, b, na, nb, metric):
    d = cohens_d(a, b)
    u, p = stats.mannwhitneyu(a, b, alternative="greater")
    return dict(
        metric=metric, A=na, B=nb,
        A_mean=float(a.mean()), A_median=float(np.median(a)),
        B_mean=float(b.mean()), B_median=float(np.median(b)),
        cohens_d=d, MW_U=float(u), MW_p=float(p),
        significant=bool(p < 0.05 and d > 0),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-dir", default="outputs_layerwise")
    ap.add_argument("--out", default="analysis_results/comparison")
    args = ap.parse_args()

    outputs_root = Path(args.outputs_dir)
    if not outputs_root.is_absolute():
        outputs_root = ROOT / outputs_root
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for depth_label, e_l, r_l in MATCHED_PAIRS:
        e_csv = outputs_root / "esm2_small" / f"layer_{e_l}" / "struct_seq_metrics.csv"
        r_csv = outputs_root / "rita_small" / f"layer_{r_l}" / "struct_seq_metrics.csv"
        if not (e_csv.exists() and r_csv.exists()):
            print(f"  skip depth {depth_label} — missing csv "
                  f"(esm_small={e_csv.exists()}, rita_small={r_csv.exists()})")
            continue
        e = pd.read_csv(e_csv)
        r = pd.read_csv(r_csv)

        # H1: ESM-2 small struct > RITA small struct
        row = run_test(e.struct_delta.values, r.struct_delta.values,
                       "esm2_small", "rita_small", "struct_delta")
        row.update(depth=depth_label, hypothesis="H1",
                   esm2_small_layer=e_l, rita_small_layer=r_l)
        rows.append(row)

        # H2: RITA small seq > ESM-2 small seq
        row = run_test(r.seq_delta.values, e.seq_delta.values,
                       "rita_small", "esm2_small", "seq_delta")
        row.update(depth=depth_label, hypothesis="H2",
                   esm2_small_layer=e_l, rita_small_layer=r_l)
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_out = out_dir / "H1_scale_small.csv"
    df.to_csv(csv_out, index=False)
    print(f"\nWritten → {csv_out}")
    print(df[["depth", "hypothesis", "metric", "A_mean", "B_mean",
              "cohens_d", "MW_p", "significant"]]
          .to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

    lines = [
        "H1/H2 at smaller PLM scale — ESM-2 t12 (35M) vs RITA-s (85M)\n",
        "=" * 72 + "\n\n",
        "Paper's headline pair uses ESM-2 t33 (650M) vs RITA-l (680M).\n",
        "If H1 holds here too, the ESM-vs-RITA architectural effect is\n",
        "not an artifact of the particular parameter scale.\n\n",
    ]
    if len(df):
        lines.append(df[["depth", "hypothesis", "metric",
                         "A_mean", "B_mean", "cohens_d",
                         "MW_p", "significant"]].to_string(
                             index=False,
                             float_format=lambda v: f"{v:+.4f}") + "\n\n")
        for hyp in ("H1", "H2"):
            sub = df[df.hypothesis == hyp]
            if len(sub):
                n_sig = int(sub.significant.sum())
                lines.append(f"{hyp}: {n_sig}/{len(sub)} depths significant.\n")
    (out_dir / "H1_scale_small.txt").write_text("".join(lines))
    print(f"Summary → {out_dir / 'H1_scale_small.txt'}")


if __name__ == "__main__":
    main()
