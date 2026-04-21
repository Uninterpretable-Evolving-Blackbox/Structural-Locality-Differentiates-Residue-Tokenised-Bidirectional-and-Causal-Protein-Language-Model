#!/usr/bin/env python3
"""
experiment_h1h2_rita.py — Clean H1/H2 test: ESM-2 vs RITA.

RITA has residue-level tokenization (1 token per amino acid), so its
sequential-locality metric is directly comparable to ESM-2 without any
BPE inter-token correction. This script runs the same Mann-Whitney tests
as the ESM-2 vs ProtGPT2 comparison in analyze_hypotheses.py, but on the
residue-level autoregressive model — giving a fair bidirectional-vs-causal
comparison.

Inputs:
  outputs_layerwise/{esm2,rita}/layer_{N}/struct_seq_metrics.csv

Output:
  analysis_results_rita/comparison/
    H1_H2_esm2_vs_rita.csv       per-depth Cohen's d + MW p + verdicts
    hypothesis_report_rita.txt   human-readable summary
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent

# (depth_label, esm2_layer, rita_layer)
MATCHED_PAIRS = [
    ("0%",    0,  0),
    ("25%",   8,  6),
    ("50%",  16, 12),
    ("75%",  24, 18),
    ("100%", 32, 23),
]


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    pooled = np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2.0 + 1e-12)
    return float((a.mean() - b.mean()) / pooled)


def run_test(a: np.ndarray, b: np.ndarray, name_a: str, name_b: str, metric: str) -> dict:
    d = cohens_d(a, b)
    u, p = stats.mannwhitneyu(a, b, alternative="greater")
    return dict(
        metric=metric, A=name_a, B=name_b,
        A_mean=float(a.mean()), A_median=float(np.median(a)),
        B_mean=float(b.mean()), B_median=float(np.median(b)),
        cohens_d=d, MW_U=float(u), MW_p=float(p),
        significant=bool(p < 0.05 and d > 0),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-dir", default="outputs_layerwise")
    ap.add_argument("--out", default="analysis_results_rita")
    args = ap.parse_args()

    outputs_root = (ROOT / args.outputs_dir) if not Path(args.outputs_dir).is_absolute() else Path(args.outputs_dir)
    out_dir = Path(args.out) / "comparison"; out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for depth_label, esm_l, pg_l in MATCHED_PAIRS:
        esm_csv = outputs_root / "esm2"    / f"layer_{esm_l}" / "struct_seq_metrics.csv"
        pg_csv  = outputs_root / "rita" / f"layer_{pg_l}"  / "struct_seq_metrics.csv"
        if not esm_csv.exists() or not pg_csv.exists():
            print(f"  missing CSV at depth {depth_label} (esm={esm_csv.exists()}, "
                  f"rita={pg_csv.exists()}) — skipping")
            continue
        esm = pd.read_csv(esm_csv)
        pg  = pd.read_csv(pg_csv)

        # H1: ESM-2 structural > RITA structural
        r = run_test(esm.struct_delta.values, pg.struct_delta.values,
                     "esm2", "rita", "struct_delta")
        r.update(depth=depth_label, hypothesis="H1",
                 esm2_layer=esm_l, rita_layer=pg_l)
        rows.append(r)

        # H2 (residue-level clean): RITA sequential > ESM-2 sequential
        r = run_test(pg.seq_delta.values, esm.seq_delta.values,
                     "rita", "esm2", "seq_delta")
        r.update(depth=depth_label, hypothesis="H2",
                 esm2_layer=esm_l, rita_layer=pg_l)
        rows.append(r)

    df = pd.DataFrame(rows)
    csv_out = out_dir / "H1_H2_esm2_vs_rita.csv"
    df.to_csv(csv_out, index=False)
    print(f"\nWritten → {csv_out}")

    # Readable report
    lines = [
        "ESM-2 vs RITA — residue-level bidirectional-vs-causal H1/H2\n",
        "=" * 72 + "\n\n",
        "RITA uses 1-token-per-residue (no BPE). Sequential-locality Δ is\n",
        "therefore directly comparable to ESM-2 without any inter-token correction.\n\n",
        df[["depth", "hypothesis", "metric", "A_mean", "B_mean",
            "cohens_d", "MW_p", "significant"]].to_string(
            index=False, float_format=lambda v: f"{v:+.4f}" if isinstance(v, float) else str(v)) + "\n\n",
    ]
    for hyp in ("H1", "H2"):
        sub = df[df.hypothesis == hyp]
        n_sig = int(sub.significant.sum())
        n = len(sub)
        lines.append(f"{hyp}: {n_sig}/{n} depths significant.\n")
    report = out_dir / "hypothesis_report_rita.txt"
    with open(report, "w") as f:
        f.writelines(lines)
    print(f"Report  → {report}")


if __name__ == "__main__":
    main()
