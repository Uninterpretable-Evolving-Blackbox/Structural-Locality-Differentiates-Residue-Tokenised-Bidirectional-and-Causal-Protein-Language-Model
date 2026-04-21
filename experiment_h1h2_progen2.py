#!/usr/bin/env python3
"""
experiment_h1h2_progen2.py — Clean H1/H2 at residue granularity: ESM-2 vs ProGen2.

ProGen2 uses 1-token-per-residue (no BPE merging), so its sequential-locality
metric is directly comparable to residue-level ESM-2 without the inter-token
correction that ProtGPT2 requires. This gives a fair bidirectional-vs-causal
contrast at matched relative depths.

Inputs:
  outputs_layerwise/{esm2,progen2}/layer_{N}/struct_seq_metrics.csv

Output:
  --out/comparison/
    H1_H2_esm2_vs_progen2.csv         per-depth Cohen's d + MW p
    hypothesis_report_progen2.txt     readable summary
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent

# (depth_label, esm2_layer, progen2_layer) — 27 ProGen2 blocks at 0/7/14/20/26
MATCHED_PAIRS = [
    ("0%",    0,  0),
    ("25%",   8,  7),
    ("50%",  16, 14),
    ("75%",  24, 20),
    ("100%", 32, 26),
]


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
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
    ap.add_argument("--out", default="analysis_results_progen2")
    args = ap.parse_args()

    outputs_root = Path(args.outputs_dir)
    if not outputs_root.is_absolute():
        outputs_root = ROOT / outputs_root
    out_dir = Path(args.out) / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for depth_label, esm_l, pg_l in MATCHED_PAIRS:
        esm_csv = outputs_root / "esm2"    / f"layer_{esm_l}" / "struct_seq_metrics.csv"
        pg_csv  = outputs_root / "progen2" / f"layer_{pg_l}"  / "struct_seq_metrics.csv"
        if not esm_csv.exists() or not pg_csv.exists():
            print(f"  missing CSV at depth {depth_label}: "
                  f"esm={esm_csv.exists()} progen2={pg_csv.exists()} — skip")
            continue
        esm = pd.read_csv(esm_csv)
        pg  = pd.read_csv(pg_csv)

        # H1: ESM-2 struct > ProGen2 struct
        r = run_test(esm.struct_delta.values, pg.struct_delta.values,
                     "esm2", "progen2", "struct_delta")
        r.update(depth=depth_label, hypothesis="H1",
                 esm2_layer=esm_l, progen2_layer=pg_l)
        rows.append(r)

        # H2: ProGen2 seq > ESM-2 seq (clean — no BPE correction needed)
        r = run_test(pg.seq_delta.values, esm.seq_delta.values,
                     "progen2", "esm2", "seq_delta")
        r.update(depth=depth_label, hypothesis="H2",
                 esm2_layer=esm_l, progen2_layer=pg_l)
        rows.append(r)

    df = pd.DataFrame(rows)
    csv_out = out_dir / "H1_H2_esm2_vs_progen2.csv"
    df.to_csv(csv_out, index=False)
    print(f"\nWritten → {csv_out}")

    lines = [
        "ESM-2 vs ProGen2 — residue-level bidirectional-vs-causal H1/H2\n",
        "=" * 72 + "\n\n",
        "ProGen2 uses 1-token-per-residue (no BPE merging). Sequential-locality\n",
        "Δ is therefore directly comparable to ESM-2 without any inter-token\n",
        "correction — the fair bidirectional-vs-causal test.\n\n",
        df[["depth", "hypothesis", "metric", "A_mean", "B_mean",
            "cohens_d", "MW_p", "significant"]].to_string(
            index=False, float_format=lambda v: f"{v:+.4f}" if isinstance(v, float) else str(v)) + "\n\n",
    ]
    for hyp in ("H1", "H2"):
        sub = df[df.hypothesis == hyp]
        lines.append(f"{hyp}: {int(sub.significant.sum())}/{len(sub)} depths significant.\n")
    report = out_dir / "hypothesis_report_progen2.txt"
    with open(report, "w") as f:
        f.writelines(lines)
    print(f"Report  → {report}")


if __name__ == "__main__":
    main()
