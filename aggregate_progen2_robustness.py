#!/usr/bin/env python3
"""
aggregate_progen2_robustness.py — consolidate the ProGen2 5-run table.

Reads the five per-variant H1/H2 CSVs produced by the overnight chain and
writes one master table: per depth × per axis, Cohen's d for H1 and H2
clean (residue-level, no BPE correction).
"""

from pathlib import Path

import pandas as pd

OUT = Path("analysis_results_master_progen2"); OUT.mkdir(exist_ok=True)

RUNS = {
    "main":    "analysis_results_progen2/comparison/H1_H2_esm2_vs_progen2.csv",
    "seed43":  "analysis_results_progen2_seed43/comparison/H1_H2_esm2_vs_progen2.csv",
    "seed44":  "analysis_results_progen2_seed44/comparison/H1_H2_esm2_vs_progen2.csv",
    "k=128":   "analysis_results_progen2_k128/comparison/H1_H2_esm2_vs_progen2.csv",
    "split99": "analysis_results_progen2_split99/comparison/H1_H2_esm2_vs_progen2.csv",
}


def main():
    frames = []
    for run, path in RUNS.items():
        p = Path(path)
        if not p.exists():
            print(f"  missing: {p}")
            continue
        df = pd.read_csv(p).assign(run=run)
        frames.append(df)
    big = pd.concat(frames, ignore_index=True)
    big.to_csv(OUT / "raw.csv", index=False)

    depth_order = ["0%", "25%", "50%", "75%", "100%"]
    runs_order = [r for r in ("main", "seed43", "seed44", "k=128", "split99")
                  if r in big.run.unique()]

    for hyp, tag in [("H1", "h1"), ("H2", "h2")]:
        sub = big[big.hypothesis == hyp]
        piv = sub.pivot(index="depth", columns="run", values="cohens_d")
        piv = piv[[r for r in runs_order if r in piv.columns]].reindex(depth_order)
        piv.to_csv(OUT / f"{tag}_cohens_d_5runs.csv")
        print(f"\n=== {hyp} (Cohen's d per run × depth) ===")
        print(piv.to_string(float_format=lambda v: f"{v:+6.3f}"))
        n_sig_cells = int((sub.significant).sum())
        n_tot = len(sub)
        print(f"  total significant: {n_sig_cells}/{n_tot}")

    print(f"\n  Written master tables → {OUT}/")


if __name__ == "__main__":
    main()
