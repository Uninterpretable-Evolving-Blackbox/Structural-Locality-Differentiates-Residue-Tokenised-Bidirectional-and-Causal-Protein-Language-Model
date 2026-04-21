#!/usr/bin/env python3
"""
experiment_val_only_rita.py — H1/H2 on val-only 150 proteins: ESM-2 vs RITA.

Mirrors experiment_val_only_progen2.py. RITA is residue-level (1 token per
amino acid, no BPE), so sequential-locality is directly comparable to ESM-2
without inter-token correction — the clean bidirectional-vs-causal contrast.
"""

import argparse
import time
from pathlib import Path

import pandas as pd

from experiment_val_only_h1h2 import locality_val, cohens_d, mw_greater

OUT_LW = Path(__file__).resolve().parent / "outputs_layerwise"

LAYERS = {"esm2": [0, 8, 16, 24, 32], "rita": [0, 6, 12, 18, 23]}
MATCHED_PAIRS = [
    ("0%",    0,  0),
    ("25%",   8,  6),
    ("50%",  16, 12),
    ("75%",  24, 18),
    ("100%", 32, 23),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="analysis_results_valonly_rita")
    ap.add_argument("--n-shuffles", type=int, default=5)
    ap.add_argument("--n-jobs", type=int, default=-1)
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    for model in ("esm2", "rita"):
        for layer in LAYERS[model]:
            val_csv = OUT_LW / model / f"layer_{layer}" / "struct_seq_metrics_val.csv"
            if val_csv.exists():
                print(f"  [skip] {val_csv}")
                continue
            locality_val(model, layer, args.n_shuffles, args.n_jobs)
    print(f"\nAll layers done in {(time.time()-t0)/60:.1f} min")

    rows = []
    for depth_label, esm_l, r_l in MATCHED_PAIRS:
        esm_all = pd.read_csv(OUT_LW / "esm2" / f"layer_{esm_l}" / "struct_seq_metrics.csv")
        r_all   = pd.read_csv(OUT_LW / "rita" / f"layer_{r_l}"  / "struct_seq_metrics.csv")
        esm_val = pd.read_csv(OUT_LW / "esm2" / f"layer_{esm_l}" / "struct_seq_metrics_val.csv")
        r_val   = pd.read_csv(OUT_LW / "rita" / f"layer_{r_l}"  / "struct_seq_metrics_val.csv")
        for split, (e, r) in [("all", (esm_all, r_all)), ("val", (esm_val, r_val))]:
            d_h1 = cohens_d(e.struct_delta.values, r.struct_delta.values)
            _, ph1 = mw_greater(e.struct_delta.values, r.struct_delta.values)
            rows.append(dict(depth=depth_label, hypothesis="H1", split=split,
                             cohens_d=d_h1, MW_p=float(ph1),
                             significant=bool(ph1 < 0.05 and d_h1 > 0),
                             esm2_layer=esm_l, rita_layer=r_l))
            d_h2 = cohens_d(r.seq_delta.values, e.seq_delta.values)
            _, ph2 = mw_greater(r.seq_delta.values, e.seq_delta.values)
            rows.append(dict(depth=depth_label, hypothesis="H2", split=split,
                             cohens_d=d_h2, MW_p=float(ph2),
                             significant=bool(ph2 < 0.05 and d_h2 > 0),
                             esm2_layer=esm_l, rita_layer=r_l))
    df = pd.DataFrame(rows)
    csv_out = out_dir / "val_only_h1h2_rita.csv"
    df.to_csv(csv_out, index=False)
    print(f"\nWritten → {csv_out}")

    lines = ["ESM-2 vs RITA H1/H2 — val-only (150 proteins) vs all (1500)\n",
             "=" * 72 + "\n"]
    for hyp in ("H1", "H2"):
        sub = df[df.hypothesis == hyp]
        n_all = int(sub[sub.split == "all"].significant.sum())
        n_val = int(sub[sub.split == "val"].significant.sum())
        lines.append(f"  {hyp}: {n_all}/5 (all) and {n_val}/5 (val) significant.\n")
    (out_dir / "val_only_h1h2_rita.txt").write_text("".join(lines))


if __name__ == "__main__":
    main()
