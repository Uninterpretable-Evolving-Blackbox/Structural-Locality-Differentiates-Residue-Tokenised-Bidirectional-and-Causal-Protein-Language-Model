#!/usr/bin/env python3
"""
summarize_concept_f1_crossmodel.py — 4-model Concept-F1 family contrast
=======================================================================

At each pre-registered relative depth, compare:
  bidirectional mean (ESM-2 + ProtBert-BFD) vs causal mean (RITA + ProGen2)
on mean top test concept-F1 per concept (seed-42 SAEs, protein split).

Writes results_concept_f1_crossmodel/summary.csv and summary.json.
"""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results_concept_f1_crossmodel"
OUT.mkdir(parents=True, exist_ok=True)

# (rel_depth_label, esm2, rita, protbert_bfd, progen2)
MATCHED = [
    ("0", 0, 0, 0, 0),
    ("13", 4, 3, 4, 3),
    ("25", 8, 6, 7, 6),
    ("38", 12, 9, 11, 10),
    ("50", 16, 12, 14, 13),
    ("63", 20, 15, 18, 16),
    ("75", 24, 18, 22, 20),
    ("88", 28, 21, 25, 23),
    ("100", 32, 23, 29, 26),
]


def load_f1(model: str, layer: int) -> float | None:
    p = ROOT / "results_concept_f1" / f"{model}_l{layer}" / "summary.json"
    if not p.exists():
        return None
    return float(json.loads(p.read_text())["mean_top_test_f1_per_concept"])


def main():
    rows = []
    for rel, esm_l, rita_l, pb_l, pg_l in MATCHED:
        esm = load_f1("esm2", esm_l)
        rita = load_f1("rita", rita_l)
        pb = load_f1("protbert_bfd", pb_l)
        pg = load_f1("progen2", pg_l)
        bidir_vals = [v for v in (esm, pb) if v is not None]
        causal_vals = [v for v in (rita, pg) if v is not None]
        bidir_mean = sum(bidir_vals) / len(bidir_vals) if bidir_vals else None
        causal_mean = sum(causal_vals) / len(causal_vals) if causal_vals else None
        rows.append({
            "rel_depth_pct": rel,
            "esm2_layer": esm_l,
            "rita_layer": rita_l,
            "protbert_layer": pb_l,
            "progen2_layer": pg_l,
            "esm2_f1": esm,
            "rita_f1": rita,
            "protbert_f1": pb,
            "progen2_f1": pg,
            "bidir_mean_f1": bidir_mean,
            "causal_mean_f1": causal_mean,
            "bidir_gt_causal": (
                bidir_mean > causal_mean
                if bidir_mean is not None and causal_mean is not None else None
            ),
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "summary.csv", index=False)

    n_clean = int(df["bidir_gt_causal"].sum())
    n_total = int(df["bidir_gt_causal"].notna().sum())
    summary = {
        "n_depths": n_total,
        "n_bidir_gt_causal": n_clean,
        "depths_bidir_gt_causal": df.loc[df["bidir_gt_causal"] == True, "rel_depth_pct"].tolist(),
        "note": (
            "Seed-42 SAE only; protein-level concept val/test split. "
            "Family-level contrast (bidir pair vs causal pair), not objective alone."
        ),
        "rows": rows,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({
        "n_depths": n_total,
        "bidir_gt_causal": f"{n_clean}/{n_total}",
        "depths": summary["depths_bidir_gt_causal"],
    }, indent=2))
    print(f"Wrote {OUT / 'summary.csv'}")


if __name__ == "__main__":
    main()
