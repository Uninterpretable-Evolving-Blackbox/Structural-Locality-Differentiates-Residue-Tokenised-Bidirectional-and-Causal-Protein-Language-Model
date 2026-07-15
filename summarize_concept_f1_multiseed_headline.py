#!/usr/bin/env python3
"""
summarize_concept_f1_multiseed_headline.py — symmetric Concept-F1 seed table
=============================================================================

Aggregates headline Concept-F1 (mean top test F1 per concept) for:
  - Trained ESM-2 L16: SAE seeds {42, 43, 44}
  - Trained RITA L18:  SAE seeds {42, 43, 44}
  - Random weights ESM-2 L16: PLM weight-init seeds {0, 1, 2}

Writes results_concept_f1_multiseed_headline/summary.csv and summary.json.
Does not modify existing results_concept_f1/ trees.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results_concept_f1_multiseed_headline"
OUT.mkdir(parents=True, exist_ok=True)

TRAINED_SEEDS = [42, 43, 44]
RANDOM_WEIGHT_SEEDS = [0, 1, 2]


def trained_result_dir(model: str, layer: int, sae_seed: int) -> Path:
    if sae_seed == 42:
        return ROOT / "results_concept_f1" / f"{model}_l{layer}"
    return ROOT / f"results_concept_f1_seed{sae_seed}" / f"{model}_l{layer}"


def random_result_dir(layer: int, weight_seed: int) -> Path:
    if weight_seed == 0:
        return ROOT / "results_concept_f1_random" / f"esm2_l{layer}"
    return ROOT / f"results_concept_f1_random_weightseed{weight_seed}" / f"esm2_l{layer}"


def load_mean_f1(path: Path) -> float | None:
    summary = path / "summary.json"
    if not summary.exists():
        return None
    return float(json.loads(summary.read_text())["mean_top_test_f1_per_concept"])


def main():
    rows = []

    for sae_seed in TRAINED_SEEDS:
        for model, layer in [("esm2", 16), ("rita", 18)]:
            p = trained_result_dir(model, layer, sae_seed)
            val = load_mean_f1(p)
            rows.append({
                "arm": "trained",
                "model": model,
                "layer": layer,
                "seed_kind": "sae_init",
                "seed": sae_seed,
                "mean_top_test_f1": val,
                "result_dir": str(p.relative_to(ROOT)) if p.exists() else str(p),
                "present": val is not None,
            })

    for weight_seed in RANDOM_WEIGHT_SEEDS:
        p = random_result_dir(16, weight_seed)
        val = load_mean_f1(p)
        rows.append({
            "arm": "random_weights",
            "model": "esm2",
            "layer": 16,
            "seed_kind": "plm_weight_init",
            "seed": weight_seed,
            "mean_top_test_f1": val,
            "result_dir": str(p.relative_to(ROOT)) if p.exists() else str(p),
            "present": val is not None,
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "summary.csv", index=False)

    def agg_block(mask):
        sub = df[mask & df["present"]]
        if sub.empty:
            return {"n": 0, "mean": None, "std": None, "min": None, "max": None, "values": []}
        vals = sub["mean_top_test_f1"].to_numpy(dtype=float)
        return {
            "n": int(len(vals)),
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=0)) if len(vals) > 1 else 0.0,
            "min": float(vals.min()),
            "max": float(vals.max()),
            "values": [float(v) for v in vals],
        }

    summary = {
        "trained_esm2_l16_sae_seeds": agg_block(
            (df["arm"] == "trained") & (df["model"] == "esm2") & (df["layer"] == 16)),
        "trained_rita_l18_sae_seeds": agg_block(
            (df["arm"] == "trained") & (df["model"] == "rita") & (df["layer"] == 18)),
        "random_esm2_l16_weight_seeds": agg_block(df["arm"] == "random_weights"),
        "rows": rows,
        "note": (
            "Symmetric headline comparison: trained uses SAE init seeds 42/43/44; "
            "random uses PLM weight-init seeds 0/1/2 at ESM-2 L16 only. "
            "All runs use protein-level concept val/test split (matches legacy "
            "seed-42 and random-weight controls; 80 concepts scored)."
        ),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps({
        "trained_esm2_l16": summary["trained_esm2_l16_sae_seeds"],
        "trained_rita_l18": summary["trained_rita_l18_sae_seeds"],
        "random_esm2_l16": summary["random_esm2_l16_weight_seeds"],
        "missing": int((~df["present"]).sum()),
    }, indent=2))
    print(f"Wrote {OUT / 'summary.csv'} and {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()
