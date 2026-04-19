#!/usr/bin/env python3
"""
aggregate_seeds.py — combine multi-seed pipeline runs into a single summary
==============================================================================

After running the pipeline at SAE_SEED ∈ {42, 43, 44} (three independent SAE
training runs with different torch RNG seeds, same protein-level holdout, same
hyperparameters), this script reads:

    outputs_layerwise/         (seed 42)
    outputs_layerwise_seed43/  (seed 43)
    outputs_layerwise_seed44/  (seed 44)

…and produces a cross-seed summary with mean ± std for every (model, layer) of:
    train_explained_variance
    val_explained_variance
    ev_gap
    struct_delta mean / std (per-layer)
    seq_delta mean / std (per-layer)

These are the load-bearing numbers for the paper's H1–H5 claims; with three
seeds we can finally report them as `mean ± std` instead of single-point.

Outputs:
    analysis_results_multiseed/cross_seed_meta.csv     (per-layer EVs across seeds)
    analysis_results_multiseed/cross_seed_summary.csv  (mean±std per layer)
    analysis_results_multiseed/cross_seed_struct_seq.csv  (cpu_stage outputs aggregated)
    analysis_results_multiseed/cross_seed_h1h2.csv     (H1/H2 effect sizes per seed)

Usage:
    python aggregate_seeds.py --seeds 42 43 44
    python aggregate_seeds.py --seeds 42 43 44 --out analysis_results_multiseed
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def root_for_seed(seed: int) -> Path:
    """Map a seed to the run's output directory."""
    if seed == 42:
        return Path("outputs_layerwise")
    return Path(f"outputs_layerwise_seed{seed}")


def collect_meta_evs(seeds: list) -> pd.DataFrame:
    """Per-layer (train_ev, val_ev, gap) for every seed."""
    rows = []
    for seed in seeds:
        root = root_for_seed(seed)
        if not root.exists():
            print(f"  ⚠ {root} not found, skipping seed {seed}")
            continue
        for model_dir in sorted(root.iterdir()):
            if not model_dir.is_dir():
                continue
            for layer_dir in sorted(model_dir.iterdir()):
                meta_path = layer_dir / "META.json"
                if not meta_path.exists():
                    continue
                m = json.loads(meta_path.read_text())
                rows.append({
                    "seed": seed,
                    "model": model_dir.name,
                    "layer": int(layer_dir.name.split("_")[-1]),
                    "train_ev": m.get("train_explained_variance"),
                    "val_ev": m.get("val_explained_variance"),
                    "ev_gap": m.get("ev_gap"),
                })
    return pd.DataFrame(rows)


def collect_struct_seq(seeds: list) -> pd.DataFrame:
    """Per-feature struct_delta and seq_delta for every (seed, model, layer)."""
    rows = []
    for seed in seeds:
        root = root_for_seed(seed)
        if not root.exists():
            continue
        for model_dir in sorted(root.iterdir()):
            if not model_dir.is_dir():
                continue
            for layer_dir in sorted(model_dir.iterdir()):
                ss_path = layer_dir / "struct_seq_metrics.csv"
                if not ss_path.exists():
                    continue
                ss = pd.read_csv(ss_path)
                rows.append({
                    "seed": seed,
                    "model": model_dir.name,
                    "layer": int(layer_dir.name.split("_")[-1]),
                    "n_features": len(ss),
                    "struct_delta_mean": float(ss["struct_delta"].mean()),
                    "struct_delta_std":  float(ss["struct_delta"].std()),
                    "struct_pct_gt0":    float((ss["struct_delta"] > 0).mean() * 100),
                    "seq_delta_mean":    float(ss["seq_delta"].mean()),
                    "seq_delta_std":     float(ss["seq_delta"].std()),
                })
    return pd.DataFrame(rows)


def collect_h1h2(seeds: list) -> pd.DataFrame:
    """H1/H2 cohen's d and p-values for every seed."""
    rows = []
    for seed in seeds:
        suffix = "" if seed == 42 else f"_seed{seed}"
        h1h2_path = Path(f"analysis_results{suffix}/comparison/H1_H2_all_depths.csv")
        if not h1h2_path.exists():
            print(f"  ⚠ {h1h2_path} not found")
            continue
        df = pd.read_csv(h1h2_path)
        df["seed"] = seed
        rows.append(df)
    if rows:
        return pd.concat(rows, ignore_index=True)
    return pd.DataFrame()


def summarize(df: pd.DataFrame, group_cols: list, value_cols: list) -> pd.DataFrame:
    """Mean ± std across seeds for each (group_cols) combo."""
    agg = {col: ["mean", "std", "count"] for col in value_cols}
    out = df.groupby(group_cols).agg(agg).reset_index()
    # Flatten the MultiIndex columns
    out.columns = [
        "_".join(c).rstrip("_") if isinstance(c, tuple) else c
        for c in out.columns
    ]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--out", default="analysis_results_multiseed")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Aggregating across seeds: {args.seeds}")
    print(f"Output: {out_dir}")
    print()

    # ── 1. Per-layer EVs ──────────────────────────────────────
    print("[1/3] Collecting per-layer META.json...")
    meta_df = collect_meta_evs(args.seeds)
    if meta_df.empty:
        print("  ❌ No META.json found in any seed directory")
        return
    print(f"  Loaded {len(meta_df)} (seed, model, layer) rows from {meta_df['seed'].nunique()} seeds")
    meta_df.to_csv(out_dir / "cross_seed_meta.csv", index=False)

    meta_summary = summarize(
        meta_df,
        group_cols=["model", "layer"],
        value_cols=["train_ev", "val_ev", "ev_gap"],
    )
    meta_summary.to_csv(out_dir / "cross_seed_summary.csv", index=False)
    print(f"  ✓ {out_dir / 'cross_seed_summary.csv'}")
    print()
    print(meta_summary.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

    # ── 2. Per-layer struct/seq deltas ────────────────────────
    print()
    print("[2/3] Collecting cpu_stage struct_seq_metrics.csv...")
    ss_df = collect_struct_seq(args.seeds)
    if ss_df.empty:
        print("  ⚠ No struct_seq_metrics.csv found")
    else:
        ss_df.to_csv(out_dir / "cross_seed_struct_seq.csv", index=False)
        ss_summary = summarize(
            ss_df,
            group_cols=["model", "layer"],
            value_cols=["struct_delta_mean", "struct_pct_gt0",
                        "seq_delta_mean"],
        )
        ss_summary.to_csv(out_dir / "cross_seed_struct_seq_summary.csv", index=False)
        print(f"  ✓ {out_dir / 'cross_seed_struct_seq_summary.csv'}")

    # ── 3. H1/H2 effect sizes ─────────────────────────────────
    print()
    print("[3/3] Collecting H1/H2 effect sizes from analyze_hypotheses outputs...")
    h1h2 = collect_h1h2(args.seeds)
    if h1h2.empty:
        print("  ⚠ No H1_H2_all_depths.csv found per seed — re-run analyze_hypotheses on each seed dir first")
    else:
        h1h2.to_csv(out_dir / "cross_seed_h1h2.csv", index=False)
        h1h2_summary = summarize(
            h1h2,
            group_cols=["depth", "hypothesis"],
            value_cols=["cohens_d", "MW_p"],
        )
        h1h2_summary.to_csv(out_dir / "cross_seed_h1h2_summary.csv", index=False)
        print(f"  ✓ {out_dir / 'cross_seed_h1h2_summary.csv'}")
        print()
        print("H1/H2 cohen's d across seeds (mean ± std):")
        print(h1h2_summary.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))

    print()
    print("✅ Done.")


if __name__ == "__main__":
    main()
