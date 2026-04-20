#!/usr/bin/env python3
"""
experiment_bpe_crossing_all_depths.py — H2 control at all 5 matched depths.

Invokes the existing experiment_bpe_correction.py (single-layer) for each of
the 5 ProtGPT2 depths paired with its ESM-2 matched depth, then aggregates
the per-layer `bpe_correction_summary.csv` files into a single table.

Output:
  results_bpe_crossing/
    depth_<label>/bpe_correction_per_feature.csv  (per-depth raw per-feature)
    depth_<label>/bpe_correction_summary.csv       (per-depth summary)
    h2_crossing_all_depths.csv                     (consolidated 5-row table)
    h2_crossing_all_depths.txt                     (human-readable verdict report)

Usage:
  python experiment_bpe_crossing_all_depths.py \
    --out results_bpe_crossing [--n-shuffles 5] [--n-jobs -1]
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

# (depth_label, ProtGPT2 layer, ESM-2 matched layer)
MATCHED_PAIRS = [
    ("0%",   0,  0),
    ("25%",  9,  8),
    ("50%", 18, 16),
    ("75%", 27, 24),
    ("100%", 35, 32),
]


def run_one(protgpt2_layer: int, esm2_layer: int, depth_label: str,
            out_dir: Path, outputs_root: Path,
            n_shuffles: int, n_jobs: int):
    """Shell out to the existing single-layer script for one depth."""
    layer_dir = outputs_root / "protgpt2" / f"layer_{protgpt2_layer}"
    esm2_dir = outputs_root / "esm2" / f"layer_{esm2_layer}"
    if not (layer_dir / "Z.npy").exists():
        raise FileNotFoundError(f"Missing {layer_dir/'Z.npy'}")
    if not (esm2_dir / "struct_seq_metrics.csv").exists():
        raise FileNotFoundError(f"Missing {esm2_dir/'struct_seq_metrics.csv'}")

    save_dir = out_dir / f"depth_{depth_label.rstrip('%')}pct"
    save_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(ROOT / "experiment_bpe_correction.py"),
        "--layer-dir", str(layer_dir),
        "--esm2-layer-dir", str(esm2_dir),
        "--n-shuffles", str(n_shuffles),
        "--n-jobs", str(n_jobs),
        "--save-dir", str(save_dir),
    ]
    print(f"\n{'=' * 72}")
    print(f"  Depth {depth_label}: ProtGPT2 L{protgpt2_layer} ↔ ESM-2 L{esm2_layer}")
    print(f"{'=' * 72}")
    print(" ".join(cmd))
    t0 = time.time()
    proc = subprocess.run(cmd, check=False)
    wall = time.time() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"depth {depth_label} failed (exit {proc.returncode})")
    print(f"  → {depth_label} complete in {wall/60:.1f} min")
    return save_dir


def aggregate(out_dir: Path):
    rows = []
    for depth_label, pg2_layer, esm_layer in MATCHED_PAIRS:
        save_dir = out_dir / f"depth_{depth_label.rstrip('%')}pct"
        summary_path = save_dir / "bpe_correction_summary.csv"
        if not summary_path.exists():
            print(f"  ⚠️  Missing {summary_path}")
            continue
        s = pd.read_csv(summary_path).iloc[0].to_dict()
        s["depth"] = depth_label
        s["protgpt2_layer"] = pg2_layer
        s["esm2_layer"] = esm_layer
        h2_d = s.get("h2_corrected_cohens_d")
        h2_p = s.get("h2_corrected_p")
        if h2_d is not None and not pd.isna(h2_d) and h2_p is not None and not pd.isna(h2_p):
            s["h2_verdict"] = ("SUPPORTED" if (h2_d > 0 and h2_p < 0.05)
                                else "NOT SUPPORTED")
        else:
            s["h2_verdict"] = "N/A"
        rows.append(s)

    df = pd.DataFrame(rows)
    cols = ["depth", "protgpt2_layer", "esm2_layer",
            "pct_neighbors_excluded",
            "original_mean", "corrected_mean", "pct_reduction",
            "wilcoxon_p", "esm2_mean",
            "h2_corrected_cohens_d", "h2_corrected_p", "h2_verdict"]
    cols = [c for c in cols if c in df.columns]
    df = df[cols]

    csv_out = out_dir / "h2_crossing_all_depths.csv"
    df.to_csv(csv_out, index=False)
    print(f"\n  Consolidated → {csv_out}")

    # Human-readable
    txt_out = out_dir / "h2_crossing_all_depths.txt"
    with open(txt_out, "w") as f:
        f.write("H2 BPE-crossing-boundary control — all 5 matched depths\n")
        f.write("=" * 72 + "\n\n")
        f.write("H2 = ProtGPT2 sequential Δ > ESM-2 sequential Δ (one-tailed MW).\n")
        f.write("Corrected: intra-BPE-token ±1/±2 residue pairs excluded.\n\n")
        f.write(df.to_string(index=False) + "\n\n")
        n_sup = int((df["h2_verdict"] == "SUPPORTED").sum())
        n_tot = int((df["h2_verdict"].isin(["SUPPORTED", "NOT SUPPORTED"])).sum())
        f.write(f"H2 (BPE-corrected) supported at {n_sup}/{n_tot} depths.\n")
    print(f"  Report       → {txt_out}")
    print(f"\n  H2 (BPE-corrected) supported at "
          f"{int((df['h2_verdict']=='SUPPORTED').sum())}/"
          f"{int(df['h2_verdict'].isin(['SUPPORTED','NOT SUPPORTED']).sum())} depths")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-dir", default="outputs_layerwise",
                    help="Source run dir (e.g. outputs_layerwise_seed43)")
    ap.add_argument("--out", default="results_bpe_crossing")
    ap.add_argument("--n-shuffles", type=int, default=5)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--aggregate-only", action="store_true",
                    help="Skip runs; just re-aggregate an existing out dir")
    args = ap.parse_args()

    outputs_root = Path(args.outputs_dir)
    if not outputs_root.is_absolute():
        outputs_root = ROOT / outputs_root
    if not outputs_root.exists():
        print(f"❌ --outputs-dir not found: {outputs_root}")
        sys.exit(2)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Source run:  {outputs_root}")
    print(f"  Writing to:  {out_dir}")

    if not args.aggregate_only:
        t0 = time.time()
        for depth_label, pg2_layer, esm_layer in MATCHED_PAIRS:
            run_one(pg2_layer, esm_layer, depth_label, out_dir, outputs_root,
                    args.n_shuffles, args.n_jobs)
        print(f"\nAll 5 depths done in {(time.time()-t0)/60:.1f} min")

    aggregate(out_dir)


if __name__ == "__main__":
    main()
