#!/usr/bin/env python3
"""
Task 4 — fixed-activation-threshold sensitivity for H1.

Currently "active" = activation > 90th percentile of feature j.  For TopK
SAEs (k=256), most features are exactly 0 at most residues, so the 90th-
percentile threshold collapses to ~"any non-zero activation" — an unstable
operational definition.

This script re-runs H1 (ESM-2 vs RITA at 9 matched depths) under 3 alternative
absolute thresholds based on a global activation scale s = median of all
non-zero feature activations across the dataset:
   active = z > 0.5 * s
   active = z >       s
   active = z > 2   * s

Compares against the paper's quantile-based d.

Outputs (in outputs_robustness/):
  threshold_sensitivity.csv  — d at 4 thresholds × 9 depths
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs_robustness"
OUT.mkdir(parents=True, exist_ok=True)

from cpu_stage import (
    load_layer, load_ref_seqs,
    build_neighbor_graphs_residue_parallel,
    adj_list_to_sparse, build_protein_permutations,
)

MATCHED_PAIRS = [
    ("0",     0,   0),
    ("13",    4,   3),
    ("25",    8,   6),
    ("38",    12,  9),
    ("50",    16, 12),
    ("63",    20, 15),
    ("75",    24, 18),
    ("88",    28, 21),
    ("100",   32, 23),
]


def cohens_d(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    pooled = np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2.0 + 1e-12)
    return float((a.mean() - b.mean()) / pooled)


def locality_at_threshold(Z, A, deg, perm_indices, threshold):
    """Compute per-feature struct_delta given a fixed activation threshold."""
    n_res, n_feat = Z.shape
    Z = Z.astype(np.float32)
    sigma_j = Z.std(axis=0, ddof=0).astype(np.float32) + 1e-6
    nbr_sum = (A @ Z).astype(np.float32)
    has_nb = deg > 0
    smoothed = np.zeros_like(Z)
    smoothed[has_nb] = nbr_sum[has_nb] / deg[has_nb, None]
    global_mean = smoothed.mean(axis=0)

    # Threshold can be scalar (uniform) or per-feature
    if np.isscalar(threshold):
        thresh_j = np.full(n_feat, threshold, dtype=np.float32)
    else:
        thresh_j = threshold.astype(np.float32)

    active = Z > thresh_j[None, :]
    n_active = active.sum(axis=0).astype(np.float32)
    active_sum = (smoothed * active).sum(axis=0)
    active_mean = active_sum / np.maximum(n_active, 1)
    obs_d = (active_mean - global_mean) / sigma_j
    obs_d[n_active < 5] = 0.0

    # Shuffle
    shuf = np.zeros(n_feat, dtype=np.float32)
    for perm in perm_indices:
        Zp = Z[perm]
        nbrp = (A @ Zp).astype(np.float32)
        smp = np.zeros_like(Zp)
        smp[has_nb] = nbrp[has_nb] / deg[has_nb, None]
        active_p = Zp > thresh_j[None, :]
        n_active_p = active_p.sum(axis=0).astype(np.float32)
        as_p = (smp * active_p).sum(axis=0)
        am_p = as_p / np.maximum(n_active_p, 1)
        gm_p = smp.mean(axis=0)
        d_p = (am_p - gm_p) / sigma_j
        d_p[n_active_p < 5] = 0.0
        shuf += d_p
    shuf /= max(len(perm_indices), 1)
    return obs_d - shuf


def main():
    print("=" * 72)
    print("  Task 4 — fixed-activation-threshold sensitivity (H1)")
    print("=" * 72)

    # Shared adjacency
    layer0_dir = ROOT / "outputs_layerwise/esm2/layer_0"
    Z0, uids, lengths = load_layer(layer0_dir)
    res_lengths = lengths.astype(np.int32)
    n_res_total = int(res_lengths.sum())
    ref_seqs = load_ref_seqs(layer0_dir)
    pdb_dir = ROOT / "cache/pdb_files"
    del Z0
    print("Building structural adjacency...")
    _, struct_adj = build_neighbor_graphs_residue_parallel(
        uids, res_lengths, ref_seqs, pdb_dir,
        n_jobs=-1, contact_cutoff=8.0, seq_gap_min=12)
    A_struct, deg_struct = adj_list_to_sparse(struct_adj, n_res_total)
    perm_indices = build_protein_permutations(res_lengths, 5)
    print(f"  adjacency: {A_struct.nnz:,} edges")

    rows = []
    for label, esm_l, rita_l in MATCHED_PAIRS:
        print(f"\n--- depth {label}%: ESM-2 L{esm_l} vs RITA L{rita_l} ---")

        # Compute global scale s per model from non-zero activations
        Z_esm = np.load(ROOT / f"outputs_layerwise/esm2/layer_{esm_l}/Z.npy")
        Z_rita = np.load(ROOT / f"outputs_layerwise/rita/layer_{rita_l}/Z.npy")

        nz_esm  = Z_esm[Z_esm > 0]
        nz_rita = Z_rita[Z_rita > 0]
        # Use median of non-zero activations as the typical magnitude.
        # Per-feature would be more rigorous but the prompt asks for one s
        # per model layer.
        s_esm  = float(np.median(nz_esm))  if len(nz_esm) else 1.0
        s_rita = float(np.median(nz_rita)) if len(nz_rita) else 1.0
        print(f"  typical magnitude s:  ESM={s_esm:.4f}  RITA={s_rita:.4f}")

        # Quantile-based (paper default) — read from existing csv
        ss_esm = pd.read_csv(ROOT / f"outputs_layerwise/esm2/layer_{esm_l}/struct_seq_metrics.csv")
        ss_rita = pd.read_csv(ROOT / f"outputs_layerwise/rita/layer_{rita_l}/struct_seq_metrics.csv")
        d_q = cohens_d(ss_esm.struct_delta.values, ss_rita.struct_delta.values)

        # Compute at fixed thresholds
        thresholds = {
            "0.5s": 0.5,
            "1.0s": 1.0,
            "2.0s": 2.0,
        }
        d_at_thresh = {}
        for label_t, mult in thresholds.items():
            t0 = time.time()
            sd_esm  = locality_at_threshold(Z_esm,  A_struct, deg_struct,
                                            perm_indices, mult * s_esm)
            sd_rita = locality_at_threshold(Z_rita, A_struct, deg_struct,
                                            perm_indices, mult * s_rita)
            d = cohens_d(sd_esm, sd_rita)
            d_at_thresh[label_t] = d
            print(f"  {label_t:>4}: d = {d:+.4f}  ({time.time()-t0:.0f}s)")

        rows.append(dict(
            rel_depth=f"{label}%", esm2_layer=esm_l, rita_layer=rita_l,
            d_quantile90=d_q,
            d_fixed_05s=d_at_thresh["0.5s"],
            d_fixed_1s=d_at_thresh["1.0s"],
            d_fixed_2s=d_at_thresh["2.0s"],
            s_esm=s_esm, s_rita=s_rita,
        ))

        # Free memory
        del Z_esm, Z_rita

    df = pd.DataFrame(rows)
    df["max_dev_from_quantile_pct"] = 100 * df[["d_fixed_05s","d_fixed_1s","d_fixed_2s"]] \
        .sub(df["d_quantile90"], axis=0).abs().max(axis=1) / df["d_quantile90"].abs().clip(lower=0.01)
    df.to_csv(OUT / "threshold_sensitivity.csv", index=False)

    print("\n" + "=" * 72)
    print(df.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    print("\n" + "=" * 72)
    median_dev = df["max_dev_from_quantile_pct"].median()
    if median_dev <= 20:
        verdict = f"H1 IS robust to active-residue definition (median max-dev = {median_dev:.1f}%, < 20%)"
    else:
        verdict = f"H1 IS SENSITIVE to active-residue definition (median max-dev = {median_dev:.1f}%, > 20%)"
    print(f"VERDICT: {verdict}")


if __name__ == "__main__":
    main()
