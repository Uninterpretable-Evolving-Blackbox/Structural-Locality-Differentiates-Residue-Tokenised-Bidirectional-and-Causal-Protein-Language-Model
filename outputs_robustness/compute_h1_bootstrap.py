#!/usr/bin/env python3
"""
compute_h1_bootstrap.py — Protein-level cluster bootstrap for the cross-model
H1 d (ESM-2 vs RITA on per-feature struct_delta).

Replaces the residue-level Mann–Whitney p<10⁻⁶ with a B=1000 bootstrap CI
that respects protein-level non-independence: residues from the same protein
are clustered (resampled together), so within-protein correlations don't
inflate effective sample size.

ASSUMPTIONS / CAVEATS — DOCUMENT IN PAPER
-----------------------------------------
1. Per-feature global SD σ_j and the 90th-percentile active-residue threshold
   are computed on the FULL data once and HELD FIXED across all 1000
   bootstrap resamples. Recomputing them per-resample would be ~3 orders of
   magnitude more expensive and is not standard practice. The reported CI
   is conditional on these nuisance parameters.

2. The mean-of-smoothed-activations "global mean" used in the locality
   numerator IS recomputed per-resample (it is a weighted aggregate over
   per-protein contributions which we precompute exactly).

3. Shuffle baseline averages over the same 5 within-protein permutations
   used by the main pipeline (seed=42, identical perms).

4. THIS BOOTSTRAP DOES NOT FIX FEATURE-LEVEL NON-INDEPENDENCE. SAE features
   share decoder weights and are correlated; the cross-model d treats the
   ~10k feature L_j values as independent samples, which they are not.
   The reported CI is conditional on the feature set and would widen
   under a paired (protein, feature) bootstrap.

5. The min-active-residues-per-protein-per-feature filter zeroes out
   (feature, protein) pairs with < 5 active residues — features that
   barely fire on a given protein contribute no signal and otherwise add
   noise. Documented in the prompt.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

# Repo root = parent of outputs_robustness/
ROOT = Path(__file__).resolve().parent.parent

import sys
sys.path.insert(0, str(ROOT))
from cpu_stage import (
    load_layer, load_ref_seqs,
    build_neighbor_graphs_residue_parallel,
    adj_list_to_sparse, build_protein_permutations,
)

OUT = ROOT / "outputs_robustness"
OUT.mkdir(parents=True, exist_ok=True)

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

N_BOOT = 1000
N_SHUF = 5
TOPK_FRAC = 0.10
MIN_ACTIVE = 5     # set to 0 via --no-min-active to disable


def cohens_d(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    pooled = np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2.0 + 1e-12)
    return float((a.mean() - b.mean()) / pooled)


def per_protein_contributions(layer_dir, A_struct, deg_struct, perm_indices,
                              protein_offsets, n_res_total):
    """For one (model, layer): per-(protein, feature) sums needed for bootstrap.

    Returns dict of arrays (n_proteins, n_feat):
      obs_active_sum, obs_active_count, obs_global_sum,
      shuf_active_sum, shuf_active_count, shuf_global_sum (averaged over shuffles),
      sigma_j (n_feat,), n_residues (n_proteins,)
    """
    Z, _uids, _lens = load_layer(layer_dir)
    Z = np.asarray(Z, dtype=np.float32)
    n_res, n_feat = Z.shape
    assert n_res == n_res_total

    # Held-fixed global stats
    sigma_j = Z.std(axis=0, ddof=0).astype(np.float32) + 1e-6
    thresh_j = np.percentile(Z, 100.0 * (1.0 - TOPK_FRAC), axis=0).astype(np.float32)

    # Build smoothed = (A·Z)/deg once
    nbr_sum = (A_struct @ Z).astype(np.float32)
    has_nb = deg_struct > 0
    smoothed = np.zeros_like(Z)
    smoothed[has_nb] = nbr_sum[has_nb] / deg_struct[has_nb, None]

    active = Z > thresh_j[None, :]    # (n_res, n_feat) bool
    active_f32 = active.astype(np.float32)

    n_proteins = len(protein_offsets) - 1

    # Pre-allocate per-protein contribution buffers
    obs_active_sum = np.zeros((n_proteins, n_feat), dtype=np.float32)
    obs_active_cnt = np.zeros((n_proteins, n_feat), dtype=np.float32)
    obs_global_sum = np.zeros((n_proteins, n_feat), dtype=np.float32)

    for p in range(n_proteins):
        s, e = protein_offsets[p], protein_offsets[p + 1]
        obs_active_sum[p] = (smoothed[s:e] * active_f32[s:e]).sum(axis=0)
        obs_active_cnt[p] = active_f32[s:e].sum(axis=0)
        obs_global_sum[p] = smoothed[s:e].sum(axis=0)

    # Free smoothed before shuffles, since we'll recompute per-perm
    del smoothed, nbr_sum, active, active_f32

    shuf_active_sum = np.zeros((n_proteins, n_feat), dtype=np.float32)
    shuf_active_cnt = np.zeros((n_proteins, n_feat), dtype=np.float32)
    shuf_global_sum = np.zeros((n_proteins, n_feat), dtype=np.float32)

    for perm in perm_indices:
        Zp = Z[perm]
        nbrp = (A_struct @ Zp).astype(np.float32)
        smp = np.zeros_like(Zp)
        smp[has_nb] = nbrp[has_nb] / deg_struct[has_nb, None]
        # Per-column distribution unchanged by within-protein perm → same threshold
        actp = (Zp > thresh_j[None, :]).astype(np.float32)
        for p in range(n_proteins):
            s, e = protein_offsets[p], protein_offsets[p + 1]
            shuf_active_sum[p] += (smp[s:e] * actp[s:e]).sum(axis=0)
            shuf_active_cnt[p] += actp[s:e].sum(axis=0)
            shuf_global_sum[p] += smp[s:e].sum(axis=0)
        del Zp, nbrp, smp, actp

    shuf_active_sum /= len(perm_indices)
    shuf_active_cnt /= len(perm_indices)
    shuf_global_sum /= len(perm_indices)

    n_residues = np.diff(protein_offsets).astype(np.float32)

    return dict(
        sigma_j=sigma_j, n_residues=n_residues,
        obs_active_sum=obs_active_sum, obs_active_cnt=obs_active_cnt,
        obs_global_sum=obs_global_sum,
        shuf_active_sum=shuf_active_sum, shuf_active_cnt=shuf_active_cnt,
        shuf_global_sum=shuf_global_sum,
    )


def apply_active_filter(c, min_active=MIN_ACTIVE):
    """Zero-out (protein, feature) pairs with too few active residues."""
    mask = c["obs_active_cnt"] < min_active
    c["obs_active_sum"][mask] = 0.0
    c["obs_active_cnt"][mask] = 0.0
    # Don't filter shuffles — they participate in the baseline subtraction


def struct_delta_under_weights(c, w):
    """Compute per-feature struct_delta_j given protein weights w (n_proteins,)."""
    w = w.astype(np.float64)
    boot_obs_act_sum = w @ c["obs_active_sum"]
    boot_obs_act_cnt = w @ c["obs_active_cnt"]
    boot_obs_glb_sum = w @ c["obs_global_sum"]
    boot_n_res       = w @ c["n_residues"]

    boot_shf_act_sum = w @ c["shuf_active_sum"]
    boot_shf_act_cnt = w @ c["shuf_active_cnt"]
    boot_shf_glb_sum = w @ c["shuf_global_sum"]

    # Avoid division by zero (features with no active residues anywhere)
    obs_active_mean = np.where(boot_obs_act_cnt > 0,
                               boot_obs_act_sum / np.maximum(boot_obs_act_cnt, 1), 0)
    obs_global_mean = boot_obs_glb_sum / max(boot_n_res, 1)
    shf_active_mean = np.where(boot_shf_act_cnt > 0,
                               boot_shf_act_sum / np.maximum(boot_shf_act_cnt, 1), 0)
    shf_global_mean = boot_shf_glb_sum / max(boot_n_res, 1)

    obs_d = (obs_active_mean - obs_global_mean) / c["sigma_j"]
    shf_d = (shf_active_mean - shf_global_mean) / c["sigma_j"]
    return (obs_d - shf_d).astype(np.float32)


def make_boot_weights(n_proteins, indices, n_boot, rng):
    """For each of n_boot iters, sample n_indices (=len(indices)) proteins
    with replacement *from the indices subset*; build (B, n_proteins) weight
    matrix where weights are 0 for non-indices."""
    out = np.zeros((n_boot, n_proteins), dtype=np.int32)
    n = len(indices)
    for b in range(n_boot):
        sample = rng.choice(indices, size=n, replace=True)
        u, c = np.unique(sample, return_counts=True)
        out[b, u] = c
    return out


def run_bootstrap(c_esm, c_rita, weights):
    """Vectorised bootstrap loop: for each of B weights, compute cross-model d."""
    n_boot = weights.shape[0]
    d_boot = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        sd_e = struct_delta_under_weights(c_esm,  weights[b])
        sd_r = struct_delta_under_weights(c_rita, weights[b])
        d_boot[b] = cohens_d(sd_e, sd_r)
    return d_boot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--depths", type=str, default="all",
                    help="comma-separated subset of depth labels, or 'all'")
    ap.add_argument("--min-active", type=int, default=MIN_ACTIVE,
                    help="drop (protein, feature) pairs with fewer than this many active residues")
    args = ap.parse_args()

    print("=" * 72)
    print("  H1 protein-level cluster bootstrap (Task 1)")
    print(f"  N_BOOT={args.n_boot}  N_SHUF={N_SHUF}  TOPK_FRAC={TOPK_FRAC}")
    print("=" * 72)

    # ---- Load shared structure ----
    layer0_dir = ROOT / "outputs_layerwise/esm2/layer_0"
    print("\nLoading shared protein metadata + structural adjacency...")
    Z0, uids, lengths = load_layer(layer0_dir)
    res_lengths = lengths.astype(np.int32)
    n_proteins = len(res_lengths)
    n_res_total = int(res_lengths.sum())
    protein_offsets = np.concatenate([[0], np.cumsum(res_lengths.astype(np.int64))])
    del Z0  # don't need actual activations from this layer

    val_uids = json.loads((layer0_dir / "META.json").read_text())["val_uids"]
    val_uid_set = set(val_uids)
    val_indices = np.array([i for i, u in enumerate(uids) if u in val_uid_set])
    full_indices = np.arange(n_proteins)
    print(f"  n_proteins={n_proteins}, val_proteins={len(val_indices)}, "
          f"n_residues={n_res_total}")

    ref_seqs = load_ref_seqs(layer0_dir)
    pdb_dir = ROOT / "cache/pdb_files"
    t0 = time.time()
    _, struct_adj = build_neighbor_graphs_residue_parallel(
        uids, res_lengths, ref_seqs, pdb_dir,
        n_jobs=-1, contact_cutoff=8.0, seq_gap_min=12)
    A_struct, deg_struct = adj_list_to_sparse(struct_adj, n_res_total)
    perm_indices = build_protein_permutations(res_lengths, N_SHUF)
    print(f"  adjacency: {A_struct.nnz:,} edges  ({time.time()-t0:.0f}s)")

    # ---- Boot weights (shared across layers) ----
    rng = np.random.default_rng(42)
    boot_w_full = make_boot_weights(n_proteins, full_indices, args.n_boot, rng)
    boot_w_val  = make_boot_weights(n_proteins, val_indices,  args.n_boot, rng)
    print(f"  bootstrap weight matrices built: full={boot_w_full.shape}, val={boot_w_val.shape}")

    # ---- Process each depth pair ----
    if args.depths == "all":
        pairs = MATCHED_PAIRS
    else:
        wanted = set(args.depths.split(","))
        pairs = [p for p in MATCHED_PAIRS if p[0] in wanted]
    print(f"\n  Will process {len(pairs)} depth pairs: {[p[0]+'%' for p in pairs]}")

    rows_full, rows_val = [], []
    sd_per_depth_full = {}  # depth -> (B,) bootstrap d trace, full
    sd_per_depth_val  = {}  # ditto, val

    for label, esm_l, rita_l in pairs:
        print(f"\n--- depth {label}%: ESM-2 L{esm_l} vs RITA L{rita_l} ---")
        t0 = time.time()
        c_esm = per_protein_contributions(
            ROOT / f"outputs_layerwise/esm2/layer_{esm_l}",
            A_struct, deg_struct, perm_indices,
            protein_offsets, n_res_total)
        print(f"  ESM-2 contributions: {time.time()-t0:.0f}s")
        t1 = time.time()
        c_rita = per_protein_contributions(
            ROOT / f"outputs_layerwise/rita/layer_{rita_l}",
            A_struct, deg_struct, perm_indices,
            protein_offsets, n_res_total)
        print(f"  RITA  contributions: {time.time()-t1:.0f}s")

        if args.min_active > 0:
            apply_active_filter(c_esm,  min_active=args.min_active)
            apply_active_filter(c_rita, min_active=args.min_active)

        # Point estimates (full data)
        w_full = np.ones(n_proteins, dtype=np.int32)
        sd_e_full  = struct_delta_under_weights(c_esm,  w_full)
        sd_r_full  = struct_delta_under_weights(c_rita, w_full)
        d_full_pt  = cohens_d(sd_e_full, sd_r_full)

        w_val_pt = np.zeros(n_proteins, dtype=np.int32)
        w_val_pt[val_indices] = 1
        sd_e_val  = struct_delta_under_weights(c_esm,  w_val_pt)
        sd_r_val  = struct_delta_under_weights(c_rita, w_val_pt)
        d_val_pt  = cohens_d(sd_e_val, sd_r_val)

        # Bootstrap loop (full)
        t2 = time.time()
        d_boot_full = run_bootstrap(c_esm, c_rita, boot_w_full)
        d_boot_val  = run_bootstrap(c_esm, c_rita, boot_w_val)
        print(f"  bootstrap {args.n_boot}x2: {time.time()-t2:.0f}s")

        ci_full = np.percentile(d_boot_full, [2.5, 97.5])
        ci_val  = np.percentile(d_boot_val,  [2.5, 97.5])
        rows_full.append(dict(
            rel_depth=f"{label}%", layer_pair=f"L{esm_l}/L{rita_l}",
            d_point=d_full_pt, ci_low=ci_full[0], ci_high=ci_full[1],
            frac_pos=float((d_boot_full > 0).mean()),
            n_proteins_used=n_proteins,
            d_boot_mean=float(d_boot_full.mean()),
            d_boot_sd=float(d_boot_full.std(ddof=1)),
        ))
        rows_val.append(dict(
            rel_depth=f"{label}%", layer_pair=f"L{esm_l}/L{rita_l}",
            d_point=d_val_pt, ci_low=ci_val[0], ci_high=ci_val[1],
            frac_pos=float((d_boot_val > 0).mean()),
            n_proteins_used=int(len(val_indices)),
            d_boot_mean=float(d_boot_val.mean()),
            d_boot_sd=float(d_boot_val.std(ddof=1)),
        ))
        sd_per_depth_full[label] = d_boot_full
        sd_per_depth_val[label]  = d_boot_val

        print(f"  full: d={d_full_pt:+.4f} CI=[{ci_full[0]:+.4f}, {ci_full[1]:+.4f}] "
              f"frac_pos={(d_boot_full>0).mean():.3f}")
        print(f"   val: d={d_val_pt:+.4f} CI=[{ci_val[0]:+.4f}, {ci_val[1]:+.4f}] "
              f"frac_pos={(d_boot_val>0).mean():.3f}")

        del c_esm, c_rita

    # ---- Save & report ----
    df_full = pd.DataFrame(rows_full)
    df_full.to_csv(OUT / "bootstrap_h1_full.csv", index=False)
    df_val = pd.DataFrame(rows_val)
    df_val.to_csv(OUT / "bootstrap_h1_val.csv", index=False)
    np.savez_compressed(OUT / "bootstrap_h1_traces.npz",
                        **{f"full_{k}": v for k, v in sd_per_depth_full.items()},
                        **{f"val_{k}": v for k, v in sd_per_depth_val.items()})

    print("\n" + "=" * 72)
    print("  FULL-set bootstrap (1500 proteins resampled with replacement)")
    print("=" * 72)
    print(df_full.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

    print("\n" + "=" * 72)
    print(f"  VAL-set bootstrap ({len(val_indices)} proteins resampled with replacement)")
    print("=" * 72)
    print(df_val.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

    # Cross-bootstrap correlation between depths (sanity check)
    if len(sd_per_depth_full) >= 2:
        labels = [p[0] for p in pairs]
        mat = np.stack([sd_per_depth_full[l] for l in labels])  # (n_depths, n_boot)
        corr = np.corrcoef(mat)
        print("\nCross-depth bootstrap correlation matrix (full set):")
        print("  rows/cols:", labels)
        with np.printoptions(precision=3, suppress=True, linewidth=120):
            print(corr)

    print(f"\nWritten: {OUT/'bootstrap_h1_full.csv'}")
    print(f"         {OUT/'bootstrap_h1_val.csv'}")
    print(f"         {OUT/'bootstrap_h1_traces.npz'}")


if __name__ == "__main__":
    main()
