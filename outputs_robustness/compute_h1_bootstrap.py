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
from cluster_bootstrap import (
    clusters_for_uids, make_cluster_weights, design_effect, cluster_stats,
)

OUT = ROOT / "outputs_robustness"
OUT.mkdir(parents=True, exist_ok=True)

MATCHED_PAIRS_ESM_RITA = [
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

# Pre-registered 9-depth grids for the additional bidirectional/causal pair
# (ProtBert-BFD 30 blocks, ProGen2-base 27 blocks; same relative-depth matching).
MATCHED_PAIRS_PROTbert_PROGEN2 = [
    ("0",     0,   0),
    ("13",    4,   3),
    ("25",    7,   6),
    ("38",    11,  10),
    ("50",    14,  13),
    ("63",    18,  16),
    ("75",    22,  20),
    ("88",    25,  23),
    ("100",   29,  26),
]

MATCHED_PAIRS_CTRL = [
    ("0", 0, 0), ("25", 3, 3), ("50", 6, 6), ("75", 9, 9), ("100", 11, 11),
]

PAIR_PRESETS = {
    "ctrl": dict(
        model_a="ctrl_mlm_A", model_b="ctrl_clm_A",
        output_root="outputs_ctrl",
        matched_pairs=MATCHED_PAIRS_CTRL,
        out_stem="bootstrap_h1_ctrl",
    ),
    # k_sparse=96 rerun. The original ctrl SAEs used k=256, which was ablated on
    # ESM-2 L16 (1280-dim) and gives k/embed_dim = 53% on these 480-dim models
    # (vs 20% for ESM-2, 17% for RITA) -> near-invertible basis, val_EV up to 0.997,
    # i.e. above the 0.99 at which ProGen2 was dropped as degenerate. k=96 restores
    # ESM-2 L16's regime exactly on both axes (k/embed 20.0%, k/hidden 2.50%).
    "ctrl_k96": dict(
        model_a="ctrl_mlm_A", model_b="ctrl_clm_A",
        output_root="outputs_ctrl_k96",
        matched_pairs=MATCHED_PAIRS_CTRL,
        out_stem="bootstrap_h1_ctrl_k96",
    ),
    "esm_rita": dict(
        model_a="esm2", model_b="rita",
        output_root="outputs_layerwise",
        matched_pairs=MATCHED_PAIRS_ESM_RITA,
        out_stem="bootstrap_h1",
    ),
    "protbert_progen2": dict(
        model_a="protbert_bfd", model_b="progen2",
        output_root="outputs_layerwise_newpair",
        matched_pairs=MATCHED_PAIRS_PROTbert_PROGEN2,
        out_stem="bootstrap_h1_newpair",
    ),
}

# Back-compat alias
MATCHED_PAIRS = MATCHED_PAIRS_ESM_RITA

N_BOOT = 1000
N_SHUF = 5
TOPK_FRAC = 0.10
MIN_ACTIVE = 0     # 0 = no filter; reproduces the paper's committed bootstrap CIs.
                   # >0 (via --min-active N) drops (protein,feature) pairs with fewer
                   # than N active residues. NOTE: the committed paper CSVs were made
                   # with 0; the previous default of 5 did NOT reproduce them.


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
    ap.add_argument("--preset", choices=sorted(PAIR_PRESETS), default="esm_rita",
                    help="model pair preset: esm_rita (paper H1) or protbert_progen2")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--depths", type=str, default="all",
                    help="comma-separated subset of depth labels, or 'all'")
    ap.add_argument("--min-active", type=int, default=MIN_ACTIVE,
                    help="drop (protein, feature) pairs with fewer than this many active residues")
    ap.add_argument("--cluster-levels", default="fold,protein",
                    help="comma-separated resampling units from {protein,fold,superfamily,family}; "
                         "all are computed in ONE run sharing the (expensive) adjacency build. "
                         "'protein' reproduces the original paper bootstrap for direct comparison.")
    ap.add_argument("--two-stage", dest="two_stage", action="store_true", default=True,
                    help="hierarchical resample: clusters, then proteins within each cluster")
    ap.add_argument("--one-stage", dest="two_stage", action="store_false",
                    help="resample whole clusters only (between-cluster variance)")
    ap.add_argument("--fasta", default=str(ROOT / "cache/scope_40.fa"),
                    help="SCOPe FASTA providing the fold/superfamily clustering key")
    args = ap.parse_args()

    preset = PAIR_PRESETS[args.preset]
    model_a = preset["model_a"]
    model_b = preset["model_b"]
    output_root = ROOT / preset["output_root"]
    matched_pairs = preset["matched_pairs"]
    out_stem = preset["out_stem"]

    print("=" * 72)
    print(f"  H1 cluster bootstrap — preset={args.preset}")
    print(f"  {model_a} (bidirectional) vs {model_b} (causal)")
    print(f"  N_BOOT={args.n_boot}  N_SHUF={N_SHUF}  TOPK_FRAC={TOPK_FRAC}  "
          f"min_active={args.min_active}")
    print("=" * 72)

    # ---- Load shared structure ----
    layer0_dir = output_root / f"{model_a}/layer_0"
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

    # ---- Cluster assignments + boot weights PER LEVEL (shared across layers) ----
    ALLOWED = {"protein", "fold", "superfamily", "family"}
    levels = [x.strip() for x in args.cluster_levels.split(",") if x.strip()]
    bad = [lv for lv in levels if lv not in ALLOWED]
    if bad:
        raise SystemExit(f"unknown cluster level(s) {bad}; allowed {sorted(ALLOWED)}")
    cluster_ids, boot_w_full, boot_w_val = {}, {}, {}
    for lvl in levels:
        cid = clusters_for_uids(uids, args.fasta, level=lvl)
        cst = cluster_stats(cid, full_indices)
        print(f"  cluster-level={lvl} ({'two-stage' if args.two_stage else 'one-stage'}): "
              f"{cst['n_clusters']} clusters / {cst['n_units']} proteins "
              f"(mean {cst['mean_cluster_size']:.2f}, max {cst['max_cluster_size']})")
        rng_l = np.random.default_rng(42)
        cluster_ids[lvl] = cid
        boot_w_full[lvl] = make_cluster_weights(n_proteins, cid, full_indices, args.n_boot, rng_l,
                                                two_stage=args.two_stage)
        boot_w_val[lvl]  = make_cluster_weights(n_proteins, cid, val_indices, args.n_boot, rng_l,
                                                two_stage=args.two_stage)

    # ---- Process each depth pair ----
    if args.depths == "all":
        pairs = matched_pairs
    else:
        wanted = set(args.depths.split(","))
        pairs = [p for p in matched_pairs if p[0] in wanted]
    print(f"\n  Will process {len(pairs)} depth pairs: {[p[0]+'%' for p in pairs]}")

    rows_full, rows_val = [], []
    sd_per_depth_full = {}  # depth -> (B,) bootstrap d trace, full
    sd_per_depth_val  = {}  # ditto, val

    for label, layer_a, layer_b in pairs:
        print(f"\n--- depth {label}%: {model_a} L{layer_a} vs {model_b} L{layer_b} ---")
        t0 = time.time()
        c_a = per_protein_contributions(
            output_root / f"{model_a}/layer_{layer_a}",
            A_struct, deg_struct, perm_indices,
            protein_offsets, n_res_total)
        print(f"  {model_a} contributions: {time.time()-t0:.0f}s")
        t1 = time.time()
        c_b = per_protein_contributions(
            output_root / f"{model_b}/layer_{layer_b}",
            A_struct, deg_struct, perm_indices,
            protein_offsets, n_res_total)
        print(f"  {model_b} contributions: {time.time()-t1:.0f}s")

        if args.min_active > 0:
            apply_active_filter(c_a, min_active=args.min_active)
            apply_active_filter(c_b, min_active=args.min_active)

        # Point estimates (clustering-independent; computed once per depth)
        w_full = np.ones(n_proteins, dtype=np.int32)
        d_full_pt = cohens_d(struct_delta_under_weights(c_a, w_full),
                             struct_delta_under_weights(c_b, w_full))
        w_val_pt = np.zeros(n_proteins, dtype=np.int32); w_val_pt[val_indices] = 1
        d_val_pt = cohens_d(struct_delta_under_weights(c_a, w_val_pt),
                            struct_delta_under_weights(c_b, w_val_pt))
        # Per-protein L_struct summary for the design effect (clustering-independent).
        _am = np.where(c_a["obs_active_cnt"] > 0,
                       c_a["obs_active_sum"] / np.maximum(c_a["obs_active_cnt"], 1), 0.0)
        _gm = c_a["obs_global_sum"] / np.maximum(c_a["n_residues"][:, None], 1.0)
        _per_prot = ((_am - _gm) / c_a["sigma_j"][None, :]).mean(axis=1)

        layer_pair = f"L{layer_a}/L{layer_b}"
        for lvl in levels:
            t2 = time.time()
            d_boot_full = run_bootstrap(c_a, c_b, boot_w_full[lvl])
            d_boot_val  = run_bootstrap(c_a, c_b, boot_w_val[lvl])
            ci_full = np.percentile(d_boot_full, [2.5, 97.5])
            ci_val  = np.percentile(d_boot_val,  [2.5, 97.5])
            de = design_effect(_per_prot, cluster_ids[lvl], full_indices)
            rows_full.append(dict(
                rel_depth=f"{label}%", layer_pair=layer_pair, cluster_level=lvl,
                model_a=model_a, model_b=model_b, preset=args.preset,
                d_point=d_full_pt, ci_low=ci_full[0], ci_high=ci_full[1],
                frac_pos=float((d_boot_full > 0).mean()), n_proteins_used=n_proteins,
                d_boot_mean=float(d_boot_full.mean()), d_boot_sd=float(d_boot_full.std(ddof=1)),
                icc=de["icc"], design_effect=de["design_effect"],
                n_eff=de["n_eff"], n_clusters=de["n_clusters"]))
            rows_val.append(dict(
                rel_depth=f"{label}%", layer_pair=layer_pair, cluster_level=lvl,
                model_a=model_a, model_b=model_b, preset=args.preset,
                d_point=d_val_pt, ci_low=ci_val[0], ci_high=ci_val[1],
                frac_pos=float((d_boot_val > 0).mean()), n_proteins_used=int(len(val_indices)),
                d_boot_mean=float(d_boot_val.mean()), d_boot_sd=float(d_boot_val.std(ddof=1))))
            sd_per_depth_full[(label, lvl)] = d_boot_full
            sd_per_depth_val[(label, lvl)] = d_boot_val
            print(f"  [{lvl:11s}] full d_pt={d_full_pt:+.4f} dbar={d_boot_full.mean():+.4f} "
                  f"CI=[{ci_full[0]:+.4f},{ci_full[1]:+.4f}] fracpos={(d_boot_full>0).mean():.3f} "
                  f"| Deff={de['design_effect']:.2f} n_eff={de['n_eff']:.0f} ({time.time()-t2:.0f}s)")

        del c_a, c_b

    # ---- Save & report ----
    # Combined by-level CSVs (cluster_level column distinguishes rows). min_active
    # is in the filename so the original paper CSVs are never clobbered and the
    # setting is self-documenting.
    tag = f"minact{args.min_active}"
    df_full = pd.DataFrame(rows_full)
    df_full.to_csv(OUT / f"{out_stem}_full_bylevel_{tag}.csv", index=False)
    df_val = pd.DataFrame(rows_val)
    df_val.to_csv(OUT / f"{out_stem}_val_bylevel_{tag}.csv", index=False)
    np.savez_compressed(OUT / f"{out_stem}_traces_bylevel_{tag}.npz",
                        preset=args.preset,
                        **{f"full_{lvl}_{lab}": v for (lab, lvl), v in sd_per_depth_full.items()},
                        **{f"val_{lvl}_{lab}": v for (lab, lvl), v in sd_per_depth_val.items()})

    print("\n" + "=" * 72)
    print("  FULL-set bootstrap (1500 proteins resampled with replacement)")
    print("=" * 72)
    print(df_full.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

    print("\n" + "=" * 72)
    print(f"  VAL-set bootstrap ({len(val_indices)} proteins resampled with replacement)")
    print("=" * 72)
    print(df_val.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

    print(f"\nWritten: {OUT / f'{out_stem}_full_bylevel_{tag}.csv'}")
    print(f"         {OUT / f'{out_stem}_val_bylevel_{tag}.csv'}")
    print(f"         {OUT / f'{out_stem}_traces_bylevel_{tag}.npz'}")


if __name__ == "__main__":
    main()
