#!/usr/bin/env python3
"""
v2: optimized. Per (model, layer): load Z once, compute ALL needed contribution
variants (struct top-decile, seq top-decile, struct@0.5s, struct@1s, struct@2s) in
one pass. Vectorized per-protein sums via np.add.reduceat. flush=True everywhere.

Processes:
  Phase A: ESM-2 vs RITA — L_struct (already done), L_seq, threshold variants,
           plus per-model L_struct trajectory means.
  Phase B: ProtT5 enc vs dec — L_struct, L_seq, plus per-model trajectory means.
"""

import os
# Use all cores for numpy / BLAS BEFORE imports
os.environ.setdefault("OMP_NUM_THREADS", "18")
os.environ.setdefault("MKL_NUM_THREADS", "18")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "18")

import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs_robustness"
PROG = OUT / "v2_progress.txt"

from cpu_stage import (
    load_layer, load_ref_seqs,
    build_neighbor_graphs_residue_parallel,
    adj_list_to_sparse, build_protein_permutations,
)

N_BOOT = 1000
N_SHUF = 5
TOPK_FRAC = 0.10
SEED = 42

def log(msg):
    s = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(s, flush=True)
    with open(PROG, "a") as f:
        f.write(s + "\n")

DEPTHS_ER = [
    ("0",   0,  0), ("13",  4,  3), ("25",  8,  6),
    ("38", 12,  9), ("50", 16, 12), ("63", 20, 15),
    ("75", 24, 18), ("88", 28, 21), ("100",32, 23),
]
DEPTHS_PT5 = [
    ("0",  0), ("13", 3), ("25", 6), ("38", 9), ("50",12),
    ("63",15), ("75",18), ("88",21), ("100",23),
]


def cohens_d(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    pooled = np.sqrt((a.std(ddof=1)**2 + b.std(ddof=1)**2)/2 + 1e-12)
    return float((a.mean() - b.mean()) / pooled)


def per_protein_contribs_multi(layer_dir, A_struct, deg_struct, A_seq, deg_seq,
                                perm_indices, protein_offsets):
    """One Z load → contributions for 5 variants:
       struct_topdec, seq_topdec, struct_0.5s, struct_1.0s, struct_2.0s.
    Uses np.add.reduceat for fast per-protein sums.
    """
    t0 = time.time()
    Z, _u, _l = load_layer(layer_dir)
    Z = np.asarray(Z, dtype=np.float32)
    n_res, n_feat = Z.shape
    log(f"    Z loaded shape={Z.shape} ({time.time()-t0:.1f}s)")

    t0 = time.time()
    sigma_j = Z.std(axis=0, ddof=0).astype(np.float32) + 1e-6
    log(f"    sigma_j ({time.time()-t0:.1f}s)")

    t0 = time.time()
    thresh_top = np.percentile(Z, 100*(1-TOPK_FRAC), axis=0).astype(np.float32)
    log(f"    top-decile thresh ({time.time()-t0:.1f}s)")

    t0 = time.time()
    nz = Z[Z > 0]
    s_med = float(np.median(nz)) if len(nz) else 1.0
    log(f"    s_median = {s_med:.4f}  ({time.time()-t0:.1f}s)")

    # Active masks: 5 variants
    masks = {
        "struct_topdec": Z > thresh_top[None, :],
        "seq_topdec":    Z > thresh_top[None, :],   # same mask, different smoothing
        "struct_0.5s":   Z > 0.5 * s_med,
        "struct_1.0s":   Z > 1.0 * s_med,
        "struct_2.0s":   Z > 2.0 * s_med,
    }

    # Smoothed activations for each adjacency
    t0 = time.time()
    nbr_struct = (A_struct @ Z).astype(np.float32)
    has_s = deg_struct > 0
    sm_struct = np.zeros_like(Z)
    sm_struct[has_s] = nbr_struct[has_s] / deg_struct[has_s, None]
    del nbr_struct
    log(f"    A_struct @ Z + smooth ({time.time()-t0:.1f}s)")

    t0 = time.time()
    nbr_seq = (A_seq @ Z).astype(np.float32)
    has_q = deg_seq > 0
    sm_seq = np.zeros_like(Z)
    sm_seq[has_q] = nbr_seq[has_q] / deg_seq[has_q, None]
    del nbr_seq
    log(f"    A_seq @ Z + smooth ({time.time()-t0:.1f}s)")

    n_proteins = len(protein_offsets) - 1
    starts = protein_offsets[:-1]  # for reduceat

    def per_protein_sums(active_mask, smoothed):
        """Returns (obs_active_sum, obs_active_cnt, obs_global_sum) of shape (n_proteins, n_feat)."""
        actf = active_mask.astype(np.float32)
        prod = smoothed * actf
        active_sum = np.add.reduceat(prod, starts, axis=0)
        active_cnt = np.add.reduceat(actf, starts, axis=0)
        global_sum = np.add.reduceat(smoothed, starts, axis=0)
        return active_sum, active_cnt, global_sum

    # Build observed contributions for each variant
    contribs = {}
    for name in masks:
        smoothed = sm_struct if name.startswith("struct") else sm_seq
        t0 = time.time()
        a_s, a_c, g_s = per_protein_sums(masks[name], smoothed)
        contribs[name] = dict(obs_as=a_s, obs_ac=a_c, obs_gs=g_s,
                              shf_as=np.zeros_like(a_s),
                              shf_ac=np.zeros_like(a_s),
                              shf_gs=np.zeros_like(a_s))
        log(f"    obs_contribs[{name}] ({time.time()-t0:.1f}s)")

    # Permuted contributions: for each perm, recompute smoothed and sums.
    # NOTE: for top-decile mask, perm is applied to Z but thresh_j is column-wise so
    # active mask under permutation has identical column distributions → reuse mask
    # of permuted Z. For fixed threshold variants (0.5/1/2 s), Z values are permuted
    # so absolute threshold gives different mask.
    for pi, perm in enumerate(perm_indices):
        t0 = time.time()
        Zp = Z[perm]
        # Smoothed under perm
        nbr_s = (A_struct @ Zp).astype(np.float32)
        smp_s = np.zeros_like(Zp); smp_s[has_s] = nbr_s[has_s] / deg_struct[has_s, None]
        del nbr_s
        nbr_q = (A_seq @ Zp).astype(np.float32)
        smp_q = np.zeros_like(Zp); smp_q[has_q] = nbr_q[has_q] / deg_seq[has_q, None]
        del nbr_q

        masks_p = {
            "struct_topdec": Zp > thresh_top[None, :],
            "seq_topdec":    Zp > thresh_top[None, :],
            "struct_0.5s":   Zp > 0.5 * s_med,
            "struct_1.0s":   Zp > 1.0 * s_med,
            "struct_2.0s":   Zp > 2.0 * s_med,
        }
        for name in masks_p:
            smoothed = smp_s if name.startswith("struct") else smp_q
            a_s, a_c, g_s = per_protein_sums(masks_p[name], smoothed)
            contribs[name]["shf_as"] += a_s
            contribs[name]["shf_ac"] += a_c
            contribs[name]["shf_gs"] += g_s
        del Zp, smp_s, smp_q, masks_p
        log(f"    perm {pi+1}/{len(perm_indices)} done ({time.time()-t0:.1f}s)")

    for name in contribs:
        contribs[name]["shf_as"] /= len(perm_indices)
        contribs[name]["shf_ac"] /= len(perm_indices)
        contribs[name]["shf_gs"] /= len(perm_indices)

    n_residues = np.diff(protein_offsets).astype(np.float32)
    return contribs, sigma_j, n_residues, s_med


def delta_under_w(c, sigma_j, n_residues, w):
    w = w.astype(np.float64)
    a_s, a_c, g_s = w@c["obs_as"], w@c["obs_ac"], w@c["obs_gs"]
    s_s, s_c, s_g = w@c["shf_as"], w@c["shf_ac"], w@c["shf_gs"]
    n = w@n_residues
    am = np.where(a_c > 0, a_s / np.maximum(a_c, 1), 0)
    gm = g_s / max(n, 1)
    sm = np.where(s_c > 0, s_s / np.maximum(s_c, 1), 0)
    sg = s_g / max(n, 1)
    return ((am - gm) - (sm - sg)) / sigma_j


def make_weights(n_proteins, indices, n_boot, rng):
    out = np.zeros((n_boot, n_proteins), dtype=np.int32)
    for b in range(n_boot):
        sample = rng.choice(indices, size=len(indices), replace=True)
        u, cnt = np.unique(sample, return_counts=True)
        out[b, u] = cnt
    return out


def boot_pair(c_a, sa, na, c_b, sb, nb, weights):
    B = weights.shape[0]
    out = np.empty(B)
    for i in range(B):
        out[i] = cohens_d(delta_under_w(c_a, sa, na, weights[i]),
                          delta_under_w(c_b, sb, nb, weights[i]))
    return out


def boot_within_mean(c, sigma_j, n_res, weights):
    B = weights.shape[0]
    out = np.empty(B)
    for i in range(B):
        out[i] = float(delta_under_w(c, sigma_j, n_res, weights[i]).mean())
    return out


def cell_summary(d_point, trace):
    sd = trace.std(ddof=1)
    return dict(d_point=d_point, d_boot_mean=float(trace.mean()),
                boot_sd=float(sd),
                ci_normal_lo=d_point - 1.96*sd, ci_normal_hi=d_point + 1.96*sd,
                ci_pct_lo=float(np.percentile(trace, 2.5)),
                ci_pct_hi=float(np.percentile(trace, 97.5)))


def main():
    PROG.unlink(missing_ok=True)
    log("=" * 76)
    log("  v2 bootstrap CIs — paired ESM/RITA + ProtT5 enc/dec, B=1000")
    log("=" * 76)

    layer0 = ROOT / "outputs_layerwise/esm2/layer_0"
    Z0, uids, lengths = load_layer(layer0)
    res_lengths = lengths.astype(np.int32)
    n_proteins = len(res_lengths)
    n_res_total = int(res_lengths.sum())
    protein_offsets = np.concatenate([[0], np.cumsum(res_lengths.astype(np.int64))])
    val_uids = json.loads((layer0/"META.json").read_text())["val_uids"]
    val_indices = np.array([i for i, u in enumerate(uids) if u in set(val_uids)])
    full_indices = np.arange(n_proteins)
    log(f"  n_proteins={n_proteins}, val={len(val_indices)}, n_res={n_res_total}")
    del Z0

    ref_seqs = load_ref_seqs(layer0)
    pdb_dir = ROOT / "cache/pdb_files"

    rng = np.random.default_rng(SEED)
    perm_indices = build_protein_permutations(res_lengths, N_SHUF)
    W_full = make_weights(n_proteins, full_indices, N_BOOT, rng)
    W_val  = make_weights(n_proteins, val_indices,  N_BOOT, np.random.default_rng(SEED+1))
    log(f"  weights: full {W_full.shape}, val {W_val.shape}")

    log("Building 8 A struct + seq adjacency...")
    t0 = time.time()
    seq_adj, struct_adj = build_neighbor_graphs_residue_parallel(
        uids, res_lengths, ref_seqs, pdb_dir,
        n_jobs=-1, contact_cutoff=8.0, seq_gap_min=12)
    A_struct, deg_struct = adj_list_to_sparse(struct_adj, n_res_total)
    A_seq,    deg_seq    = adj_list_to_sparse(seq_adj,    n_res_total)
    log(f"  struct edges: {A_struct.nnz:,}  seq edges: {A_seq.nnz:,}  ({time.time()-t0:.1f}s)")
    del struct_adj, seq_adj

    pair_rows = []
    traj_rows = []

    # ───── Phase A: ESM-2 vs RITA ─────
    for label, e_l, r_l in DEPTHS_ER:
        log(f"\n=== ESM-2 L{e_l} vs RITA L{r_l}  (depth {label}%) ===")

        log(f"  ESM-2 layer {e_l}:")
        t_cell = time.time()
        c_e, sig_e, nres, s_e = per_protein_contribs_multi(
            ROOT/f"outputs_layerwise/esm2/layer_{e_l}",
            A_struct, deg_struct, A_seq, deg_seq,
            perm_indices, protein_offsets)
        log(f"  ESM-2 cache built ({time.time()-t_cell:.0f}s)")

        log(f"  RITA layer {r_l}:")
        t_cell = time.time()
        c_r, sig_r, _, s_r = per_protein_contribs_multi(
            ROOT/f"outputs_layerwise/rita/layer_{r_l}",
            A_struct, deg_struct, A_seq, deg_seq,
            perm_indices, protein_offsets)
        log(f"  RITA cache built ({time.time()-t_cell:.0f}s)")

        # pair bootstraps for all 5 variants × 2 splits
        for variant in ["struct_topdec", "seq_topdec",
                        "struct_0.5s", "struct_1.0s", "struct_2.0s"]:
            for split, indices, W in [("full", full_indices, W_full),
                                       ("val",  val_indices,  W_val)]:
                w_pt = np.zeros(n_proteins); w_pt[indices] = 1
                d_pt = cohens_d(
                    delta_under_w(c_e[variant], sig_e, nres, w_pt),
                    delta_under_w(c_r[variant], sig_r, nres, w_pt))
                t0 = time.time()
                trace = boot_pair(c_e[variant], sig_e, nres,
                                  c_r[variant], sig_r, nres, W)
                pair_rows.append(dict(
                    pair="esm2_vs_rita", variant=variant,
                    rel_depth=f"{label}%", layer_pair=f"L{e_l}/L{r_l}",
                    s_esm=s_e, s_rita=s_r, split=split,
                    **cell_summary(d_pt, trace)))
                log(f"    {variant:14s} {split:4s} d_pt={d_pt:+.4f}  boot {time.time()-t0:.0f}s")

        # within-model trajectory (struct, full only)
        for model_name, c, sig in [("esm2", c_e, sig_e), ("rita", c_r, sig_r)]:
            w_pt = np.ones(n_proteins)
            mean_pt = float(delta_under_w(c["struct_topdec"], sig, nres, w_pt).mean())
            trace = boot_within_mean(c["struct_topdec"], sig, nres, W_full)
            sd = trace.std(ddof=1)
            traj_rows.append(dict(
                model=model_name, rel_depth=f"{label}%",
                layer=e_l if model_name=="esm2" else r_l,
                mean_point=mean_pt, mean_boot=float(trace.mean()),
                boot_sd=float(sd),
                ci_lo=mean_pt - 1.96*sd, ci_hi=mean_pt + 1.96*sd))

        del c_e, c_r, sig_e, sig_r
        # Save partial after each depth
        pd.DataFrame(pair_rows).to_csv(OUT/"v2_cis_pair_esm_rita.csv", index=False)
        pd.DataFrame(traj_rows).to_csv(OUT/"v2_cis_trajectory.csv", index=False)
        log(f"  → wrote partial CSVs ({len(pair_rows)} pair rows, {len(traj_rows)} traj rows)")

    # ───── Phase B: ProtT5 enc vs dec ─────
    pt5_pair_rows = []
    for label, L in DEPTHS_PT5:
        log(f"\n=== ProtT5 enc vs dec L{L}  (depth {label}%) ===")
        log(f"  ProtT5 enc layer {L}:")
        t_cell = time.time()
        c_e, sig_e, nres, _ = per_protein_contribs_multi(
            ROOT/f"outputs_layerwise/prott5_enc/layer_{L}",
            A_struct, deg_struct, A_seq, deg_seq,
            perm_indices, protein_offsets)
        log(f"  enc cache built ({time.time()-t_cell:.0f}s)")

        log(f"  ProtT5 dec layer {L}:")
        t_cell = time.time()
        c_d, sig_d, _, _ = per_protein_contribs_multi(
            ROOT/f"outputs_layerwise/prott5_dec/layer_{L}",
            A_struct, deg_struct, A_seq, deg_seq,
            perm_indices, protein_offsets)
        log(f"  dec cache built ({time.time()-t_cell:.0f}s)")

        for variant in ["struct_topdec", "seq_topdec"]:
            for split, indices, W in [("full", full_indices, W_full),
                                       ("val",  val_indices,  W_val)]:
                w_pt = np.zeros(n_proteins); w_pt[indices] = 1
                d_pt = cohens_d(
                    delta_under_w(c_e[variant], sig_e, nres, w_pt),
                    delta_under_w(c_d[variant], sig_d, nres, w_pt))
                t0 = time.time()
                trace = boot_pair(c_e[variant], sig_e, nres,
                                  c_d[variant], sig_d, nres, W)
                pt5_pair_rows.append(dict(
                    pair="pt5_enc_vs_dec", variant=variant,
                    rel_depth=f"{label}%", layer=f"L{L}", split=split,
                    **cell_summary(d_pt, trace)))
                log(f"    {variant:14s} {split:4s} d_pt={d_pt:+.4f}  boot {time.time()-t0:.0f}s")

        # within-model trajectory
        for model_name, c, sig in [("prott5_enc", c_e, sig_e), ("prott5_dec", c_d, sig_d)]:
            w_pt = np.ones(n_proteins)
            mean_pt = float(delta_under_w(c["struct_topdec"], sig, nres, w_pt).mean())
            trace = boot_within_mean(c["struct_topdec"], sig, nres, W_full)
            sd = trace.std(ddof=1)
            traj_rows.append(dict(
                model=model_name, rel_depth=f"{label}%", layer=L,
                mean_point=mean_pt, mean_boot=float(trace.mean()),
                boot_sd=float(sd),
                ci_lo=mean_pt - 1.96*sd, ci_hi=mean_pt + 1.96*sd))

        del c_e, c_d, sig_e, sig_d
        pd.DataFrame(pt5_pair_rows).to_csv(OUT/"v2_cis_pair_pt5.csv", index=False)
        pd.DataFrame(traj_rows).to_csv(OUT/"v2_cis_trajectory.csv", index=False)
        log(f"  → wrote partial CSVs ({len(pt5_pair_rows)} pt5 rows, {len(traj_rows)} traj rows)")

    log("\nALL DONE.")


if __name__ == "__main__":
    main()
