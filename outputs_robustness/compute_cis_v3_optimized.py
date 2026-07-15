#!/usr/bin/env python3
"""
v3-optimized — Item 3 (val cutoff + window CIs) with maximum parallelism.

Three key optimizations vs v3:
  1. BATCHED bootstrap: stack all 1000 weights as (B, n_proteins) and do a
     single W @ contribs matmul instead of 1000 small ones. BLAS-multithreaded.
     ~100x speedup on the inner loop (100s → ~1s).
  2. ThreadPoolExecutor for the 3 independent reduceat calls in per_protein_sums.
     numpy releases GIL during reduceat → ~2-3x speedup on the bottleneck.
  3. Single Z load per (model, layer); sigma/percentile/active_mask shared
     across the 4 adjacency variants (struct 6, struct 10, seq ±1, seq ±4).
     Saves ~3 min/cell on redundant percentile+sigma computation.

Targets: 9 depths × 2 models × 4 adjacencies = 72 cells of cache build, with
each cell using a single variant (top-decile of Z); paired full+val bootstrap.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "18")
os.environ.setdefault("MKL_NUM_THREADS", "18")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "18")

import json, sys, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs_robustness"
PROG = OUT / "v3opt_progress.txt"

from cpu_stage import (
    load_layer, load_ref_seqs,
    build_neighbor_graphs_residue_parallel,
    adj_list_to_sparse, build_protein_permutations,
)
from cluster_bootstrap import clusters_for_uids, make_cluster_weights, cluster_stats

N_BOOT = 1000
N_SHUF = 5
TOPK_FRAC = 0.10
SEED = 42
# Cluster bootstrap controls (env-var driven). CLUSTER_LEVEL=fold (default) ->
# honest SCOPe-fold-clustered CIs; =protein -> original protein-level bootstrap.
CLUSTER_LEVEL = os.environ.get("CLUSTER_LEVEL", "fold")
CLUSTER_TWO_STAGE = os.environ.get("CLUSTER_TWO_STAGE", "1") == "1"
FASTA = ROOT / "cache/scope_40.fa"
SFX = "" if CLUSTER_LEVEL == "protein" else f"_{CLUSTER_LEVEL}"

DEPTHS_ER = [
    ("0",   0,  0), ("13",  4,  3), ("25",  8,  6),
    ("38", 12,  9), ("50", 16, 12), ("63", 20, 15),
    ("75", 24, 18), ("88", 28, 21), ("100",32, 23),
]


def log(msg):
    s = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(s, flush=True)
    with open(PROG, "a") as f:
        f.write(s + "\n")


def cohens_d(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    pooled = np.sqrt((a.std(ddof=1)**2 + b.std(ddof=1)**2)/2 + 1e-12)
    return float((a.mean() - b.mean()) / pooled)


def per_protein_sums_threaded(active_mask_f32, smoothed, starts):
    """3 reduceat calls in parallel via ThreadPool. Returns (act_sum, act_cnt, glob_sum)."""
    prod = smoothed * active_mask_f32
    with ThreadPoolExecutor(max_workers=3) as ex:
        f_as = ex.submit(np.add.reduceat, prod,             starts, axis=0)
        f_ac = ex.submit(np.add.reduceat, active_mask_f32,  starts, axis=0)
        f_gs = ex.submit(np.add.reduceat, smoothed,         starts, axis=0)
        return f_as.result(), f_ac.result(), f_gs.result()


def build_one_adjacency_cache(active_mask_f32, A, deg, perm_indices, starts, Z):
    """Build single-variant cache for one adjacency, given Z + precomputed mask."""
    has = deg > 0
    n_res, n_feat = Z.shape

    # Smoothed under this adjacency
    nbr = (A @ Z).astype(np.float32)
    smoothed = np.zeros_like(Z)
    smoothed[has] = nbr[has] / deg[has, None]
    del nbr

    obs_as, obs_ac, obs_gs = per_protein_sums_threaded(active_mask_f32, smoothed, starts)

    shf_as = np.zeros_like(obs_as)
    shf_ac = np.zeros_like(obs_as)
    shf_gs = np.zeros_like(obs_as)
    for perm in perm_indices:
        Zp = Z[perm]
        nbrp = (A @ Zp).astype(np.float32)
        smp = np.zeros_like(Zp); smp[has] = nbrp[has] / deg[has, None]
        # mask under perm: same column distribution → same thresh, but applied to permuted Z
        # We need active_mask for permuted Z. For top-decile (column-wise threshold),
        # the threshold is unchanged, but the mask values are different (permuted).
        # active_mask[perm] gives us the permuted mask.
        actp_f32 = active_mask_f32[perm]
        a_s, a_c, g_s = per_protein_sums_threaded(actp_f32, smp, starts)
        shf_as += a_s; shf_ac += a_c; shf_gs += g_s
        del Zp, nbrp, smp, actp_f32
    shf_as /= len(perm_indices); shf_ac /= len(perm_indices); shf_gs /= len(perm_indices)

    return dict(obs_as=obs_as, obs_ac=obs_ac, obs_gs=obs_gs,
                shf_as=shf_as, shf_ac=shf_ac, shf_gs=shf_gs)


def boot_pair_batched(c_a, sigma_a, n_res_a, c_b, sigma_b, n_res_b, W):
    """Batched bootstrap: ONE big matmul per contribution array.

    W: (B, n_proteins) — bootstrap weight matrix
    Returns d_boot: (B,) array of cohens_d per resample.
    """
    Wf = W.astype(np.float64)

    def per_cache(c, sigma, n_res):
        a_s = Wf @ c["obs_as"]   # (B, n_feat)
        a_c = Wf @ c["obs_ac"]
        g_s = Wf @ c["obs_gs"]
        s_s = Wf @ c["shf_as"]
        s_c = Wf @ c["shf_ac"]
        s_g = Wf @ c["shf_gs"]
        n   = Wf @ n_res         # (B,)
        n_safe = np.maximum(n, 1)[:, None]

        am = np.where(a_c > 0, a_s / np.maximum(a_c, 1), 0)
        gm = g_s / n_safe
        sm = np.where(s_c > 0, s_s / np.maximum(s_c, 1), 0)
        sg = s_g / n_safe

        d = ((am - gm) - (sm - sg)) / sigma
        return d.astype(np.float64)

    d_a = per_cache(c_a, sigma_a, n_res_a)  # (B, n_feat)
    d_b = per_cache(c_b, sigma_b, n_res_b)

    # Vectorised cohens_d per row
    ma = d_a.mean(axis=1); mb = d_b.mean(axis=1)
    va = d_a.var(axis=1, ddof=1); vb = d_b.var(axis=1, ddof=1)
    return (ma - mb) / np.sqrt((va + vb) / 2.0 + 1e-12)


def make_weights(n_proteins, indices, n_boot, rng):
    out = np.zeros((n_boot, n_proteins), dtype=np.int32)
    for b in range(n_boot):
        sample = rng.choice(indices, size=len(indices), replace=True)
        u, cnt = np.unique(sample, return_counts=True)
        out[b, u] = cnt
    return out


def cell_summary(d_point, trace):
    sd = trace.std(ddof=1)
    return dict(d_point=d_point, d_boot_mean=float(trace.mean()),
                boot_sd=float(sd),
                ci_normal_lo=d_point - 1.96*sd, ci_normal_hi=d_point + 1.96*sd,
                ci_pct_lo=float(np.percentile(trace, 2.5)),
                ci_pct_hi=float(np.percentile(trace, 97.5)))


def build_seq_adj_window(uids, res_lengths, window, n_res_total):
    rows, cols = [], []
    offset = 0
    for Lr in res_lengths:
        Lr = int(Lr)
        for r in range(Lr):
            for d in range(-window, window+1):
                if d == 0: continue
                rr = r + d
                if 0 <= rr < Lr:
                    rows.append(offset + r); cols.append(offset + rr)
        offset += Lr
    from scipy import sparse
    A = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32),
         (np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32))),
        shape=(n_res_total, n_res_total))
    deg = np.asarray(A.sum(axis=1)).ravel().astype(np.float32)
    return A, deg


def correctness_test():
    """Quick sanity test that batched bootstrap matches the loop version on tiny data."""
    log("=== correctness test: batched vs loop bootstrap ===")
    np.random.seed(0)
    n_proteins, n_feat, B = 50, 100, 30
    c_a = {k: np.random.randn(n_proteins, n_feat).astype(np.float32) for k in
           ["obs_as","obs_ac","obs_gs","shf_as","shf_ac","shf_gs"]}
    c_a["obs_ac"] = np.abs(c_a["obs_ac"]) + 0.5  # avoid div by zero
    c_a["shf_ac"] = np.abs(c_a["shf_ac"]) + 0.5
    c_b = {k: np.random.randn(n_proteins, n_feat).astype(np.float32) for k in c_a}
    c_b["obs_ac"] = np.abs(c_b["obs_ac"]) + 0.5
    c_b["shf_ac"] = np.abs(c_b["shf_ac"]) + 0.5
    sigma_a = np.abs(np.random.randn(n_feat).astype(np.float32)) + 0.1
    sigma_b = np.abs(np.random.randn(n_feat).astype(np.float32)) + 0.1
    n_res_a = np.abs(np.random.randn(n_proteins).astype(np.float32)) * 100 + 10
    n_res_b = n_res_a.copy()

    W = np.random.randint(0, 5, (B, n_proteins)).astype(np.int32)

    # Loop version
    def delta_loop(c, sigma, n_res, w):
        w = w.astype(np.float64)
        a_s, a_c, g_s = w@c["obs_as"], w@c["obs_ac"], w@c["obs_gs"]
        s_s, s_c, s_g = w@c["shf_as"], w@c["shf_ac"], w@c["shf_gs"]
        n = w@n_res
        am = np.where(a_c > 0, a_s/np.maximum(a_c,1), 0)
        gm = g_s/max(n, 1)
        sm = np.where(s_c > 0, s_s/np.maximum(s_c,1), 0)
        sg = s_g/max(n, 1)
        return ((am - gm) - (sm - sg)) / sigma

    d_loop = np.empty(B)
    for i in range(B):
        d_a = delta_loop(c_a, sigma_a, n_res_a, W[i])
        d_b = delta_loop(c_b, sigma_b, n_res_b, W[i])
        d_loop[i] = cohens_d(d_a, d_b)

    d_batched = boot_pair_batched(c_a, sigma_a, n_res_a, c_b, sigma_b, n_res_b, W)

    max_err = np.max(np.abs(d_loop - d_batched))
    log(f"  max abs error (loop vs batched): {max_err:.2e}")
    assert max_err < 1e-6, f"BATCHED BOOTSTRAP FAILS CORRECTNESS: max err {max_err:.2e}"
    log("  ✓ batched bootstrap matches loop version exactly")


def main():
    PROG.unlink(missing_ok=True)
    log("=" * 76)
    log("  v3-optimized: batched bootstrap + threaded reductions, B=1000")
    log("=" * 76)

    correctness_test()

    layer0 = ROOT / "outputs_layerwise/esm2/layer_0"
    Z0, uids, lengths = load_layer(layer0)
    res_lengths = lengths.astype(np.int32)
    n_proteins = len(res_lengths)
    n_res_total = int(res_lengths.sum())
    protein_offsets = np.concatenate([[0], np.cumsum(res_lengths.astype(np.int64))])
    starts = protein_offsets[:-1]
    val_uids = json.loads((layer0/"META.json").read_text())["val_uids"]
    val_indices = np.array([i for i, u in enumerate(uids) if u in set(val_uids)])
    full_indices = np.arange(n_proteins)
    log(f"  n_proteins={n_proteins}, val={len(val_indices)}, n_res={n_res_total}")
    del Z0
    ref_seqs = load_ref_seqs(layer0)
    pdb_dir = ROOT / "cache/pdb_files"

    rng = np.random.default_rng(SEED)
    perm_indices = build_protein_permutations(res_lengths, N_SHUF)
    cluster_id = clusters_for_uids(uids, FASTA, level=CLUSTER_LEVEL)
    cst = cluster_stats(cluster_id, full_indices)
    log(f"  cluster-level={CLUSTER_LEVEL} ({'two-stage' if CLUSTER_TWO_STAGE else 'one-stage'}): "
        f"{cst['n_clusters']} clusters / {cst['n_units']} proteins (mean {cst['mean_cluster_size']:.2f})")
    W_full = make_cluster_weights(n_proteins, cluster_id, full_indices, N_BOOT, rng,
                                  two_stage=CLUSTER_TWO_STAGE)
    W_val  = make_cluster_weights(n_proteins, cluster_id, val_indices,  N_BOOT,
                                  np.random.default_rng(SEED + 1), two_stage=CLUSTER_TWO_STAGE)

    # ---- Build all 4 adjacencies up front ----
    adj = {}
    for cutoff in [6.0, 10.0]:
        log(f"Building struct adjacency at {cutoff} A ...")
        t0 = time.time()
        _, struct_adj = build_neighbor_graphs_residue_parallel(
            uids, res_lengths, ref_seqs, pdb_dir, n_jobs=-1,
            contact_cutoff=cutoff, seq_gap_min=12)
        A, d = adj_list_to_sparse(struct_adj, n_res_total)
        adj[f'struct_{int(cutoff)}'] = (A, d)
        log(f"  struct_{int(cutoff)} edges={A.nnz:,}  ({time.time()-t0:.0f}s)")
        del struct_adj
    for window in [1, 4]:
        log(f"Building seq adjacency at ±{window} ...")
        t0 = time.time()
        A, d = build_seq_adj_window(uids, res_lengths, window, n_res_total)
        adj[f'seq_{window}'] = (A, d)
        log(f"  seq_{window} edges={A.nnz:,}  ({time.time()-t0:.0f}s)")

    VARIANT_KEYS = ["struct_6", "struct_10", "seq_1", "seq_4"]
    rows = []

    # ---- For each (depth, model_pair), one Z load per model, share across 4 adjacencies ----
    for label, e_l, r_l in DEPTHS_ER:
        log(f"\n=== depth {label}% — ESM-2 L{e_l} vs RITA L{r_l} ===")

        # Per-model: load Z, compute sigma + active_mask once, then loop over adjacencies
        caches_for_pair = {}  # caches_for_pair[(model, variant)] = cache dict
        meta = {}             # meta[model] = (sigma_j, n_residues)

        for model_name, layer in [("esm2", e_l), ("rita", r_l)]:
            t_model = time.time()
            log(f"  {model_name} L{layer}: load + setup")
            Z, _, _ = load_layer(ROOT/f"outputs_layerwise/{model_name}/layer_{layer}")
            Z = np.asarray(Z, dtype=np.float32)
            sigma_j = Z.std(axis=0, ddof=0).astype(np.float32) + 1e-6
            log(f"    sigma_j ({time.time()-t_model:.1f}s elapsed)")

            t = time.time()
            thresh_top = np.percentile(Z, 100*(1-TOPK_FRAC), axis=0).astype(np.float32)
            log(f"    percentile ({time.time()-t:.1f}s)")

            t = time.time()
            active_mask_f32 = (Z > thresh_top[None, :]).astype(np.float32)
            log(f"    active_mask ({time.time()-t:.1f}s)")

            n_residues = np.diff(protein_offsets).astype(np.float32)

            for vk in VARIANT_KEYS:
                t = time.time()
                A, d = adj[vk]
                cache = build_one_adjacency_cache(active_mask_f32, A, d,
                                                  perm_indices, starts, Z)
                caches_for_pair[(model_name, vk)] = cache
                log(f"    {vk:10s} cache built ({time.time()-t:.0f}s)")
            meta[model_name] = (sigma_j, n_residues)
            del Z, active_mask_f32, thresh_top
            log(f"  {model_name} done ({time.time()-t_model:.0f}s)")

        # ---- Bootstrap for each (variant, split) — batched, fast ----
        for vk in VARIANT_KEYS:
            c_e = caches_for_pair[("esm2", vk)]
            c_r = caches_for_pair[("rita", vk)]
            sig_e, nres_e = meta["esm2"]
            sig_r, nres_r = meta["rita"]

            for split, indices, W in [("full", full_indices, W_full),
                                       ("val",  val_indices,  W_val)]:
                w_pt = np.zeros((1, n_proteins), dtype=np.int32); w_pt[0, indices] = 1
                d_pt_arr = boot_pair_batched(c_e, sig_e, nres_e, c_r, sig_r, nres_r, w_pt)
                d_pt = float(d_pt_arr[0])

                t = time.time()
                trace = boot_pair_batched(c_e, sig_e, nres_e, c_r, sig_r, nres_r, W)
                rows.append(dict(
                    pair="esm2_vs_rita", variant=vk,
                    rel_depth=f"{label}%", layer_pair=f"L{e_l}/L{r_l}",
                    split=split, **cell_summary(d_pt, trace)))
                log(f"    {vk:10s} {split:4s}  d_pt={d_pt:+.4f}  batched-boot {time.time()-t:.1f}s")

        del caches_for_pair, meta
        pd.DataFrame(rows).to_csv(OUT/f"v3opt_cis_val_sweeps{SFX}.csv", index=False)
        log(f"  → wrote v3opt_cis_val_sweeps.csv ({len(rows)} rows)")

    log("\nALL v3-optimized DONE.")


if __name__ == "__main__":
    main()
