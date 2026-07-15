#!/usr/bin/env python3
"""
experiment_null_calibration.py — calibrate L_struct against a structural null
=============================================================================

Reviewer U2Yp asked: "how big is the ESM-2 vs RITA L_struct difference? Give a
condition under which L_struct should be ~0 as a baseline."

This recomputes the paper's L_struct (struct_delta = observed - within-protein
shuffle) but replaces the REAL Cα-contact graph with a DEGREE-MATCHED RANDOM
graph: each residue keeps its real number of structural neighbours, but the
partners are resampled at random from the same protein subject to the same
sequence-separation filter (|i-j| >= seq_gap_min). Spatial identity is destroyed
while degree, intra-protein locality, and separation are preserved.

If real-graph L_struct >> null-graph L_struct (~0), the metric is measuring
genuine 3D contact geometry, not graph density / degree artefacts — and the
null mean is the principled floor against which d=+1.22 should be read.

Reuses the exact L_struct kernel from cpu_stage.py.

Usage:
  python experiment_null_calibration.py --layer-dir outputs_layerwise/esm2/layer_16 \
    --pdb-dir cache/pdb_files --save-dir results_null/esm2_l16

  python experiment_null_calibration.py --layer-dir outputs_layerwise/esm2/layer_16 \
    --save-dir /tmp/null_smoke --max-features 512 --n-shuffles 2
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from cpu_stage import (
    load_layer, load_ref_seqs,
    build_neighbor_graphs_residue_parallel, adj_list_to_sparse,
    build_protein_permutations, _cohens_d_vectorized,
    DEFAULT_CONTACT_CUTOFF, DEFAULT_SEQ_GAP_MIN, DEFAULT_TOPK_FRAC,
)

warnings.filterwarnings("ignore")


def degree_matched_null(struct_adj, res_lengths, offsets, seq_gap_min, seed=42):
    """Build a degree-matched random adjacency list.

    For each residue, keep its real out-degree but resample partners uniformly
    from the same protein with |i-j| >= seq_gap_min.
    """
    rng = np.random.default_rng(seed)
    null = [[] for _ in range(len(struct_adj))]
    for p_idx, (base, L) in enumerate(zip(offsets, res_lengths)):
        L = int(L)
        for local in range(L):
            g = base + local
            deg = len(struct_adj[g])
            if deg == 0:
                continue
            lo_ok = np.arange(L)
            cand = lo_ok[np.abs(lo_ok - local) >= seq_gap_min]
            if cand.size == 0:
                continue
            k = min(deg, cand.size)
            picks = rng.choice(cand, size=k, replace=False)
            null[g] = [int(base + q) for q in picks]
    return null


def global_shuffle_null(struct_adj, n_res, seq_gap_min, seed=42):
    """Even stronger null: partners drawn from ANYWHERE (destroys intra-protein
    locality too). Keeps per-residue degree."""
    rng = np.random.default_rng(seed)
    null = [[] for _ in range(n_res)]
    for g in range(n_res):
        deg = len(struct_adj[g])
        if deg == 0:
            continue
        picks = rng.integers(0, n_res, size=deg)
        null[g] = [int(x) for x in picks]
    return null


def struct_delta_for_graph(Z, feat_idx, A, deg, perms, topk_frac, chunk=256):
    """Compute struct_delta (= observed d - mean shuffled d) per feature for one graph."""
    n_feat = len(feat_idx)
    out = np.zeros(n_feat, dtype=np.float32)
    for s in tqdm(range(0, n_feat, chunk), desc="    chunks", leave=False):
        cols = feat_idx[s:s + chunk]
        acts = np.asarray(Z[:, cols], dtype=np.float32)
        gstds = acts.std(axis=0).astype(np.float32)
        obs = _cohens_d_vectorized(acts, A, deg, gstds, topk_frac)
        sh = np.zeros(len(cols), dtype=np.float32)
        for perm in perms:
            sh += _cohens_d_vectorized(acts[perm], A, deg, gstds, topk_frac)
        if perms:
            sh /= len(perms)
        out[s:s + len(cols)] = obs - sh
    return out


def main():
    ap = argparse.ArgumentParser(description="Null calibration for L_struct")
    ap.add_argument("--layer-dir", required=True)
    ap.add_argument("--pdb-dir", default="cache/pdb_files")
    ap.add_argument("--n-shuffles", type=int, default=5)
    ap.add_argument("--contact-cutoff", type=float, default=DEFAULT_CONTACT_CUTOFF)
    ap.add_argument("--seq-gap-min", type=int, default=DEFAULT_SEQ_GAP_MIN)
    ap.add_argument("--topk-frac", type=float, default=DEFAULT_TOPK_FRAC)
    ap.add_argument("--nulls", default="degree_matched,global_shuffle")
    ap.add_argument("--max-features", type=int, default=0)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--save-dir", required=True)
    args = ap.parse_args()

    layer_dir = Path(args.layer_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    null_types = [x.strip() for x in args.nulls.split(",") if x.strip()]

    print("=" * 70)
    print("  NULL CALIBRATION FOR L_struct")
    print("=" * 70)

    Z, uids, lengths = load_layer(layer_dir)
    uids = [str(u) for u in uids]
    ref = load_ref_seqs(layer_dir)
    n_res = int(Z.shape[0])
    n_features = int(Z.shape[1])
    feat_idx = np.arange(n_features, dtype=np.int64)
    if args.max_features and args.max_features < n_features:
        feat_idx = feat_idx[:args.max_features]

    offsets, off = [], 0
    for L in lengths:
        offsets.append(off)
        off += int(L)

    print(f"  Building real Cα-contact graph ({args.contact_cutoff} Å, sep >= {args.seq_gap_min})...")
    _, struct_adj = build_neighbor_graphs_residue_parallel(
        uids, lengths, ref, Path(args.pdb_dir), n_jobs=args.n_jobs,
        contact_cutoff=args.contact_cutoff, seq_gap_min=args.seq_gap_min)
    A_real, deg_real = adj_list_to_sparse(struct_adj, n_res)
    print(f"    Real structural edges: {A_real.nnz:,}")

    perms = build_protein_permutations(lengths, args.n_shuffles)

    graphs = {"real": (A_real, deg_real)}
    if "degree_matched" in null_types:
        print("  Building degree-matched null graph...")
        null_adj = degree_matched_null(struct_adj, lengths, offsets, args.seq_gap_min)
        graphs["degree_matched"] = adj_list_to_sparse(null_adj, n_res)
        print(f"    Degree-matched edges: {graphs['degree_matched'][0].nnz:,}")
    if "global_shuffle" in null_types:
        print("  Building global-shuffle null graph...")
        gnull = global_shuffle_null(struct_adj, n_res, args.seq_gap_min)
        graphs["global_shuffle"] = adj_list_to_sparse(gnull, n_res)
        print(f"    Global-shuffle edges: {graphs['global_shuffle'][0].nnz:,}")

    results = {}
    for name, (A, deg) in graphs.items():
        print(f"  Computing struct_delta on graph: {name}")
        results[name] = struct_delta_for_graph(
            Z, feat_idx, A, deg, perms, args.topk_frac)

    df = pd.DataFrame({"feature_idx": feat_idx})
    for name, vals in results.items():
        df[f"struct_delta_{name}"] = vals
    df.to_csv(save_dir / "null_calibration_per_feature.csv", index=False)

    summary = {"layer_dir": str(layer_dir), "n_features": int(len(feat_idx))}
    for name, vals in results.items():
        summary[name] = {
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "frac_positive": float(np.mean(vals > 0)),
            "p95": float(np.percentile(vals, 95)),
        }
    real_mean = summary["real"]["mean"]
    for name in results:
        if name != "real":
            nm = summary[name]["mean"]
            summary[name]["real_to_null_ratio"] = float(real_mean / nm) if abs(nm) > 1e-9 else float("inf")
    (save_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # plot distributions
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = {"real": "#2196F3", "degree_matched": "#FF9800", "global_shuffle": "#9E9E9E"}
    for name, vals in results.items():
        ax.hist(vals, bins=80, alpha=0.55, density=True,
                label=f"{name} (mean {np.mean(vals):.4f})",
                color=colors.get(name))
    ax.axvline(0, color="k", ls="--", lw=1)
    ax.set_xlabel("struct_delta (L_struct)")
    ax.set_ylabel("density")
    ax.set_title(f"L_struct: real contact graph vs degree-matched null\n{layer_dir.name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_dir / "null_calibration.png", dpi=200)
    plt.close(fig)

    print("\n  Summary:")
    print(json.dumps(summary, indent=2))
    print(f"\n  Saved to {save_dir}/")


if __name__ == "__main__":
    main()
