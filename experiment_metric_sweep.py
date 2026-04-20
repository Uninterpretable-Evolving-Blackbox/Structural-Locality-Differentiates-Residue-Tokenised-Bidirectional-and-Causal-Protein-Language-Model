#!/usr/bin/env python3
"""
experiment_metric_sweep.py — H1/H2' robustness to metric hyperparameters.

Sweeps seq_gap_min ∈ {8, 12, 24} × topk_frac ∈ {0.05, 0.10, 0.20} on the
main seed=42 run. For each (model, layer, seq_gap_min) it computes smoothed
activation matrices ONCE and derives Cohen's d at all 3 topk_fracs from the
same matmuls — this saves ~3× over the naive cell-by-cell loop.

For ProtGPT2, the sequential adjacency excludes intra-BPE-token ±1/±2
pairs (per Option 2 / H2'). For ESM-2 it's the standard residue adjacency.

Output:
  results_metric_sweep/
    {model}/layer_{N}/cell_sg{s}_tk{t}.csv      per-feature deltas per cell
    sweep_raw.csv                               concat of all per-cell deltas
    h1_sweep.csv                                H1 Cohen's d + MW p per cell × depth
    h2prime_sweep.csv                           H2' Cohen's d + MW p per cell × depth
    sweep_summary.txt                           readable pass/fail per cell

Usage:
  python experiment_metric_sweep.py --out results_metric_sweep --n-shuffles 5
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy import stats
from joblib import Parallel, delayed, cpu_count

from cpu_stage import (
    load_layer, load_ref_seqs,
    build_neighbor_graphs_residue_parallel,
    adj_list_to_sparse, build_protein_permutations,
    build_protgpt2_projection,
)
from experiment_bpe_correction import (
    build_bpe_token_map, build_bpe_corrected_seq_neighbors,
    build_original_seq_neighbors,
)

ROOT = Path(__file__).resolve().parent

LAYERS = {
    "esm2":     [0, 8, 16, 24, 32],
    "protgpt2": [0, 9, 18, 27, 35],
}

# (depth_label, esm2_layer, protgpt2_layer)
MATCHED_PAIRS = [
    ("0%",    0,  0),
    ("25%",   8,  9),
    ("50%",  16, 18),
    ("75%",  24, 27),
    ("100%", 32, 35),
]

SEQ_GAP_MINS = [8, 12, 24]
TOPK_FRACS   = [0.05, 0.10, 0.20]


# ---------------------------------------------------------------------
#  Core: multi-topk Cohen's d from a shared smoothed matrix
# ---------------------------------------------------------------------

def _cohens_d_multi_topk(acts_chunk, A_sp, deg, global_stds, topk_fracs):
    """Like _cohens_d_vectorized but computes d at several topk_fracs from one matmul."""
    n_res, n_feat = acts_chunk.shape
    nbr_sums = np.asarray(A_sp @ acts_chunk, dtype=np.float32)
    has_nbrs = deg > 0
    nbr_sums[has_nbrs] /= deg[has_nbrs, None]
    nbr_sums[~has_nbrs] = 0.0
    smoothed = nbr_sums
    global_mean = smoothed.mean(axis=0)
    out = {}
    for tf in topk_fracs:
        thresh = np.percentile(acts_chunk, 100.0 * (1.0 - tf), axis=0)
        active = acts_chunk > thresh[None, :]
        n_active = active.sum(axis=0).astype(np.float32)
        active_sum = (smoothed * active).sum(axis=0)
        n_safe = n_active.copy(); n_safe[n_safe == 0] = 1.0
        active_mean = active_sum / n_safe
        d = (active_mean - global_mean) / (global_stds + 1e-6)
        d[n_active < 5] = 0.0
        out[tf] = d.astype(np.float32)
    return out


def _sweep_chunk(chunk_idx, chunk_size, Z, A_proj,
                 A_struct, deg_struct, A_seq, deg_seq,
                 perm_indices, n_features, topk_fracs):
    """Per chunk of features: compute obs + shuffle deltas for struct + seq across topk_fracs."""
    i = chunk_idx * chunk_size
    end = min(i + chunk_size, n_features)
    if A_proj is None:
        acts = np.asarray(Z[:, i:end], dtype=np.float32)
    else:
        acts = np.asarray(A_proj @ np.asarray(Z[:, i:end], dtype=np.float32),
                          dtype=np.float32)
    gstds = np.std(acts, axis=0).astype(np.float32)
    struct_obs = _cohens_d_multi_topk(acts, A_struct, deg_struct, gstds, topk_fracs)
    seq_obs    = _cohens_d_multi_topk(acts, A_seq,    deg_seq,    gstds, topk_fracs)

    cs = end - i
    struct_sh = {tf: np.zeros(cs, dtype=np.float32) for tf in topk_fracs}
    seq_sh    = {tf: np.zeros(cs, dtype=np.float32) for tf in topk_fracs}
    for perm in perm_indices:
        acts_p = acts[perm]
        sr = _cohens_d_multi_topk(acts_p, A_struct, deg_struct, gstds, topk_fracs)
        sq = _cohens_d_multi_topk(acts_p, A_seq,    deg_seq,    gstds, topk_fracs)
        for tf in topk_fracs:
            struct_sh[tf] += sr[tf]
            seq_sh[tf]    += sq[tf]
    n_sh = max(len(perm_indices), 1)
    for tf in topk_fracs:
        struct_sh[tf] /= n_sh
        seq_sh[tf]    /= n_sh

    idx = np.arange(i, end, dtype=np.int32)
    return idx, struct_obs, seq_obs, struct_sh, seq_sh


# ---------------------------------------------------------------------
#  Per-(model, layer) sweep
# ---------------------------------------------------------------------

def run_model_layer(outputs_root: Path, out_root: Path, model: str, layer: int,
                    n_shuffles: int, n_jobs: int, pdb_dir: Path):
    layer_dir = outputs_root / model / f"layer_{layer}"
    print(f"\n  [{model}/layer_{layer}]")
    Z, uids, tok_lengths = load_layer(layer_dir)
    ref_seqs = load_ref_seqs(layer_dir)

    # Residue lengths / offsets / projection
    if model == "protgpt2":
        A_proj, res_offsets, res_lengths = build_protgpt2_projection(
            uids, ref_seqs, tok_lengths, "nferruz/ProtGPT2")
        # BPE token map for inter-token adjacency
        token_ids, _, _ = build_bpe_token_map(
            uids, ref_seqs, tok_lengths, "nferruz/ProtGPT2")
        seq_adj = build_bpe_corrected_seq_neighbors(
            uids, res_lengths, res_offsets, token_ids)
        seq_kind = "inter-token"
    else:
        A_proj = None
        res_lengths = tok_lengths.astype(np.int32)
        res_offsets = {}; off = 0
        for uid, Lr in zip(uids, res_lengths):
            res_offsets[uid] = off; off += int(Lr)
        seq_adj = build_original_seq_neighbors(uids, res_lengths, res_offsets)
        seq_kind = "standard"

    n_res = int(np.sum(res_lengths))
    A_seq, deg_seq = adj_list_to_sparse(seq_adj, n_res)
    print(f"    n_res={n_res:,}, seq edges ({seq_kind}): {A_seq.nnz:,}")

    perm_indices = build_protein_permutations(res_lengths, n_shuffles)

    n_features = int(Z.shape[1])
    chunk_size = 256
    n_chunks = (n_features + chunk_size - 1) // chunk_size

    mem_per_worker = n_res * chunk_size * 4 * 5 / 1e9
    mem_budget_gb = float(os.environ.get("CPU_STAGE_MEM_GB", 100.0))
    max_safe_jobs = max(1, int(mem_budget_gb / max(mem_per_worker, 0.1)))
    eff_jobs = min(n_jobs if n_jobs > 0 else cpu_count(), max_safe_jobs)

    # Build ALL 3 struct adjacencies up front (sparse, cheap in memory)
    struct_adj_by_sg = {}
    for sg in SEQ_GAP_MINS:
        print(f"    Building struct adj seq_gap_min={sg}...")
        # Need per-protein graphs ONLY for struct — reuse seq_adj we already have
        # We re-use cpu_stage's builder but throw away its seq half.
        _, struct_adj = build_neighbor_graphs_residue_parallel(
            uids, res_lengths, ref_seqs, pdb_dir,
            n_jobs=eff_jobs, contact_cutoff=8.0, seq_gap_min=sg)
        struct_adj_by_sg[sg] = struct_adj

    per_cell_rows = []

    for sg in SEQ_GAP_MINS:
        A_struct, deg_struct = adj_list_to_sparse(struct_adj_by_sg[sg], n_res)
        print(f"    seq_gap_min={sg}: struct edges {A_struct.nnz:,} — "
              f"sweeping topk_fracs {TOPK_FRACS}...")
        t0 = time.time()
        results = Parallel(n_jobs=eff_jobs, verbose=1)(
            delayed(_sweep_chunk)(
                ci, chunk_size, Z, A_proj,
                A_struct, deg_struct, A_seq, deg_seq,
                perm_indices, n_features, TOPK_FRACS)
            for ci in range(n_chunks))
        print(f"      done in {time.time()-t0:.0f}s")

        # Assemble deltas for each topk_frac
        for tf in TOPK_FRACS:
            all_idx = np.concatenate([r[0] for r in results])
            str_obs = np.concatenate([r[1][tf] for r in results])
            seq_obs = np.concatenate([r[2][tf] for r in results])
            str_sh  = np.concatenate([r[3][tf] for r in results])
            seq_sh  = np.concatenate([r[4][tf] for r in results])
            order = np.argsort(all_idx)
            df_cell = pd.DataFrame({
                "feature_idx":  all_idx[order].astype(np.int32),
                "struct_delta": (str_obs - str_sh)[order],
                "seq_delta":    (seq_obs - seq_sh)[order],
            })
            df_cell["model"] = model
            df_cell["layer"] = layer
            df_cell["seq_gap_min"] = sg
            df_cell["topk_frac"] = tf
            df_cell["seq_kind"] = seq_kind
            per_cell_rows.append(df_cell)
            # Save per-cell CSV
            cell_dir = out_root / model / f"layer_{layer}"
            cell_dir.mkdir(parents=True, exist_ok=True)
            df_cell.to_csv(
                cell_dir / f"cell_sg{sg}_tk{int(tf*100):02d}.csv", index=False)

    return pd.concat(per_cell_rows, ignore_index=True)


# ---------------------------------------------------------------------
#  Aggregation: H1 + H2' per cell × depth
# ---------------------------------------------------------------------

def cohens_d(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    pooled = np.sqrt((a.std(ddof=1)**2 + b.std(ddof=1)**2) / 2.0 + 1e-12)
    return float((a.mean() - b.mean()) / pooled)


def aggregate(out_root: Path, sweep_df: pd.DataFrame):
    h1_rows, h2_rows = [], []
    for depth_label, esm_l, pg2_l in MATCHED_PAIRS:
        for sg in SEQ_GAP_MINS:
            for tf in TOPK_FRACS:
                esm = sweep_df[(sweep_df.model == "esm2") & (sweep_df.layer == esm_l)
                               & (sweep_df.seq_gap_min == sg) & (sweep_df.topk_frac == tf)]
                pg2 = sweep_df[(sweep_df.model == "protgpt2") & (sweep_df.layer == pg2_l)
                               & (sweep_df.seq_gap_min == sg) & (sweep_df.topk_frac == tf)]
                if len(esm) == 0 or len(pg2) == 0:
                    continue

                # H1: ESM-2 struct > ProtGPT2 struct
                a, b = esm.struct_delta.values, pg2.struct_delta.values
                d  = cohens_d(a, b)
                _, p = stats.mannwhitneyu(a, b, alternative="greater")
                h1_rows.append(dict(depth=depth_label, seq_gap_min=sg, topk_frac=tf,
                                    esm_layer=esm_l, pg2_layer=pg2_l,
                                    cohens_d=d, MW_p=float(p),
                                    esm_mean=float(a.mean()), pg2_mean=float(b.mean()),
                                    significant=bool(p < 0.05 and d > 0)))

                # H2': ESM-2 seq > ProtGPT2 seq (inter-token for pg2)
                a2, b2 = esm.seq_delta.values, pg2.seq_delta.values
                d2 = cohens_d(a2, b2)
                _, p2 = stats.mannwhitneyu(a2, b2, alternative="greater")
                h2_rows.append(dict(depth=depth_label, seq_gap_min=sg, topk_frac=tf,
                                    esm_layer=esm_l, pg2_layer=pg2_l,
                                    cohens_d=d2, MW_p=float(p2),
                                    esm_mean=float(a2.mean()), pg2_mean=float(b2.mean()),
                                    significant=bool(p2 < 0.05 and d2 > 0)))

    h1_df = pd.DataFrame(h1_rows)
    h2_df = pd.DataFrame(h2_rows)
    h1_df.to_csv(out_root / "h1_sweep.csv", index=False)
    h2_df.to_csv(out_root / "h2prime_sweep.csv", index=False)

    # Readable summary
    lines = ["Metric sweep — H1 and H2' robustness\n",
             "=" * 72 + "\n\n",
             f"Grid: seq_gap_min ∈ {SEQ_GAP_MINS} × topk_frac ∈ {TOPK_FRACS}\n",
             "(seq_gap_min affects only structural locality; for H2' only topk_frac varies effectively.)\n\n"]
    for name, df in [("H1 (ESM-2 struct > ProtGPT2 struct)", h1_df),
                     ("H2' (ESM-2 seq > ProtGPT2 seq, pg2 inter-token)", h2_df)]:
        lines.append(f"\n--- {name} ---\n")
        cells = df.groupby(["seq_gap_min", "topk_frac"])["significant"].agg(["sum", "count"])
        lines.append(cells.to_string() + "\n")
        total_sig = int(df.significant.sum()); total = len(df)
        lines.append(f"\nTotal: {total_sig} / {total} depth-cells significant "
                     f"({100*total_sig/total:.0f}%)\n")
    with open(out_root / "sweep_summary.txt", "w") as f:
        f.writelines(lines)
    print(f"\nSummary written → {out_root}/sweep_summary.txt")


# ---------------------------------------------------------------------
#  MAIN
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-dir", default="outputs_layerwise")
    ap.add_argument("--out", default="results_metric_sweep")
    ap.add_argument("--n-shuffles", type=int, default=5)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--pdb-dir", default="cache/pdb_files")
    ap.add_argument("--models", nargs="+", default=["esm2", "protgpt2"])
    ap.add_argument("--layers-only", nargs="+", type=int, default=None,
                    help="Only process these layer indices (smoke test)")
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()

    outputs_root = (ROOT / args.outputs_dir) if not Path(args.outputs_dir).is_absolute() else Path(args.outputs_dir)
    out_root = Path(args.out); out_root.mkdir(parents=True, exist_ok=True)
    pdb_dir = Path(args.pdb_dir)
    print(f"Source outputs: {outputs_root}\nResults dir:    {out_root}\n")

    if not args.aggregate_only:
        all_frames = []
        t0 = time.time()
        for model in args.models:
            layers = LAYERS[model]
            if args.layers_only:
                layers = [L for L in layers if L in args.layers_only]
            for layer in layers:
                df = run_model_layer(outputs_root, out_root, model, layer,
                                     args.n_shuffles, args.n_jobs, pdb_dir)
                all_frames.append(df)
        big = pd.concat(all_frames, ignore_index=True)
        big.to_csv(out_root / "sweep_raw.csv", index=False)
        print(f"\nSweep raw CSV: {out_root/'sweep_raw.csv'}")
        print(f"Total sweep wall: {(time.time()-t0)/60:.1f} min")
    else:
        big = pd.read_csv(out_root / "sweep_raw.csv")
        print(f"Loaded existing sweep_raw.csv ({len(big)} rows)")

    aggregate(out_root, big)


if __name__ == "__main__":
    main()
