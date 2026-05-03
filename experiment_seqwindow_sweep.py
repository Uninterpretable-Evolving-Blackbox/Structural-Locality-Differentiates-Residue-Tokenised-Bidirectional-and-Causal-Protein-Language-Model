#!/usr/bin/env python3
"""
experiment_seqwindow_sweep.py — sequential-window robustness sweep.

For each model in {esm2, rita, protgpt2} and each depth in the 9-point
densified grid, rebuilds the sequential neighbour adjacency at
window ∈ {±1, ±2, ±4} and recomputes per-feature seq_delta.

For ProtGPT2 we compute BOTH:
  * raw (residue-projected, including intra-BPE-token pairs)
  * inter-token (H2' style, excluding pairs within the same BPE token)

Outputs:
  results_seqwindow_sweep/
    {model}/layer_{N}/cell_w{w}[_inter].csv   per-feature seq_delta per cell
    window_raw.csv                             concat of all per-cell deltas
    h2_window_sweep.csv   H2 d for ESM-2/RITA and ESM-2/ProtGPT2 (both variants)
    window_sweep_summary.txt
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from joblib import Parallel, delayed, cpu_count

from cpu_stage import (
    load_layer, load_ref_seqs,
    adj_list_to_sparse, build_protein_permutations,
    build_protgpt2_projection,
)
from experiment_metric_sweep import _cohens_d_multi_topk
from experiment_bpe_correction import (
    build_bpe_token_map, build_bpe_corrected_seq_neighbors,
)

ROOT = Path(__file__).resolve().parent

LAYER_PLAN = {
    "esm2":     [0, 4, 8, 12, 16, 20, 24, 28, 32],
    "rita":     [0, 3, 6, 9, 12, 15, 18, 21, 23],
    "protgpt2": [0, 4, 9, 13, 18, 22, 27, 31, 35],
}
WINDOWS   = [1, 2, 4]
TOPK_FRAC = 0.10
N_BLOCKS  = {"esm2": 33, "rita": 24, "protgpt2": 36}


def build_seq_neighbors_window(uids, res_lengths, res_offsets, window: int):
    """±window sequential neighbours for each residue. Residue-level only."""
    n_res = int(np.sum(res_lengths))
    seq_adj = [[] for _ in range(n_res)]
    for uid, Lr in zip(uids, res_lengths):
        base = res_offsets[uid]; Lr = int(Lr)
        for r in range(Lr):
            for d in range(-window, window + 1):
                if d == 0: continue
                rr = r + d
                if 0 <= rr < Lr:
                    seq_adj[base + r].append(base + rr)
    return seq_adj


def build_bpe_window_intertoken(uids, res_lengths, res_offsets,
                                token_ids, window: int):
    """±window sequential adjacency, excluding residue pairs within same BPE token."""
    n_res = int(np.sum(res_lengths))
    seq_adj = [[] for _ in range(n_res)]
    # token_ids is list aligned to res idx giving the token id at each residue
    for uid, Lr in zip(uids, res_lengths):
        base = res_offsets[uid]; Lr = int(Lr)
        for r in range(Lr):
            tok_r = token_ids[base + r]
            for d in range(-window, window + 1):
                if d == 0: continue
                rr = r + d
                if 0 <= rr < Lr and token_ids[base + rr] != tok_r:
                    seq_adj[base + r].append(base + rr)
    return seq_adj


def _chunk(chunk_idx, chunk_size, Z, A_proj,
           A_seq, deg_seq, perm_indices, n_features, topk_frac):
    """Observed - shuffled seq_delta per feature."""
    i = chunk_idx * chunk_size
    end = min(i + chunk_size, n_features)
    if A_proj is None:
        acts = np.asarray(Z[:, i:end], dtype=np.float32)
    else:
        acts = np.asarray(A_proj @ np.asarray(Z[:, i:end], dtype=np.float32),
                          dtype=np.float32)
    gstds = np.std(acts, axis=0).astype(np.float32)
    seq_obs = _cohens_d_multi_topk(acts, A_seq, deg_seq, gstds, [topk_frac])[topk_frac]
    seq_sh = np.zeros(end - i, dtype=np.float32)
    for perm in perm_indices:
        sq = _cohens_d_multi_topk(acts[perm], A_seq, deg_seq, gstds, [topk_frac])[topk_frac]
        seq_sh += sq
    seq_sh /= max(len(perm_indices), 1)
    return np.arange(i, end, dtype=np.int32), seq_obs - seq_sh


def run_model_layer_windows(outputs_root, out_root, model, layer,
                            n_shuffles, n_jobs):
    layer_dir = outputs_root / model / f"layer_{layer}"
    print(f"\n  [{model}/layer_{layer}]")
    Z, uids, tok_lengths = load_layer(layer_dir)
    ref_seqs = load_ref_seqs(layer_dir)

    if model == "protgpt2":
        A_proj, res_offsets, res_lengths = build_protgpt2_projection(
            uids, ref_seqs, tok_lengths, "nferruz/ProtGPT2")
        token_ids, _, _ = build_bpe_token_map(
            uids, ref_seqs, tok_lengths, "nferruz/ProtGPT2")
    else:
        A_proj = None
        res_lengths = tok_lengths.astype(np.int32)
        res_offsets = {}; off = 0
        for uid, Lr in zip(uids, res_lengths):
            res_offsets[uid] = off; off += int(Lr)
        token_ids = None

    n_res = int(np.sum(res_lengths))
    perm_indices = build_protein_permutations(res_lengths, n_shuffles)

    n_features = int(Z.shape[1])
    chunk_size = 256
    n_chunks = (n_features + chunk_size - 1) // chunk_size

    mem_per_worker = n_res * chunk_size * 4 * 5 / 1e9
    mem_budget_gb = float(os.environ.get("CPU_STAGE_MEM_GB", 100.0))
    max_safe = max(1, int(mem_budget_gb / max(mem_per_worker, 0.1)))
    eff_jobs = min(n_jobs if n_jobs > 0 else cpu_count(), max_safe)

    per_cell_rows = []
    variants = [("raw", False)]
    if model == "protgpt2":
        variants.append(("inter", True))

    for w in WINDOWS:
        for vname, inter in variants:
            if inter:
                seq_adj = build_bpe_window_intertoken(
                    uids, res_lengths, res_offsets, token_ids, w)
            else:
                seq_adj = build_seq_neighbors_window(
                    uids, res_lengths, res_offsets, w)
            A_seq, deg_seq = adj_list_to_sparse(seq_adj, n_res)
            t0 = time.time()
            results = Parallel(n_jobs=eff_jobs, verbose=0)(
                delayed(_chunk)(
                    ci, chunk_size, Z, A_proj,
                    A_seq, deg_seq, perm_indices, n_features, TOPK_FRAC)
                for ci in range(n_chunks))
            all_idx = np.concatenate([r[0] for r in results])
            seq_delta = np.concatenate([r[1] for r in results])
            order = np.argsort(all_idx)
            df_cell = pd.DataFrame({
                "feature_idx": all_idx[order].astype(np.int32),
                "seq_delta":   seq_delta[order],
            })
            df_cell["model"] = model
            df_cell["layer"] = layer
            df_cell["window"] = w
            df_cell["variant"] = vname
            cell_dir = out_root / model / f"layer_{layer}"
            cell_dir.mkdir(parents=True, exist_ok=True)
            suffix = f"cell_w{w}" + ("_inter" if inter else "") + ".csv"
            df_cell.to_csv(cell_dir / suffix, index=False)
            print(f"    window ±{w} {vname:5s}:  edges {A_seq.nnz:,}  "
                  f"mean seq_delta = {seq_delta.mean():+.4f}  ({time.time()-t0:.0f}s)")
            per_cell_rows.append(df_cell)
    return pd.concat(per_cell_rows, ignore_index=True)


def cohens_d(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    pooled = np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2.0 + 1e-12)
    return float((a.mean() - b.mean()) / pooled)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-dir", default="outputs_layerwise")
    ap.add_argument("--out", default="results_seqwindow_sweep")
    ap.add_argument("--n-shuffles", type=int, default=5)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--only-model", default=None)
    args = ap.parse_args()

    outputs_root = (ROOT / args.outputs_dir) if not Path(args.outputs_dir).is_absolute() \
        else Path(args.outputs_dir)
    out_root = Path(args.out); out_root.mkdir(parents=True, exist_ok=True)

    models = [args.only_model] if args.only_model else list(LAYER_PLAN.keys())
    all_frames = []
    t0 = time.time()
    for model in models:
        for layer in LAYER_PLAN[model]:
            df = run_model_layer_windows(outputs_root, out_root, model, layer,
                                         args.n_shuffles, args.n_jobs)
            all_frames.append(df)
    big = pd.concat(all_frames, ignore_index=True)
    big.to_csv(out_root / "window_raw.csv", index=False)
    print(f"\nTotal sweep wall: {(time.time()-t0)/60:.1f} min")

    # Matched-pair H2 analyses per window
    MATCHED_9 = [
        (1,   0,   0,   0),
        (2,   4,   3,   4),     # (esm, rita, protgpt2)
        (3,   8,   6,   9),
        (4,   12,  9,  13),
        (5,   16, 12,  18),
        (6,   20, 15,  22),
        (7,   24, 18,  27),
        (8,   28, 21,  31),
        (9,   32, 23,  35),
    ]

    h2_rows = []
    for w in WINDOWS:
        for idx, eL, rL, pL in MATCHED_9:
            sub_e = big[(big.model == "esm2")    & (big.layer == eL) & (big.window == w) & (big.variant == "raw")]
            sub_r = big[(big.model == "rita")    & (big.layer == rL) & (big.window == w) & (big.variant == "raw")]
            sub_p_raw = big[(big.model == "protgpt2") & (big.layer == pL) & (big.window == w) & (big.variant == "raw")]
            sub_p_int = big[(big.model == "protgpt2") & (big.layer == pL) & (big.window == w) & (big.variant == "inter")]
            if len(sub_e) == 0: continue

            # ESM vs RITA (both residue-level)
            if len(sub_r):
                ev, rv = sub_e.seq_delta.values, sub_r.seq_delta.values
                d = cohens_d(rv, ev)
                _, p = stats.mannwhitneyu(rv, ev, alternative="greater")
                h2_rows.append(dict(hypothesis="H2 (RITA seq > ESM-2)",
                                    pair=idx, window=w,
                                    model_a="rita", layer_a=rL,
                                    model_b="esm2", layer_b=eL,
                                    variant="raw",
                                    cohens_d=d, MW_p=float(p),
                                    significant=bool(p < 0.05 and d > 0)))

            # ESM vs ProtGPT2 (raw = original BPE-conflated)
            if len(sub_p_raw):
                ev, pv = sub_e.seq_delta.values, sub_p_raw.seq_delta.values
                d = cohens_d(pv, ev)
                _, p = stats.mannwhitneyu(pv, ev, alternative="greater")
                h2_rows.append(dict(hypothesis="H2 raw (ProtGPT2 seq > ESM-2)",
                                    pair=idx, window=w,
                                    model_a="protgpt2", layer_a=pL,
                                    model_b="esm2", layer_b=eL,
                                    variant="raw",
                                    cohens_d=d, MW_p=float(p),
                                    significant=bool(p < 0.05 and d > 0)))

            # ESM vs ProtGPT2 (inter-token = H2')
            if len(sub_p_int):
                ev, pv = sub_e.seq_delta.values, sub_p_int.seq_delta.values
                d = cohens_d(pv, ev)
                _, p = stats.mannwhitneyu(pv, ev, alternative="greater")
                h2_rows.append(dict(hypothesis="H2' (ProtGPT2 inter-token seq > ESM-2)",
                                    pair=idx, window=w,
                                    model_a="protgpt2", layer_a=pL,
                                    model_b="esm2", layer_b=eL,
                                    variant="inter",
                                    cohens_d=d, MW_p=float(p),
                                    significant=bool(p < 0.05 and d > 0)))
    h2_df = pd.DataFrame(h2_rows)
    h2_df.to_csv(out_root / "h2_window_sweep.csv", index=False)

    # Summary
    lines = [
        "Sequential-window robustness sweep\n",
        "=" * 72 + "\n\n",
        f"Windows: ±{WINDOWS}   |   topk_frac={TOPK_FRAC}\n\n",
    ]
    for hyp in h2_df["hypothesis"].unique():
        sub = h2_df[h2_df.hypothesis == hyp]
        lines.append(f"\n--- {hyp} ---\n")
        cells = sub.groupby("window")["significant"].agg(["sum", "count"])
        lines.append(cells.to_string() + "\n")
        ns = int(sub.significant.sum()); n = len(sub)
        lines.append(f"Total: {ns}/{n} ({100*ns/n:.0f}%)\n")
    (out_root / "window_sweep_summary.txt").write_text("".join(lines))
    print(f"\nSummary → {out_root/'window_sweep_summary.txt'}")


if __name__ == "__main__":
    main()
