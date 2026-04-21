#!/usr/bin/env python3
"""
experiment_metric_sweep_rita.py — ESM-2 vs RITA metric-hyperparameter sweep.

Same 9-cell grid (seq_gap_min × topk_frac) as experiment_metric_sweep.py but
on the residue-level AR model RITA. Mirror of experiment_metric_sweep_progen2.
Both models are 1 token/residue so no inter-token correction is applied.

Output:
  results_metric_sweep_rita/
    {esm2,rita}/layer_{N}/cell_sg{s}_tk{t}.csv
    sweep_raw.csv, h1_sweep.csv, h2_sweep.csv, sweep_summary.txt
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
    build_neighbor_graphs_residue_parallel,
    adj_list_to_sparse, build_protein_permutations,
)
from experiment_metric_sweep import (
    _sweep_chunk, SEQ_GAP_MINS, TOPK_FRACS, cohens_d,
)

ROOT = Path(__file__).resolve().parent

LAYERS = {
    "esm2": [0, 8, 16, 24, 32],
    "rita": [0, 6, 12, 18, 23],
}

MATCHED_PAIRS = [
    ("0%",    0,  0),
    ("25%",   8,  6),
    ("50%",  16, 12),
    ("75%",  24, 18),
    ("100%", 32, 23),
]


def run_model_layer(outputs_root: Path, out_root: Path, model: str, layer: int,
                    n_shuffles: int, n_jobs: int, pdb_dir: Path):
    layer_dir = outputs_root / model / f"layer_{layer}"
    print(f"\n  [{model}/layer_{layer}]")
    Z, uids, lengths = load_layer(layer_dir)
    ref_seqs = load_ref_seqs(layer_dir)

    res_lengths = lengths.astype(np.int32)
    res_offsets = {}
    off = 0
    for uid, Lr in zip(uids, res_lengths):
        res_offsets[uid] = off; off += int(Lr)
    n_res = int(np.sum(res_lengths))

    seq_adj = [[] for _ in range(n_res)]
    for uid, Lr in zip(uids, res_lengths):
        base = res_offsets[uid]
        Lr = int(Lr)
        for r in range(Lr):
            for d in (-2, -1, 1, 2):
                rr = r + d
                if 0 <= rr < Lr:
                    seq_adj[base + r].append(base + rr)
    A_seq, deg_seq = adj_list_to_sparse(seq_adj, n_res)
    print(f"    n_res={n_res:,}, seq edges: {A_seq.nnz:,}")

    perm_indices = build_protein_permutations(res_lengths, n_shuffles)

    n_features = int(Z.shape[1])
    chunk_size = 256
    n_chunks = (n_features + chunk_size - 1) // chunk_size

    mem_per_worker = n_res * chunk_size * 4 * 5 / 1e9
    mem_budget_gb = float(os.environ.get("CPU_STAGE_MEM_GB", 100.0))
    max_safe = max(1, int(mem_budget_gb / max(mem_per_worker, 0.1)))
    eff_jobs = min(n_jobs if n_jobs > 0 else cpu_count(), max_safe)

    per_cell_rows = []
    for sg in SEQ_GAP_MINS:
        print(f"    building struct adj seq_gap_min={sg}...")
        _, struct_adj = build_neighbor_graphs_residue_parallel(
            uids, res_lengths, ref_seqs, pdb_dir,
            n_jobs=eff_jobs, contact_cutoff=8.0, seq_gap_min=sg)
        A_struct, deg_struct = adj_list_to_sparse(struct_adj, n_res)
        print(f"    seq_gap_min={sg}: struct edges {A_struct.nnz:,} — "
              f"sweep topk_fracs {TOPK_FRACS}...")
        t0 = time.time()
        results = Parallel(n_jobs=eff_jobs, verbose=1)(
            delayed(_sweep_chunk)(
                ci, chunk_size, Z, None,
                A_struct, deg_struct, A_seq, deg_seq,
                perm_indices, n_features, TOPK_FRACS)
            for ci in range(n_chunks))
        print(f"      done in {time.time()-t0:.0f}s")

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
            per_cell_rows.append(df_cell)
            cell_dir = out_root / model / f"layer_{layer}"
            cell_dir.mkdir(parents=True, exist_ok=True)
            df_cell.to_csv(
                cell_dir / f"cell_sg{sg}_tk{int(tf*100):02d}.csv", index=False)

    return pd.concat(per_cell_rows, ignore_index=True)


def aggregate(out_root: Path, sweep_df: pd.DataFrame):
    h1_rows, h2_rows = [], []
    for depth_label, esm_l, r_l in MATCHED_PAIRS:
        for sg in SEQ_GAP_MINS:
            for tf in TOPK_FRACS:
                esm = sweep_df[(sweep_df.model == "esm2") & (sweep_df.layer == esm_l)
                               & (sweep_df.seq_gap_min == sg) & (sweep_df.topk_frac == tf)]
                r = sweep_df[(sweep_df.model == "rita") & (sweep_df.layer == r_l)
                             & (sweep_df.seq_gap_min == sg) & (sweep_df.topk_frac == tf)]
                if len(esm) == 0 or len(r) == 0:
                    continue
                a, b = esm.struct_delta.values, r.struct_delta.values
                d  = cohens_d(a, b)
                _, p = stats.mannwhitneyu(a, b, alternative="greater")
                h1_rows.append(dict(depth=depth_label, seq_gap_min=sg, topk_frac=tf,
                                    esm_layer=esm_l, rita_layer=r_l,
                                    cohens_d=d, MW_p=float(p),
                                    esm_mean=float(a.mean()), rita_mean=float(b.mean()),
                                    significant=bool(p < 0.05 and d > 0)))
                a2, b2 = r.seq_delta.values, esm.seq_delta.values
                d2 = cohens_d(a2, b2)
                _, p2 = stats.mannwhitneyu(a2, b2, alternative="greater")
                h2_rows.append(dict(depth=depth_label, seq_gap_min=sg, topk_frac=tf,
                                    esm_layer=esm_l, rita_layer=r_l,
                                    cohens_d=d2, MW_p=float(p2),
                                    rita_mean=float(a2.mean()), esm_mean=float(b2.mean()),
                                    significant=bool(p2 < 0.05 and d2 > 0)))

    h1 = pd.DataFrame(h1_rows); h2 = pd.DataFrame(h2_rows)
    h1.to_csv(out_root / "h1_sweep.csv", index=False)
    h2.to_csv(out_root / "h2_sweep.csv", index=False)

    lines = ["ESM-2 vs RITA metric sweep — H1 + H2 (residue-level, no BPE correction)\n",
             "=" * 72 + "\n\n",
             f"Grid: seq_gap_min ∈ {SEQ_GAP_MINS} × topk_frac ∈ {TOPK_FRACS}\n\n"]
    for name, df in [("H1 (ESM-2 struct > RITA struct)", h1),
                     ("H2 (RITA seq > ESM-2 seq)",       h2)]:
        lines.append(f"\n--- {name} ---\n")
        cells = df.groupby(["seq_gap_min", "topk_frac"])["significant"].agg(["sum", "count"])
        lines.append(cells.to_string() + "\n")
        ns = int(df.significant.sum()); n = len(df)
        lines.append(f"\nTotal: {ns} / {n} depth-cells significant ({100*ns/n:.0f}%)\n")
    with open(out_root / "sweep_summary.txt", "w") as f:
        f.writelines(lines)
    print(f"\nSummary → {out_root/'sweep_summary.txt'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-dir", default="outputs_layerwise")
    ap.add_argument("--out", default="results_metric_sweep_rita")
    ap.add_argument("--n-shuffles", type=int, default=5)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--pdb-dir", default="cache/pdb_files")
    args = ap.parse_args()

    outputs_root = (ROOT / args.outputs_dir) if not Path(args.outputs_dir).is_absolute() else Path(args.outputs_dir)
    out_root = Path(args.out); out_root.mkdir(parents=True, exist_ok=True)
    pdb_dir = Path(args.pdb_dir)

    print(f"Source outputs: {outputs_root}\nResults dir:    {out_root}\n")

    all_frames = []
    t0 = time.time()
    for model in ("esm2", "rita"):
        for layer in LAYERS[model]:
            df = run_model_layer(outputs_root, out_root, model, layer,
                                 args.n_shuffles, args.n_jobs, pdb_dir)
            all_frames.append(df)
    big = pd.concat(all_frames, ignore_index=True)
    big.to_csv(out_root / "sweep_raw.csv", index=False)
    print(f"\nTotal sweep wall: {(time.time()-t0)/60:.1f} min")

    aggregate(out_root, big)


if __name__ == "__main__":
    main()
