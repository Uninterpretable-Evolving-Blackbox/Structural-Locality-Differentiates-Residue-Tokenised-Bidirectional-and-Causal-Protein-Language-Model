#!/usr/bin/env python3
"""
experiment_cutoff_sweep.py — Cα contact-cutoff robustness sweep.

For each model in {esm2, rita, prott5_enc, prott5_dec} and each depth
in the 9-point densified grid, rebuilds the structural neighbour adjacency
at Cα ∈ {6, 8, 10} Å (seq_gap_min kept at 12, the paper default) and
recomputes per-feature struct_delta.

Outputs:
  results_cutoff_sweep/
    {model}/layer_{N}/cell_ca{cutoff}.csv      per-feature struct_delta per cell
    cutoff_raw.csv                              concat of all per-cell deltas
    h1_cutoff_sweep.csv    H1 (ESM-2 vs RITA) Cohen's d per (Cα, depth) pair
    h3_cutoff_sweep.csv    H3 (ProtT5 enc vs dec) Cohen's d per (Cα, depth)
    h5_cutoff_means.csv    per-layer mean struct_delta × (model, Cα) for H5-style trend
    cutoff_sweep_summary.txt
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
from experiment_metric_sweep import _cohens_d_multi_topk
from experiment_bpe_correction import build_original_seq_neighbors

ROOT = Path(__file__).resolve().parent

LAYER_PLAN = {
    "esm2":       [0, 4, 8, 12, 16, 20, 24, 28, 32],
    "rita":       [0, 3, 6, 9, 12, 15, 18, 21, 23],
    "prott5_enc": [0, 3, 6, 9, 12, 15, 18, 21, 23],
    "prott5_dec": [0, 3, 6, 9, 12, 15, 18, 21, 23],
}
CUTOFFS      = [6.0, 8.0, 10.0]
SEQ_GAP_MIN  = 12          # paper default
TOPK_FRAC    = 0.10        # paper default
N_BLOCKS = {"esm2": 33, "rita": 24, "prott5_enc": 24, "prott5_dec": 24}

H1_PAIRS = [("esm2", "rita", LAYER_PLAN["esm2"], LAYER_PLAN["rita"])]
H3_PAIRS = [("prott5_enc", "prott5_dec", LAYER_PLAN["prott5_enc"], LAYER_PLAN["prott5_dec"])]


def _chunk(chunk_idx, chunk_size, Z, A_struct, deg_struct, A_seq, deg_seq,
           perm_indices, n_features, topk_frac):
    """Observed - shuffled struct_delta per feature, single cell."""
    i = chunk_idx * chunk_size
    end = min(i + chunk_size, n_features)
    acts = np.asarray(Z[:, i:end], dtype=np.float32)
    gstds = np.std(acts, axis=0).astype(np.float32)
    str_obs = _cohens_d_multi_topk(acts, A_struct, deg_struct, gstds, [topk_frac])[topk_frac]
    str_sh = np.zeros(end - i, dtype=np.float32)
    for perm in perm_indices:
        sr = _cohens_d_multi_topk(acts[perm], A_struct, deg_struct, gstds, [topk_frac])[topk_frac]
        str_sh += sr
    str_sh /= max(len(perm_indices), 1)
    return np.arange(i, end, dtype=np.int32), str_obs - str_sh


def run_model_layer_cutoffs(outputs_root, out_root, model, layer,
                            n_shuffles, n_jobs, pdb_dir, cutoffs):
    layer_dir = outputs_root / model / f"layer_{layer}"
    print(f"\n  [{model}/layer_{layer}]")
    Z, uids, lengths = load_layer(layer_dir)
    ref_seqs = load_ref_seqs(layer_dir)
    res_lengths = lengths.astype(np.int32)
    res_offsets = {}; off = 0
    for uid, Lr in zip(uids, res_lengths):
        res_offsets[uid] = off; off += int(Lr)
    n_res = int(np.sum(res_lengths))

    # Sequential adj is window=2 (paper default), shared across Cα cells
    seq_adj = build_original_seq_neighbors(uids, res_lengths, res_offsets)
    A_seq, deg_seq = adj_list_to_sparse(seq_adj, n_res)

    perm_indices = build_protein_permutations(res_lengths, n_shuffles)

    n_features = int(Z.shape[1])
    chunk_size = 256
    n_chunks = (n_features + chunk_size - 1) // chunk_size

    mem_per_worker = n_res * chunk_size * 4 * 5 / 1e9
    mem_budget_gb = float(os.environ.get("CPU_STAGE_MEM_GB", 100.0))
    max_safe = max(1, int(mem_budget_gb / max(mem_per_worker, 0.1)))
    eff_jobs = min(n_jobs if n_jobs > 0 else cpu_count(), max_safe)

    per_cell_rows = []
    for ca in cutoffs:
        print(f"    Cα={ca}Å — building struct adj…")
        t0 = time.time()
        _, struct_adj = build_neighbor_graphs_residue_parallel(
            uids, res_lengths, ref_seqs, pdb_dir,
            n_jobs=eff_jobs, contact_cutoff=ca, seq_gap_min=SEQ_GAP_MIN)
        A_struct, deg_struct = adj_list_to_sparse(struct_adj, n_res)
        print(f"      struct edges {A_struct.nnz:,}  ({time.time()-t0:.0f}s)")

        t0 = time.time()
        results = Parallel(n_jobs=eff_jobs, verbose=0)(
            delayed(_chunk)(
                ci, chunk_size, Z, A_struct, deg_struct,
                A_seq, deg_seq, perm_indices, n_features, TOPK_FRAC)
            for ci in range(n_chunks))
        all_idx = np.concatenate([r[0] for r in results])
        str_delta = np.concatenate([r[1] for r in results])
        order = np.argsort(all_idx)
        df_cell = pd.DataFrame({
            "feature_idx":  all_idx[order].astype(np.int32),
            "struct_delta": str_delta[order],
        })
        df_cell["model"] = model
        df_cell["layer"] = layer
        df_cell["contact_cutoff"] = ca
        cell_dir = out_root / model / f"layer_{layer}"
        cell_dir.mkdir(parents=True, exist_ok=True)
        df_cell.to_csv(cell_dir / f"cell_ca{int(ca)}.csv", index=False)
        print(f"      locality {time.time()-t0:.0f}s — mean struct_delta = {str_delta.mean():+.5f}")
        per_cell_rows.append(df_cell)
    return pd.concat(per_cell_rows, ignore_index=True)


def cohens_d(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    pooled = np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2.0 + 1e-12)
    return float((a.mean() - b.mean()) / pooled)


def aggregate_pair_d(sweep_df, model_a, model_b, layers_a, layers_b, label_a_gt_b):
    """For each Cα and each matched (layer_a, layer_b), compute Cohen's d and MW p."""
    rows = []
    for la, lb in zip(layers_a, layers_b):
        for ca in CUTOFFS:
            a = sweep_df[(sweep_df.model == model_a) & (sweep_df.layer == la)
                         & (sweep_df.contact_cutoff == ca)]
            b = sweep_df[(sweep_df.model == model_b) & (sweep_df.layer == lb)
                         & (sweep_df.contact_cutoff == ca)]
            if len(a) == 0 or len(b) == 0:
                continue
            av, bv = a.struct_delta.values, b.struct_delta.values
            d = cohens_d(av, bv)
            u, p = stats.mannwhitneyu(av, bv, alternative="greater")
            rows.append(dict(
                model_a=model_a, layer_a=la,
                model_b=model_b, layer_b=lb,
                rel_depth_a=la / (N_BLOCKS[model_a] - 1),
                rel_depth_b=lb / (N_BLOCKS[model_b] - 1),
                contact_cutoff=ca,
                mean_a=float(av.mean()), mean_b=float(bv.mean()),
                cohens_d=d, MW_p=float(p),
                significant=bool(p < 0.05 and d > 0),
                hypothesis=label_a_gt_b,
            ))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-dir", default="outputs_layerwise")
    ap.add_argument("--out", default="results_cutoff_sweep")
    ap.add_argument("--n-shuffles", type=int, default=5)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--pdb-dir", default="cache/pdb_files")
    ap.add_argument("--cutoffs", default="6,8,10",
                    help="comma-separated list of Cα cutoffs in Å")
    ap.add_argument("--only-model", default=None, help="limit sweep to one model")
    args = ap.parse_args()

    outputs_root = (ROOT / args.outputs_dir) if not Path(args.outputs_dir).is_absolute() \
        else Path(args.outputs_dir)
    out_root = Path(args.out); out_root.mkdir(parents=True, exist_ok=True)
    pdb_dir = Path(args.pdb_dir)
    cutoffs = [float(x) for x in args.cutoffs.split(",")]

    models = [args.only_model] if args.only_model else list(LAYER_PLAN.keys())
    all_frames = []
    t0 = time.time()
    for model in models:
        for layer in LAYER_PLAN[model]:
            df = run_model_layer_cutoffs(outputs_root, out_root, model, layer,
                                         args.n_shuffles, args.n_jobs, pdb_dir,
                                         cutoffs)
            all_frames.append(df)
    big = pd.concat(all_frames, ignore_index=True)
    big.to_csv(out_root / "cutoff_raw.csv", index=False)
    print(f"\nTotal sweep wall: {(time.time()-t0)/60:.1f} min")

    # H1 (ESM-2 > RITA) per (Cα, depth pair)
    h1 = aggregate_pair_d(big, "esm2", "rita",
                          LAYER_PLAN["esm2"], LAYER_PLAN["rita"],
                          "H1 (ESM-2 struct > RITA struct)")
    h1.to_csv(out_root / "h1_cutoff_sweep.csv", index=False)

    # H3 (enc > dec) per (Cα, depth pair)
    h3 = aggregate_pair_d(big, "prott5_enc", "prott5_dec",
                          LAYER_PLAN["prott5_enc"], LAYER_PLAN["prott5_dec"],
                          "H3 (ProtT5 enc struct > dec struct)")
    h3.to_csv(out_root / "h3_cutoff_sweep.csv", index=False)

    # H5-style per-(model, Cα, layer) mean struct_delta
    h5 = (big.groupby(["model", "layer", "contact_cutoff"])
             .struct_delta.agg(["mean", "median", "std", "count"])
             .reset_index())
    h5.to_csv(out_root / "h5_cutoff_means.csv", index=False)

    # Summary
    lines = [
        "Cα contact-cutoff robustness sweep\n",
        "=" * 72 + "\n\n",
        f"Cutoffs: {cutoffs} Å   |   seq_gap_min={SEQ_GAP_MIN} (fixed)   |   topk_frac={TOPK_FRAC}\n\n",
    ]
    for name, df in [("H1 (ESM-2 > RITA)", h1), ("H3 (ProtT5 enc > dec)", h3)]:
        lines.append(f"\n--- {name} ---\n")
        cells = df.groupby("contact_cutoff")["significant"].agg(["sum", "count"])
        lines.append(cells.to_string() + "\n")
        ns = int(df.significant.sum()); n = len(df)
        lines.append(f"Total: {ns}/{n} cells significant ({100*ns/n:.0f}%)\n")
    (out_root / "cutoff_sweep_summary.txt").write_text("".join(lines))
    print(f"\nSummary → {out_root/'cutoff_sweep_summary.txt'}")


if __name__ == "__main__":
    main()
