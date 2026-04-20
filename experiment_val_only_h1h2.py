#!/usr/bin/env python3
"""
experiment_val_only_h1h2.py — Recompute H1/H2 on the 150-protein val subset.

Addresses the §2.1 Methods promise: H1 (ESM-2 struct > ProtGPT2 struct) and
H2 (ProtGPT2 seq > ESM-2 seq) should also hold on the held-out val proteins.

Reuses cached Z.npy per layer — no SAE training. Per layer:
  • Parse META.val_uids (150 per layer, same protein set across all models)
  • Filter Z rows to val-protein residues/tokens
  • Build neighbor graphs on val uids only
  • Compute per-feature struct/seq Cohen's d + shuffle null (n_shuffles=5)
  • Save struct_seq_metrics_val.csv alongside the existing CSV

Then for each of the 5 matched depths, run H1 and H2 via Mann-Whitney (matching
analyze_hypotheses.py methodology) on both the all-protein and val-only slices,
and emit a comparison table.

Usage:
  python experiment_val_only_h1h2.py --out analysis_results_valonly [--n-shuffles 5]
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
    build_neighbor_graphs_residue_parallel,
    adj_list_to_sparse, build_protein_permutations,
    _process_struct_seq_chunk_v3,
    build_protgpt2_projection,
    DEFAULT_CONTACT_CUTOFF, DEFAULT_SEQ_GAP_MIN, DEFAULT_TOPK_FRAC,
)

ROOT = Path(__file__).resolve().parent
OUT_LW = ROOT / "outputs_layerwise"
PDB_DIR = ROOT / "cache" / "pdb_files"

LAYERS = {
    "esm2":       [0, 8, 16, 24, 32],
    "protgpt2":   [0, 9, 18, 27, 35],
    "prott5_enc": [0, 6, 12, 18, 23],
    "prott5_dec": [0, 6, 12, 18, 23],
}

# (depth_label, esm2_layer, protgpt2_layer) — H1/H2 matched pairs
MATCHED_PAIRS = [
    ("0%",    0,  0),
    ("25%",   8,  9),
    ("50%",  16, 18),
    ("75%",  24, 27),
    ("100%", 32, 35),
]


# -------------------------------------------------------------------
#  PER-LAYER: val-only struct/seq locality
# -------------------------------------------------------------------

def load_layer_slim(model: str, layer: int):
    """Load Z + META + uids + token lengths + sequences for one layer."""
    d = OUT_LW / model / f"layer_{layer}"
    Z = np.load(d / "Z.npy", mmap_mode="r")
    uids = json.loads((d / "uids.json").read_text())
    lengths_tok = np.load(d / "lengths.npy").astype(np.int64)
    seqs = json.loads((d / "sequences.json").read_text())
    if isinstance(seqs, list):
        seqs = dict(zip(uids, seqs))
    meta = json.loads((d / "META.json").read_text())
    return d, Z, uids, lengths_tok, seqs, meta


def filter_to_val(Z, uids, lengths_tok, seqs, val_uids):
    """Return (Z_val, val_uids_sorted, val_tok_lengths, val_res_lengths, val_seqs).

    val_uids_sorted: subset of uids, preserving uids.json order.
    val_tok_lengths/val_res_lengths: per-protein token/residue counts.
    """
    keep = set(val_uids)
    val_uids_sorted = [u for u in uids if u in keep]
    uid_to_idx = {u: i for i, u in enumerate(uids)}
    val_idx = [uid_to_idx[u] for u in val_uids_sorted]

    tok_offsets = np.concatenate([[0], np.cumsum(lengths_tok)]).astype(np.int64)
    val_rows = np.concatenate([
        np.arange(tok_offsets[i], tok_offsets[i + 1], dtype=np.int64)
        for i in val_idx
    ])
    # Materialise val slice (it is small — ~30k residues × features)
    Z_val = np.asarray(Z[val_rows], dtype=np.float16)  # match Z.npy dtype
    val_tok_lengths = lengths_tok[val_idx]
    val_res_lengths = np.array([len(seqs[u]) for u in val_uids_sorted], dtype=np.int64)
    val_seqs = {u: seqs[u] for u in val_uids_sorted}
    return Z_val, val_uids_sorted, val_tok_lengths, val_res_lengths, val_seqs


def locality_val(model: str, layer: int, n_shuffles: int, n_jobs: int):
    """Return a DataFrame with per-feature struct/seq deltas on val proteins."""
    print(f"\n  [{model}/layer_{layer}] val-only locality")
    d, Z, uids, lengths_tok, seqs, meta = load_layer_slim(model, layer)
    val_uids = meta["val_uids"]
    print(f"    val_uids: {len(val_uids)} proteins")

    Z_val, val_uids_sorted, val_tok_len, val_res_len, val_seqs = filter_to_val(
        Z, uids, lengths_tok, seqs, val_uids)
    n_res_val = int(val_res_len.sum())
    print(f"    val residues: {n_res_val}, val tokens: {int(val_tok_len.sum())}")

    # For ProtGPT2, rebuild token→residue projection on val subset
    if model == "protgpt2":
        A_proj, _, _ = build_protgpt2_projection(
            val_uids_sorted, val_seqs, val_tok_len, "nferruz/ProtGPT2")
    else:
        A_proj = None
        if int(val_tok_len.sum()) != n_res_val:
            raise RuntimeError(
                f"{model}: val tokens != val residues, expected 1:1")

    # Neighbor graphs on val uids
    seq_adj, struct_adj = build_neighbor_graphs_residue_parallel(
        val_uids_sorted, val_res_len, val_seqs, PDB_DIR,
        n_jobs=n_jobs,
        contact_cutoff=DEFAULT_CONTACT_CUTOFF,
        seq_gap_min=DEFAULT_SEQ_GAP_MIN)
    A_seq, deg_seq = adj_list_to_sparse(seq_adj, n_res_val)
    A_struct, deg_struct = adj_list_to_sparse(struct_adj, n_res_val)
    del seq_adj, struct_adj

    print(f"    seq edges: {A_seq.nnz:,}, struct edges: {A_struct.nnz:,}")

    # Permutations
    perm_indices = build_protein_permutations(val_res_len, n_shuffles)

    # Process chunks in parallel
    n_features = int(Z_val.shape[1])
    chunk_size = 256
    n_chunks = (n_features + chunk_size - 1) // chunk_size

    mem_per_worker = n_res_val * chunk_size * 4 * 5 / 1e9
    mem_budget_gb = float(os.environ.get("CPU_STAGE_MEM_GB", 100.0))
    max_safe_jobs = max(1, int(mem_budget_gb / max(mem_per_worker, 0.1)))
    eff_jobs = min(n_jobs if n_jobs > 0 else cpu_count(), max_safe_jobs)

    print(f"    {n_features} features in {n_chunks} chunks, {eff_jobs} workers")
    t0 = time.time()
    results = Parallel(n_jobs=eff_jobs, verbose=1)(
        delayed(_process_struct_seq_chunk_v3)(
            ci, chunk_size, Z_val, A_proj,
            A_seq, deg_seq, A_struct, deg_struct,
            perm_indices, n_features, DEFAULT_TOPK_FRAC)
        for ci in range(n_chunks))
    elapsed = time.time() - t0

    all_idx = np.concatenate([r[0] for r in results])
    all_seq_obs = np.concatenate([r[1] for r in results])
    all_str_obs = np.concatenate([r[2] for r in results])
    all_seq_sh = np.concatenate([r[3] for r in results])
    all_str_sh = np.concatenate([r[4] for r in results])
    order = np.argsort(all_idx)

    df = pd.DataFrame({
        "feature_idx": all_idx[order].astype(np.int32),
        "seq_effect_obs":    all_seq_obs[order],
        "struct_effect_obs": all_str_obs[order],
        "seq_effect_shuffle":    all_seq_sh[order],
        "struct_effect_shuffle": all_str_sh[order],
        "seq_delta":    (all_seq_obs - all_seq_sh)[order],
        "struct_delta": (all_str_obs - all_str_sh)[order],
    })
    out_path = d / "struct_seq_metrics_val.csv"
    df.to_csv(out_path, index=False)
    print(f"    saved → {out_path}  ({elapsed:.0f}s)")
    return df


# -------------------------------------------------------------------
#  HYPOTHESIS TESTS ON val vs all
# -------------------------------------------------------------------

def cohens_d(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    pooled = np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2.0 + 1e-12)
    return float((a.mean() - b.mean()) / pooled)


def mw_greater(a, b):
    u, p = stats.mannwhitneyu(a, b, alternative="greater")
    return float(u), float(p)


def run_hypothesis(df_a, df_b, metric, name_a, name_b):
    """Mann-Whitney one-tailed (A > B) + pooled Cohen's d."""
    a = df_a[metric].values
    b = df_b[metric].values
    d = cohens_d(a, b)
    u, p = mw_greater(a, b)
    return {
        "metric": metric,
        "A": name_a, "B": name_b,
        "A_mean": float(a.mean()), "A_median": float(np.median(a)),
        "B_mean": float(b.mean()), "B_median": float(np.median(b)),
        "cohens_d": d, "MW_U": u, "MW_p": p,
        "significant": bool(p < 0.05 and d > 0),
    }


def consolidate(out_dir: Path):
    rows = []
    for depth_label, esm_layer, pg2_layer in MATCHED_PAIRS:
        # All-protein deltas from main run
        esm_all = pd.read_csv(OUT_LW / "esm2" / f"layer_{esm_layer}" /
                              "struct_seq_metrics.csv")
        pg2_all = pd.read_csv(OUT_LW / "protgpt2" / f"layer_{pg2_layer}" /
                              "struct_seq_metrics.csv")
        # Val-only deltas from this run
        esm_val = pd.read_csv(OUT_LW / "esm2" / f"layer_{esm_layer}" /
                              "struct_seq_metrics_val.csv")
        pg2_val = pd.read_csv(OUT_LW / "protgpt2" / f"layer_{pg2_layer}" /
                              "struct_seq_metrics_val.csv")

        # H1: ESM-2 struct > ProtGPT2 struct
        for split, (e, p) in [("all", (esm_all, pg2_all)), ("val", (esm_val, pg2_val))]:
            r = run_hypothesis(e, p, "struct_delta", "esm2", "protgpt2")
            r.update(dict(depth=depth_label, hypothesis="H1", split=split,
                          esm2_layer=esm_layer, protgpt2_layer=pg2_layer))
            rows.append(r)
        # H2: ProtGPT2 seq > ESM-2 seq
        for split, (e, p) in [("all", (esm_all, pg2_all)), ("val", (esm_val, pg2_val))]:
            r = run_hypothesis(p, e, "seq_delta", "protgpt2", "esm2")
            r.update(dict(depth=depth_label, hypothesis="H2", split=split,
                          esm2_layer=esm_layer, protgpt2_layer=pg2_layer))
            rows.append(r)

    df = pd.DataFrame(rows)
    out_csv = out_dir / "val_only_h1h2.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n  Consolidated → {out_csv}")

    # Report
    txt = out_dir / "val_only_h1h2.txt"
    lines = ["Val-only H1/H2 check (all 1,500 vs held-out 150 proteins)\n",
             "=" * 72 + "\n\n",
             "H1 = ESM-2 structural Δ > ProtGPT2 structural Δ (MW one-tailed).\n",
             "H2 = ProtGPT2 sequential Δ > ESM-2 sequential Δ.\n\n"]
    for hyp in ["H1", "H2"]:
        lines.append(f"\n--- {hyp} ---\n")
        sub = df[df["hypothesis"] == hyp]
        compact = sub.pivot_table(index="depth", columns="split",
                                  values=["cohens_d", "MW_p", "significant"],
                                  aggfunc="first")
        lines.append(compact.to_string() + "\n")
        n_sup_all = int((sub[sub["split"] == "all"]["significant"]).sum())
        n_sup_val = int((sub[sub["split"] == "val"]["significant"]).sum())
        lines.append(f"\n  {hyp}: supported at {n_sup_all}/5 depths (all) "
                     f"and {n_sup_val}/5 depths (val)\n")
    with open(txt, "w") as f:
        f.writelines(lines)
    print(f"  Report       → {txt}")


# -------------------------------------------------------------------
#  MAIN
# -------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="analysis_results_valonly")
    ap.add_argument("--n-shuffles", type=int, default=5)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--models", nargs="+",
                    default=["esm2", "protgpt2"],
                    help="Only esm2+protgpt2 needed for H1/H2; add others for completeness")
    ap.add_argument("--consolidate-only", action="store_true",
                    help="Skip per-layer recomputation; only aggregate existing val CSVs")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.consolidate_only:
        t0 = time.time()
        for model in args.models:
            for layer in LAYERS[model]:
                val_csv = OUT_LW / model / f"layer_{layer}" / "struct_seq_metrics_val.csv"
                if val_csv.exists():
                    print(f"  [skip] {val_csv} already present")
                    continue
                locality_val(model, layer, args.n_shuffles, args.n_jobs)
        print(f"\nAll {sum(len(LAYERS[m]) for m in args.models)} layers done "
              f"in {(time.time()-t0)/60:.1f} min")

    consolidate(out_dir)


if __name__ == "__main__":
    main()
