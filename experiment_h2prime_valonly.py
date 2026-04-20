#!/usr/bin/env python3
"""
experiment_h2prime_valonly.py — H2' (inter-token) restricted to val 150 proteins.

Recomputes ProtGPT2 sequential Δ on val-only residues using the BPE inter-token
adjacency, then runs H2' = Mann-Whitney one-tailed(ESM-2 val seq-Δ > ProtGPT2
val inter-token seq-Δ) at all 5 matched depths.

ESM-2 val seq-Δ already exists in struct_seq_metrics_val.csv (from the earlier
experiment_val_only_h1h2.py run, which used the standard residue adjacency —
that's exactly what we want for ESM-2, since it has no BPE).

Usage:
  python experiment_h2prime_valonly.py --out analysis_results_h2prime_valonly
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from joblib import Parallel, delayed, cpu_count

from cpu_stage import (
    load_layer, load_ref_seqs,
    adj_list_to_sparse, build_protein_permutations,
    _process_struct_seq_chunk_v3,
    build_protgpt2_projection,
)
from experiment_bpe_correction import (
    build_bpe_token_map, build_bpe_corrected_seq_neighbors,
)

ROOT = Path(__file__).resolve().parent
OUT_LW = ROOT / "outputs_layerwise"

MATCHED_PAIRS = [
    ("0%",    0,  0),
    ("25%",   8,  9),
    ("50%",  16, 18),
    ("75%",  24, 27),
    ("100%", 32, 35),
]


def run_protgpt2_layer(layer: int, n_shuffles: int, n_jobs: int):
    """Compute per-feature inter-token seq Δ on val-only residues for one ProtGPT2 layer."""
    d = OUT_LW / "protgpt2" / f"layer_{layer}"
    out_csv = d / "struct_seq_metrics_val_intertoken.csv"
    if out_csv.exists():
        print(f"    [skip] {out_csv} already exists")
        return pd.read_csv(out_csv)

    print(f"\n  [protgpt2/layer_{layer}] val-only inter-token seq Δ")
    Z = np.load(d / "Z.npy", mmap_mode="r")
    uids = json.loads((d / "uids.json").read_text())
    tok_lengths = np.load(d / "lengths.npy").astype(np.int64)
    seqs = json.loads((d / "sequences.json").read_text())
    if isinstance(seqs, list):
        seqs = dict(zip(uids, seqs))
    meta = json.loads((d / "META.json").read_text())
    val_uids = meta["val_uids"]

    # Val subsetting
    uid_to_idx = {u: i for i, u in enumerate(uids)}
    val_uids_sorted = [u for u in uids if u in set(val_uids)]
    val_idx = [uid_to_idx[u] for u in val_uids_sorted]
    tok_offsets = np.concatenate([[0], np.cumsum(tok_lengths)]).astype(np.int64)
    val_rows = np.concatenate([
        np.arange(tok_offsets[i], tok_offsets[i + 1], dtype=np.int64)
        for i in val_idx
    ])
    Z_val = np.asarray(Z[val_rows], dtype=np.float16)
    val_tok_len = tok_lengths[val_idx]
    val_seqs = {u: seqs[u] for u in val_uids_sorted}
    print(f"    val tokens: {int(val_tok_len.sum())}, "
          f"val residues: {sum(len(s) for s in val_seqs.values())}")

    # Projection + token map on val subset
    A_proj, res_offsets, val_res_len = build_protgpt2_projection(
        val_uids_sorted, val_seqs, val_tok_len, "nferruz/ProtGPT2")
    token_ids, _, _ = build_bpe_token_map(
        val_uids_sorted, val_seqs, val_tok_len, "nferruz/ProtGPT2")
    seq_adj = build_bpe_corrected_seq_neighbors(
        val_uids_sorted, val_res_len, res_offsets, token_ids)

    n_res = int(val_res_len.sum())
    from scipy import sparse
    A_seq, deg_seq = adj_list_to_sparse(seq_adj, n_res)
    A_struct = sparse.csr_matrix((n_res, n_res), dtype=np.float32)
    deg_struct = np.zeros(n_res, dtype=np.float32)
    print(f"    inter-token seq edges: {A_seq.nnz:,}")

    perm_indices = build_protein_permutations(val_res_len, n_shuffles)

    # Parallel chunked Cohen's d via cpu_stage's helper
    n_features = int(Z_val.shape[1])
    chunk_size = 256
    n_chunks = (n_features + chunk_size - 1) // chunk_size

    mem_per_worker = n_res * chunk_size * 4 * 5 / 1e9
    mem_budget_gb = float(os.environ.get("CPU_STAGE_MEM_GB", 100.0))
    max_safe = max(1, int(mem_budget_gb / max(mem_per_worker, 0.1)))
    eff_jobs = min(n_jobs if n_jobs > 0 else cpu_count(), max_safe)

    t0 = time.time()
    results = Parallel(n_jobs=eff_jobs, verbose=1)(
        delayed(_process_struct_seq_chunk_v3)(
            ci, chunk_size, Z_val, A_proj,
            A_seq, deg_seq, A_struct, deg_struct,
            perm_indices, n_features, 0.10)
        for ci in range(n_chunks))
    print(f"    done in {time.time()-t0:.0f}s")

    all_idx = np.concatenate([r[0] for r in results])
    seq_obs = np.concatenate([r[1] for r in results])
    seq_sh = np.concatenate([r[3] for r in results])
    order = np.argsort(all_idx)
    df = pd.DataFrame({
        "feature_idx":          all_idx[order].astype(np.int32),
        "seq_effect_obs":       seq_obs[order],
        "seq_effect_shuffle":   seq_sh[order],
        "seq_delta_intertoken": (seq_obs - seq_sh)[order],
    })
    df.to_csv(out_csv, index=False)
    print(f"    saved → {out_csv}")
    return df


def cohens_d(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    pooled = np.sqrt((a.std(ddof=1)**2 + b.std(ddof=1)**2) / 2.0 + 1e-12)
    return float((a.mean() - b.mean()) / pooled)


def consolidate(pg2_dfs, out_dir: Path):
    """Run H2'_val = MW one-tailed(ESM-2 val seq-Δ > ProtGPT2 val inter-token seq-Δ)."""
    rows = []
    for depth_label, esm_l, pg2_l in MATCHED_PAIRS:
        esm_val_csv = OUT_LW / "esm2" / f"layer_{esm_l}" / "struct_seq_metrics_val.csv"
        if not esm_val_csv.exists():
            print(f"  missing {esm_val_csv} — skipping depth {depth_label}")
            continue
        esm_val = pd.read_csv(esm_val_csv)
        esm_seq = esm_val["seq_delta"].values
        pg2_inter = pg2_dfs[pg2_l]["seq_delta_intertoken"].values

        d = cohens_d(esm_seq, pg2_inter)
        _, p = stats.mannwhitneyu(esm_seq, pg2_inter, alternative="greater")
        rows.append(dict(
            depth=depth_label, esm2_layer=esm_l, protgpt2_layer=pg2_l,
            esm_val_seq_mean=float(esm_seq.mean()),
            pg2_val_intertoken_seq_mean=float(pg2_inter.mean()),
            cohens_d_h2prime_val=d,
            MW_p=float(p),
            significant=bool(p < 0.05 and d > 0),
        ))

    df = pd.DataFrame(rows)
    csv_out = out_dir / "h2prime_valonly.csv"
    df.to_csv(csv_out, index=False)
    print(f"\nWritten → {csv_out}")

    # Readable
    txt = out_dir / "h2prime_valonly.txt"
    lines = [
        "H2' on val-only 150 proteins (inter-token seq-Δ for ProtGPT2)\n",
        "=" * 72 + "\n\n",
        "H2'_val = MW one-tailed(ESM-2 val seq-Δ > ProtGPT2 val inter-token seq-Δ).\n\n",
        df.to_string(index=False, float_format=lambda v: f"{v:+.4f}" if isinstance(v, float) else str(v)) + "\n\n",
    ]
    n_sig = int(df.significant.sum()); n_tot = len(df)
    lines.append(f"H2' supported at {n_sig}/{n_tot} matched depths on val-only.\n")
    with open(txt, "w") as f:
        f.writelines(lines)
    print(f"Report  → {txt}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="analysis_results_h2prime_valonly")
    ap.add_argument("--n-shuffles", type=int, default=5)
    ap.add_argument("--n-jobs", type=int, default=-1)
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    pg2_dfs = {}
    t0 = time.time()
    for _, _, pg2_l in MATCHED_PAIRS:
        pg2_dfs[pg2_l] = run_protgpt2_layer(pg2_l, args.n_shuffles, args.n_jobs)
    print(f"\nAll 5 ProtGPT2 layers done in {(time.time()-t0)/60:.1f} min")

    consolidate(pg2_dfs, out_dir)


if __name__ == "__main__":
    main()
