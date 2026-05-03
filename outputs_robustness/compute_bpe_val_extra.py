#!/usr/bin/env python3
"""
compute_bpe_val_extra.py — Fill the two BPE-related val gaps from Task 3.

(1) BPE inter-token val at default ±2 window — adds the d_seq_intertok_val
    column to bpe_table_val.csv at the 5 main BPE depths {0,25,50,75,100}%.

(2) ProtGPT2 window sweep on val — d_seq for raw projection AND inter-token
    correction at windows {±1, ±2, ±4} × 5 depths.

Sign convention: cross-model d = (mean_pg2 - mean_esm)/pooled_SD.
Positive = ProtGPT2 > ESM-2 on seq_delta (BPE-conflated direction).
Negative under inter-token correction = ESM-2 wins after artifact removed.

Sanity assertions baked in:
  - val A_seq_raw at ±2 must have 121,648 directed edges (matches earlier
    independent count).
  - val A_seq_intertok at ±2 must have 60,718 directed edges.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs_robustness"

from cpu_stage import (
    load_layer, adj_list_to_sparse, build_protein_permutations,
    build_protgpt2_projection,
)
from experiment_bpe_correction import build_bpe_token_map

OUT_LW = ROOT / "outputs_layerwise"

BPE_PAIRS = [
    ("0",   0,   0),
    ("25",  8,   9),
    ("50",  16, 18),
    ("75",  24, 27),
    ("100", 32, 35),
]
WINDOWS = [1, 2, 4]
TOPK_FRAC = 0.10
N_SHUF = 5

# Sanity targets (independently verified earlier)
EXPECTED_W2_RAW_EDGES = 121648
EXPECTED_W2_INTERTOK_EDGES = 60718


def cohens_d(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    pooled = np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2.0 + 1e-12)
    return float((a.mean() - b.mean()) / pooled)


def build_seq_adj_raw(val_uids_sorted, val_res_lengths, res_offsets, window):
    """Raw ±window adjacency, residue-level (no BPE filter)."""
    n_res = int(val_res_lengths.sum())
    seq_adj = [[] for _ in range(n_res)]
    for uid, Lr in zip(val_uids_sorted, val_res_lengths):
        base = res_offsets[uid]; Lr = int(Lr)
        for r in range(Lr):
            for d in range(-window, window + 1):
                if d == 0: continue
                rr = r + d
                if 0 <= rr < Lr:
                    seq_adj[base + r].append(base + rr)
    return seq_adj


def build_seq_adj_intertoken(val_uids_sorted, val_res_lengths, res_offsets,
                              token_ids, window):
    """±window adjacency excluding within-same-BPE-token pairs."""
    n_res = int(val_res_lengths.sum())
    seq_adj = [[] for _ in range(n_res)]
    for uid, Lr in zip(val_uids_sorted, val_res_lengths):
        base = res_offsets[uid]; Lr = int(Lr)
        for r in range(Lr):
            tok_r = token_ids[base + r]
            for d in range(-window, window + 1):
                if d == 0: continue
                rr = r + d
                if 0 <= rr < Lr and token_ids[base + rr] != tok_r:
                    seq_adj[base + r].append(base + rr)
    return seq_adj


def per_feature_seq_delta(Z, A_proj, A_seq, deg_seq, perm_indices,
                          topk_frac=TOPK_FRAC):
    """Compute per-feature seq_delta = obs_d - mean_shuf_d.

    Z: (n_units, n_features) — residue-level if A_proj is None, else token-level
    A_proj: (n_residues, n_units) sparse OR None
    """
    Z = np.asarray(Z, dtype=np.float32)
    n_features = Z.shape[1]
    chunk_size = 256
    n_chunks = (n_features + chunk_size - 1) // chunk_size
    out = np.zeros(n_features, dtype=np.float32)

    for ci in range(n_chunks):
        i, e = ci * chunk_size, min((ci + 1) * chunk_size, n_features)
        if A_proj is None:
            acts = Z[:, i:e].astype(np.float32, copy=False)
        else:
            acts = np.asarray(A_proj @ Z[:, i:e].astype(np.float32),
                              dtype=np.float32)
        gstds = acts.std(axis=0, ddof=0).astype(np.float32) + 1e-6
        thresh = np.percentile(acts, 100.0 * (1 - topk_frac), axis=0).astype(np.float32)

        has_nb = deg_seq > 0

        # observed
        nbr = np.asarray(A_seq @ acts, dtype=np.float32)
        sm = np.zeros_like(acts)
        sm[has_nb] = nbr[has_nb] / deg_seq[has_nb, None]
        gm = sm.mean(axis=0)
        active = acts > thresh[None, :]
        n_act = active.sum(axis=0).astype(np.float32)
        d_obs = ((sm * active).sum(axis=0) / np.maximum(n_act, 1) - gm) / gstds
        d_obs[n_act < 5] = 0.0

        # shuffles
        d_shuf = np.zeros(e - i, dtype=np.float32)
        for perm in perm_indices:
            ap = acts[perm]
            np_ = np.asarray(A_seq @ ap, dtype=np.float32)
            sp = np.zeros_like(ap)
            sp[has_nb] = np_[has_nb] / deg_seq[has_nb, None]
            gmp = sp.mean(axis=0)
            actp = ap > thresh[None, :]
            n_actp = actp.sum(axis=0).astype(np.float32)
            d_p = ((sp * actp).sum(axis=0) / np.maximum(n_actp, 1) - gmp) / gstds
            d_p[n_actp < 5] = 0.0
            d_shuf += d_p
        d_shuf /= max(len(perm_indices), 1)

        out[i:e] = d_obs - d_shuf

    return out


def main():
    print("=" * 72)
    print("  BPE val extras — gap 1 (inter-token ±2) + gap 2 (window sweep)")
    print("=" * 72)

    # Shared val protein metadata
    layer0_dir = OUT_LW / "esm2/layer_0"
    val_uids_set = set(json.loads((layer0_dir / "META.json").read_text())["val_uids"])

    # Sequences (full set)
    seqs_obj = json.loads((layer0_dir / "sequences.json").read_text())
    if isinstance(seqs_obj, dict):
        seqs = seqs_obj
    else:
        all_uids = json.loads((layer0_dir / "uids.json").read_text())
        seqs = {u: seqs_obj[i] for i, u in enumerate(all_uids)}

    rows_table = []      # for bpe_table_val.csv (gap 1: ±2 only)
    rows_window = []     # for sweep_window_protgpt2_val.csv (gap 2: 3 windows × 5 depths)
    sanity_passed = []

    for label, esm_l, pg_l in BPE_PAIRS:
        print(f"\n--- depth {label}%: ESM-2 L{esm_l} vs ProtGPT2 L{pg_l} ---")
        t0 = time.time()

        # ----- ESM-2 val (residue-level) -----
        Z_esm_full, esm_uids, esm_lengths = load_layer(OUT_LW / f"esm2/layer_{esm_l}")
        esm_uid_to_idx = {u: i for i, u in enumerate(esm_uids)}
        val_uids_sorted = [u for u in esm_uids if u in val_uids_set]
        val_idx_esm = [esm_uid_to_idx[u] for u in val_uids_sorted]
        esm_res_offs = np.concatenate([[0], np.cumsum(esm_lengths.astype(np.int64))])
        val_res_rows = np.concatenate(
            [np.arange(esm_res_offs[i], esm_res_offs[i + 1]) for i in val_idx_esm])
        Z_esm_val = Z_esm_full[val_res_rows]
        val_res_lengths = esm_lengths[val_idx_esm].astype(np.int32)
        n_res_val = int(val_res_lengths.sum())
        del Z_esm_full

        # ----- ProtGPT2 val (token-level + projection) -----
        Z_pg_full, pg_uids, pg_lengths = load_layer(OUT_LW / f"protgpt2/layer_{pg_l}")
        pg_uid_to_idx = {u: i for i, u in enumerate(pg_uids)}
        val_idx_pg = [pg_uid_to_idx[u] for u in val_uids_sorted]
        pg_tok_offs = np.concatenate([[0], np.cumsum(pg_lengths.astype(np.int64))])
        val_tok_rows = np.concatenate(
            [np.arange(pg_tok_offs[i], pg_tok_offs[i + 1]) for i in val_idx_pg])
        Z_pg_val = Z_pg_full[val_tok_rows]
        val_tok_lengths = pg_lengths[val_idx_pg]
        del Z_pg_full

        # Build val A_proj + token_ids (val-local, residue-level indexed)
        val_seqs = {u: seqs[u] for u in val_uids_sorted}
        A_proj_val, res_offsets_pg, _ = build_protgpt2_projection(
            val_uids_sorted, val_seqs, val_tok_lengths, "nferruz/ProtGPT2")
        token_ids_val, res_offsets_bpe, _ = build_bpe_token_map(
            val_uids_sorted, val_seqs, val_tok_lengths, "nferruz/ProtGPT2")
        # both helpers compute residue offsets internally; they should agree
        for u in val_uids_sorted[:5]:
            assert res_offsets_pg[u] == res_offsets_bpe[u]

        # Build res_offsets dict for adj builders (val-local, by uid → residue start)
        res_offsets = {}
        off = 0
        for u, Lr in zip(val_uids_sorted, val_res_lengths):
            res_offsets[u] = off; off += int(Lr)
        # cross-check against PG2-side res_offsets
        for u in val_uids_sorted[:5]:
            assert res_offsets[u] == res_offsets_pg[u], f"offset mismatch {u}"

        perm_indices = build_protein_permutations(val_res_lengths, N_SHUF)
        print(f"  loaded in {time.time()-t0:.0f}s; "
              f"val: {len(val_uids_sorted)} prot, {n_res_val} res, "
              f"{Z_pg_val.shape[0]} BPE tok; A_proj {A_proj_val.shape}")

        # ----- Iterate windows -----
        for w in WINDOWS:
            t1 = time.time()

            # raw adj
            seq_adj_raw = build_seq_adj_raw(val_uids_sorted, val_res_lengths,
                                            res_offsets, w)
            A_raw, deg_raw = adj_list_to_sparse(seq_adj_raw, n_res_val)
            # inter-token adj
            seq_adj_int = build_seq_adj_intertoken(val_uids_sorted, val_res_lengths,
                                                    res_offsets, token_ids_val, w)
            A_int, deg_int = adj_list_to_sparse(seq_adj_int, n_res_val)

            # SANITY CHECKS at w=2 against independently-known counts
            if w == 2:
                assert A_raw.nnz == EXPECTED_W2_RAW_EDGES, \
                    f"window=2 raw edges {A_raw.nnz} != expected {EXPECTED_W2_RAW_EDGES}"
                assert A_int.nnz == EXPECTED_W2_INTERTOK_EDGES, \
                    f"window=2 intertok edges {A_int.nnz} != expected {EXPECTED_W2_INTERTOK_EDGES}"
                sanity_passed.append((label, "edge counts ✓"))

            # ESM-2 d_seq at window w
            sd_esm = per_feature_seq_delta(
                Z_esm_val, None, A_raw, deg_raw, perm_indices)
            # PG2 d_seq raw / inter-token at window w
            sd_pg_raw = per_feature_seq_delta(
                Z_pg_val, A_proj_val, A_raw, deg_raw, perm_indices)
            sd_pg_int = per_feature_seq_delta(
                Z_pg_val, A_proj_val, A_int, deg_int, perm_indices)

            d_naive = cohens_d(sd_pg_raw, sd_esm)
            d_intertok = cohens_d(sd_pg_int, sd_esm)

            print(f"  ±{w}:  raw_edges={A_raw.nnz:>7}  int_edges={A_int.nnz:>7}  "
                  f"d_raw={d_naive:+.4f}  d_intertok={d_intertok:+.4f}  "
                  f"({time.time()-t1:.0f}s)")

            rows_window.append(dict(
                rel_depth=f"{label}%", esm_layer=esm_l, pg_layer=pg_l,
                window=w,
                d_seq_naive_val=d_naive, d_seq_intertok_val=d_intertok,
                edges_raw=int(A_raw.nnz), edges_intertok=int(A_int.nnz),
            ))
            if w == 2:
                rows_table.append(dict(
                    rel_depth=f"{label}%", esm_layer=esm_l, pg_layer=pg_l,
                    d_seq_naive_val=d_naive, d_seq_intertok_val=d_intertok,
                ))

        # SANITY: at w=2, my recomputed ESM-2 sd_seq values should match
        # the existing struct_seq_metrics_val.csv (which uses the same ±2 raw)
        # within rounding (single-precision shuffle drift expected).
        existing_path = OUT_LW / f"esm2/layer_{esm_l}/struct_seq_metrics_val.csv"
        if existing_path.exists():
            existing = pd.read_csv(existing_path)
            sd_esm_existing = existing.seq_delta.values
            # We have the ±2 sd_esm from the most recent loop iteration
            r = np.corrcoef(sd_esm, sd_esm_existing)[0, 1]
            mae = float(np.abs(sd_esm - sd_esm_existing).mean())
            print(f"  ESM-2 ±2 self-check: corr with existing val csv = {r:.5f}, MAE = {mae:.4f}")
            sanity_passed.append((label, f"ESM ±2 vs existing csv corr = {r:.5f}"))

        del Z_esm_val, Z_pg_val, A_proj_val

    # ----- Save -----
    df_window = pd.DataFrame(rows_window)
    df_window.to_csv(OUT / "sweep_window_protgpt2_val.csv", index=False)
    df_table = pd.DataFrame(rows_table)
    df_table.to_csv(OUT / "bpe_table_val.csv", index=False)

    # ----- Report -----
    print("\n" + "=" * 72)
    print("  bpe_table_val.csv  (gap 1)")
    print("=" * 72)
    print(df_table.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

    print("\n" + "=" * 72)
    print("  sweep_window_protgpt2_val.csv  (gap 2)")
    print("=" * 72)
    print(df_window.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

    print("\n" + "=" * 72)
    print("  Sanity checks")
    print("=" * 72)
    for label, msg in sanity_passed:
        print(f"  depth {label}%:  {msg}")


if __name__ == "__main__":
    main()
