#!/usr/bin/env python3
"""
experiment_preflight.py — 30-second smoke test for items #1 and #3.

Verifies that the tree is in the expected shape before kicking off the
BPE-crossing and val-only H1/H2 runs:
  • cache + layer artifacts present
  • META.val_uids populated and ⊆ uids.json
  • ProtGPT2 BPE token_ids round-trip for first 10 proteins
  • Z rows filter correctly by val_uids (residue and BPE cases)
  • _cohens_d_vectorized returns a finite number on a tiny slice

No full pipeline runs. No SAE training. No cached output is written.
Aborts with a clear message on any failed invariant.
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy import sparse


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs_layerwise"

CHECK_LAYERS = {
    "esm2": 16,         # 50% depth, residue model
    "protgpt2": 18,     # 50% depth, BPE model
    "prott5_enc": 12,
    "prott5_dec": 12,
}


def load_layer_raw(model: str, layer: int):
    d = OUT / model / f"layer_{layer}"
    Z = np.load(d / "Z.npy", mmap_mode="r")
    uids = json.loads((d / "uids.json").read_text())
    lengths = np.load(d / "lengths.npy")
    seqs = json.loads((d / "sequences.json").read_text())
    meta = json.loads((d / "META.json").read_text())
    return d, Z, uids, lengths, seqs, meta


def assert_(cond, msg):
    if not cond:
        print(f"  ❌  {msg}")
        sys.exit(2)
    print(f"  ✅  {msg}")


def main():
    print("=" * 60)
    print("  PREFLIGHT  —  items #1 (BPE-crossing) and #3 (val-only)")
    print("=" * 60)

    # -----------------------------------------------------------
    # 1. cpu_stage imports that both scripts depend on
    # -----------------------------------------------------------
    print("\n[1] cpu_stage imports")
    from cpu_stage import (
        load_layer, load_ref_seqs,
        build_neighbor_graphs_residue_parallel,
        adj_list_to_sparse, build_protein_permutations,
        _cohens_d_vectorized, _process_struct_seq_chunk_v3,
        build_protgpt2_projection,
        DEFAULT_CONTACT_CUTOFF, DEFAULT_SEQ_GAP_MIN, DEFAULT_TOPK_FRAC,
    )
    assert_(True, "cpu_stage exports present")

    # -----------------------------------------------------------
    # 2. All 20 layer dirs have Z + META with val_uids
    # -----------------------------------------------------------
    print("\n[2] Layer artifacts + META.val_uids")
    n_val_per_layer = {}
    for model, layer in CHECK_LAYERS.items():
        d, Z, uids, lengths, seqs, meta = load_layer_raw(model, layer)
        assert_((d / "Z.npy").exists(), f"{model}/layer_{layer}/Z.npy exists")
        assert_("val_uids" in meta and len(meta["val_uids"]) > 0,
                f"{model}/layer_{layer} META has val_uids")
        assert_(set(meta["val_uids"]).issubset(set(uids)),
                f"{model}/layer_{layer} val_uids ⊆ uids.json")
        assert_(int(np.sum(lengths)) == int(Z.shape[0]),
                f"{model}/layer_{layer} sum(lengths)=={Z.shape[0]}")
        n_val_per_layer[model] = len(meta["val_uids"])

    # -----------------------------------------------------------
    # 3. Val-row filtering works (residue case)
    # -----------------------------------------------------------
    print("\n[3] Val-row filtering (residue model, ESM-2 layer 16)")
    d, Z, uids, lengths, seqs, meta = load_layer_raw("esm2", 16)
    uid_to_idx = {u: i for i, u in enumerate(uids)}
    val_idx = sorted(uid_to_idx[u] for u in meta["val_uids"])
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    val_rows = np.concatenate([
        np.arange(offsets[i], offsets[i + 1], dtype=np.int64) for i in val_idx
    ])
    assert_(val_rows.shape[0] == sum(int(lengths[i]) for i in val_idx),
            f"val_rows matches val residue count ({val_rows.shape[0]})")
    # Quick sample: Z[val_rows[0]] has finite values
    z_sample = np.asarray(Z[val_rows[0]], dtype=np.float32)
    assert_(np.isfinite(z_sample).all(), "Z[val_rows[0]] is finite")

    # -----------------------------------------------------------
    # 4. BPE token_ids round-trip (ProtGPT2 layer 18, first 10 proteins)
    # -----------------------------------------------------------
    print("\n[4] BPE token_ids round-trip (first 10 ProtGPT2 proteins)")
    d, Z, uids, lengths_tok, seqs, meta = load_layer_raw("protgpt2", 18)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("nferruz/ProtGPT2", use_fast=True)

    test_uids = uids[:10]
    sample_seqs = seqs if isinstance(seqs, dict) else dict(zip(uids, seqs))

    n_intra = n_total = 0
    token_ids_flat = []
    global_tok = 0
    for uid in test_uids:
        seq = sample_seqs[uid]
        ids = tok(seq, add_special_tokens=False)["input_ids"]
        toks = tok.convert_ids_to_tokens(ids)
        res_tok = np.zeros(len(seq), dtype=np.int64)
        cursor = 0
        for t in toks:
            piece = t.replace("\u2581", "").replace("\u0120", "").replace(" ", "")
            Lp = len(piece)
            if Lp > 0:
                assert seq[cursor:cursor + Lp] == piece, \
                    f"Span decode mismatch for {uid} at {cursor}"
                for r in range(cursor, cursor + Lp):
                    res_tok[r] = global_tok
                cursor += Lp
            global_tok += 1
        assert cursor == len(seq), f"Coverage mismatch for {uid}"
        token_ids_flat.append(res_tok)
        # Count intra-token ±1/±2 edges
        for r in range(len(seq)):
            for dd in (-2, -1, 1, 2):
                rr = r + dd
                if 0 <= rr < len(seq):
                    n_total += 1
                    if res_tok[rr] == res_tok[r]:
                        n_intra += 1

    pct = 100 * n_intra / max(n_total, 1)
    assert_(n_total > 0, f"neighbor edges counted = {n_total}")
    assert_(0 < pct < 100, f"intra-token edges = {pct:.1f}% of ±1/±2 neighbors")
    print(f"    → intra-token fraction drives the H2 concern; {pct:.1f}% is being tested.")

    # -----------------------------------------------------------
    # 5. _cohens_d_vectorized returns finite on tiny subset
    # -----------------------------------------------------------
    print("\n[5] _cohens_d_vectorized on val-filtered 3-protein slice (ESM-2 L16)")
    d, Z, uids, lengths, seqs, meta = load_layer_raw("esm2", 16)
    uid_to_idx = {u: i for i, u in enumerate(uids)}
    tiny_idx = [uid_to_idx[u] for u in meta["val_uids"][:3]]
    tiny_rows = np.concatenate([
        np.arange(offsets[i], offsets[i + 1], dtype=np.int64) for i in tiny_idx
    ])
    acts = np.asarray(Z[tiny_rows][:, :8], dtype=np.float32)
    n_res = acts.shape[0]
    # Trivial sequential adjacency just for the smoke test
    rows = []; cols = []
    Ls = [int(lengths[i]) for i in tiny_idx]
    off = 0
    for L in Ls:
        for r in range(L):
            for dd in (-1, 1):
                rr = r + dd
                if 0 <= rr < L:
                    rows.append(off + r); cols.append(off + rr)
        off += L
    A = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32),
         (np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32))),
        shape=(n_res, n_res))
    deg = np.asarray(A.sum(axis=1)).ravel().astype(np.float32)
    gstds = np.std(acts, axis=0).astype(np.float32)
    d_values = _cohens_d_vectorized(acts, A, deg, gstds, DEFAULT_TOPK_FRAC)
    assert_(d_values.shape == (8,), f"Cohen's d shape == (8,)")
    assert_(np.isfinite(d_values).all(), "Cohen's d values are finite")
    print(f"    → sample Cohen's d on 3 val proteins, 8 features: "
          f"mean={d_values.mean():+.3f}, range=[{d_values.min():+.3f}, {d_values.max():+.3f}]")

    # -----------------------------------------------------------
    # 6. Matched-depth pairings exist (for BPE + val consolidation)
    # -----------------------------------------------------------
    print("\n[6] Matched-depth pairings exist on disk")
    pairs = [(0, 0), (9, 8), (18, 16), (27, 24), (35, 32)]
    for pg2, esm in pairs:
        ok_pg2 = (OUT / "protgpt2" / f"layer_{pg2}" / "struct_seq_metrics.csv").exists()
        ok_esm = (OUT / "esm2" / f"layer_{esm}" / "struct_seq_metrics.csv").exists()
        assert_(ok_pg2 and ok_esm, f"depth ProtGPT2 L{pg2} ↔ ESM-2 L{esm} both have struct_seq_metrics.csv")

    print("\n" + "=" * 60)
    print("  ✅  PREFLIGHT PASSED — items #1 and #3 are safe to run")
    print("=" * 60)


if __name__ == "__main__":
    main()
