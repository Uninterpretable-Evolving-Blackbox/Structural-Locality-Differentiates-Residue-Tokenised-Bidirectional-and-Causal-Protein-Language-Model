#!/usr/bin/env python3
"""
experiment_bpe_correction.py — Isolate BPE artefact in ProtGPT2 sequential locality
====================================================================================

Addresses Limitation 4: the mathematical duplication of BPE token hidden states
across constituent residues inflates measured sequential locality for ProtGPT2.

Experiment:
  Recalculate sequential Δ for ProtGPT2 with a strict intra-token exclusion mask.
  When computing neighbor-smoothed activation (z_hat_i), exclude any neighbor j
  that originates from the same BPE parent token as residue i.

  This isolates *inter-token* sequential locality — the true sequential signal
  uncontaminated by BPE duplication.

Comparison:
  Outputs both the original sequential Δ (from struct_seq_metrics.csv) and the
  corrected sequential Δ, so you can directly measure the artefact magnitude.

Usage:
  python experiment_bpe_correction.py \
    --layer-dir outputs_layerwise/protgpt2/layer_18 \
    --pdb-dir cache/pdb_files \
    --save-dir results_bpe_correction

Prerequisites:
  - Z.npy, uids.json, sequences.json, lengths.npy in layer-dir
  - struct_seq_metrics.csv in layer-dir (for comparison)
  - PDB files (for structural locality, unchanged)
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
from scipy import sparse
from scipy.stats import mannwhitneyu
from joblib import Parallel, delayed, cpu_count

from cpu_stage import (
    load_layer, load_ref_seqs,
    build_neighbor_graphs_residue_parallel,
    adj_list_to_sparse,
    build_protein_permutations,
    _cohens_d_vectorized,
    _process_struct_seq_chunk_v3,
    DEFAULT_CONTACT_CUTOFF, DEFAULT_SEQ_GAP_MIN, DEFAULT_TOPK_FRAC,
)

warnings.filterwarnings("ignore")


# =====================================================================
#   BUILD BPE-AWARE SEQUENTIAL NEIGHBOR GRAPH
# =====================================================================

def build_bpe_token_map(uids, seqs_by_uid, token_lengths,
                        tokenizer_name: str = "nferruz/ProtGPT2"):
    """For each residue, record which BPE token it belongs to.

    Returns:
        token_ids: np.ndarray of shape (n_residues_total,) — global token index
                   for each residue. Residues from the same BPE token share an ID.
        res_offsets: dict mapping uid -> global residue offset
        res_lengths: np.ndarray of residue counts per protein
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    res_lengths = np.array([len(seqs_by_uid[uid]) for uid in uids], dtype=np.int32)
    n_res_total = int(res_lengths.sum())

    token_ids = np.zeros(n_res_total, dtype=np.int64)

    res_offsets = {}
    off = 0
    for uid, Lr in zip(uids, res_lengths):
        res_offsets[uid] = off
        off += int(Lr)

    global_tok_id = 0
    res_cursor = 0

    for uid, Lt in tqdm(zip(uids, token_lengths), total=len(uids),
                         desc="BPE token map", leave=False):
        seq = seqs_by_uid[uid]
        ids = tok(seq, add_special_tokens=False)["input_ids"]
        if len(ids) != int(Lt):
            raise ValueError(f"Token count mismatch for {uid}: "
                             f"got {len(ids)}, expected {Lt}")
        tokens = tok.convert_ids_to_tokens(ids)

        cursor = 0
        base_res = res_offsets[uid]
        for t in tokens:
            piece = t.replace("\u2581", "").replace("\u0120", "").replace(" ", "")
            Lp = len(piece) if piece else 0
            if Lp > 0:
                if seq[cursor:cursor + Lp] != piece:
                    raise ValueError(f"Span decode mismatch for {uid} at {cursor}")
                for r in range(cursor, cursor + Lp):
                    token_ids[base_res + r] = global_tok_id
                cursor += Lp
            global_tok_id += 1

        if cursor != len(seq):
            raise ValueError(f"Span coverage mismatch for {uid}")
        res_cursor += len(seq)

    return token_ids, res_offsets, res_lengths


def build_bpe_corrected_seq_neighbors(uids, res_lengths, res_offsets,
                                       token_ids):
    """Build sequential neighbor lists (±1, ±2) EXCLUDING neighbors from
    the same BPE token.

    This is the key correction: if residues i and j share the same BPE parent
    token, j is NOT included in N_i even if |i - j| <= 2.
    """
    n_res = int(res_lengths.sum())
    seq_adj = [[] for _ in range(n_res)]

    for uid, Lr in zip(uids, res_lengths):
        base = res_offsets[uid]
        Lr = int(Lr)
        for r in range(Lr):
            global_r = base + r
            tok_r = token_ids[global_r]
            for d in (-2, -1, 1, 2):
                rr = r + d
                if 0 <= rr < Lr:
                    global_rr = base + rr
                    tok_rr = token_ids[global_rr]
                    # EXCLUSION: skip if same BPE token
                    if tok_rr != tok_r:
                        seq_adj[global_r].append(global_rr)

    return seq_adj


def build_original_seq_neighbors(uids, res_lengths, res_offsets):
    """Build the original (uncorrected) sequential neighbor lists for
    comparison — same logic as cpu_stage.py."""
    n_res = int(res_lengths.sum())
    seq_adj = [[] for _ in range(n_res)]

    for uid, Lr in zip(uids, res_lengths):
        base = res_offsets[uid]
        Lr = int(Lr)
        for r in range(Lr):
            for d in (-2, -1, 1, 2):
                rr = r + d
                if 0 <= rr < Lr:
                    seq_adj[base + r].append(base + rr)

    return seq_adj


# =====================================================================
#   COMPUTE SEQUENTIAL LOCALITY WITH BOTH NEIGHBOR DEFINITIONS
# =====================================================================

def compute_seq_deltas(Z, A_proj, seq_adj, res_lengths, n_shuffles=3,
                       topk_frac=0.10, n_jobs=-1):
    """Compute sequential Δ (observed - shuffled Cohen's d) for all features."""
    n_res = int(res_lengths.sum())
    n_features = int(Z.shape[1])
    chunk_size = 256

    # Convert to sparse
    A_seq, deg_seq = adj_list_to_sparse(seq_adj, n_res)

    # Dummy structural adjacency (zeros — we only need sequential here)
    A_struct = sparse.csr_matrix((n_res, n_res), dtype=np.float32)
    deg_struct = np.zeros(n_res, dtype=np.float32)

    # Permutations
    perm_indices = build_protein_permutations(res_lengths, n_shuffles)

    # Process chunks
    n_chunks = (n_features + chunk_size - 1) // chunk_size
    mem_pw = n_res * chunk_size * 4 * 5 / 1e9
    max_safe = max(1, int(40.0 / max(mem_pw, 0.1)))
    eff_jobs = min(n_jobs if n_jobs > 0 else cpu_count(), max_safe)

    results = Parallel(n_jobs=eff_jobs, verbose=5)(
        delayed(_process_struct_seq_chunk_v3)(
            ci, chunk_size, Z, A_proj,
            A_seq, deg_seq, A_struct, deg_struct,
            perm_indices, n_features, topk_frac)
        for ci in range(n_chunks))

    # Assemble
    all_idx = np.concatenate([r[0] for r in results])
    all_seq_obs = np.concatenate([r[1] for r in results])
    all_seq_sh = np.concatenate([r[3] for r in results])

    order = np.argsort(all_idx)
    seq_delta = (all_seq_obs - all_seq_sh)[order]

    return seq_delta


# =====================================================================
#                           MAIN
# =====================================================================

def main():
    ap = argparse.ArgumentParser(
        description="BPE intra-token exclusion for ProtGPT2 sequential locality")
    ap.add_argument("--layer-dir", required=True,
                    help="Path to outputs_layerwise/protgpt2/layer_N")
    ap.add_argument("--pdb-dir", default="cache/pdb_files")
    ap.add_argument("--tokenizer", default="nferruz/ProtGPT2")
    ap.add_argument("--n-shuffles", type=int, default=3)
    ap.add_argument("--topk-frac", type=float, default=DEFAULT_TOPK_FRAC)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--save-dir", default="results_bpe_correction")
    # Optional: also compare against ESM-2
    ap.add_argument("--esm2-layer-dir", default=None,
                    help="Path to ESM-2 matched layer for direct comparison")
    args = ap.parse_args()

    layer_dir = Path(args.layer_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    n_jobs = args.n_jobs if args.n_jobs > 0 else cpu_count()

    print("=" * 70)
    print("  BPE INTRA-TOKEN EXCLUSION EXPERIMENT")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    Z, uids, tok_lengths = load_layer(layer_dir)
    ref_seqs = load_ref_seqs(layer_dir)
    print(f"  Z: {Z.shape} ({Z.shape[0]} tokens, {Z.shape[1]} features)")
    print(f"  Proteins: {len(uids)}")

    # ------------------------------------------------------------------
    # 2. Build BPE token map and projection matrix
    # ------------------------------------------------------------------
    print("\n  Building BPE token map...")
    token_ids, res_offsets, res_lengths = build_bpe_token_map(
        uids, ref_seqs, tok_lengths, args.tokenizer)

    # Token-to-residue projection (same as cpu_stage.py)
    from cpu_stage import build_protgpt2_projection
    A_proj, _, _ = build_protgpt2_projection(
        uids, ref_seqs, tok_lengths, args.tokenizer)

    n_res = int(res_lengths.sum())
    print(f"  Total residues: {n_res:,}")

    # Count intra-token neighbors that will be excluded
    n_excluded = 0
    n_total_seq_nbrs = 0
    for uid, Lr in zip(uids, res_lengths):
        base = res_offsets[uid]
        Lr = int(Lr)
        for r in range(Lr):
            global_r = base + r
            tok_r = token_ids[global_r]
            for d in (-2, -1, 1, 2):
                rr = r + d
                if 0 <= rr < Lr:
                    n_total_seq_nbrs += 1
                    if token_ids[base + rr] == tok_r:
                        n_excluded += 1

    pct_excluded = 100 * n_excluded / max(n_total_seq_nbrs, 1)
    print(f"  Intra-token neighbors excluded: {n_excluded:,} / "
          f"{n_total_seq_nbrs:,} ({pct_excluded:.1f}%)")

    # ------------------------------------------------------------------
    # 3. Build both neighbor graphs
    # ------------------------------------------------------------------
    print("\n  Building ORIGINAL sequential neighbors...")
    seq_adj_original = build_original_seq_neighbors(uids, res_lengths, res_offsets)

    print("  Building CORRECTED sequential neighbors (BPE-excluded)...")
    seq_adj_corrected = build_bpe_corrected_seq_neighbors(
        uids, res_lengths, res_offsets, token_ids)

    # Verify edge counts
    orig_edges = sum(len(nbrs) for nbrs in seq_adj_original)
    corr_edges = sum(len(nbrs) for nbrs in seq_adj_corrected)
    print(f"  Original edges:  {orig_edges:,}")
    print(f"  Corrected edges: {corr_edges:,} ({corr_edges/max(orig_edges,1)*100:.1f}%)")

    # ------------------------------------------------------------------
    # 4. Compute sequential Δ under both definitions
    # ------------------------------------------------------------------
    print(f"\n  Computing ORIGINAL sequential Δ...")
    seq_delta_original = compute_seq_deltas(
        Z, A_proj, seq_adj_original, res_lengths,
        n_shuffles=args.n_shuffles, topk_frac=args.topk_frac, n_jobs=n_jobs)

    print(f"\n  Computing CORRECTED sequential Δ (BPE-excluded)...")
    seq_delta_corrected = compute_seq_deltas(
        Z, A_proj, seq_adj_corrected, res_lengths,
        n_shuffles=args.n_shuffles, topk_frac=args.topk_frac, n_jobs=n_jobs)

    # ------------------------------------------------------------------
    # 5. Compare
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("  RESULTS")
    print(f"{'=' * 70}")

    print(f"\n  ProtGPT2 Sequential Δ:")
    print(f"    Original:   mean={seq_delta_original.mean():.4f} "
          f"+/- {seq_delta_original.std():.4f}")
    print(f"    Corrected:  mean={seq_delta_corrected.mean():.4f} "
          f"+/- {seq_delta_corrected.std():.4f}")
    reduction = 1 - seq_delta_corrected.mean() / max(seq_delta_original.mean(), 1e-8)
    print(f"    Reduction:  {reduction*100:.1f}%")

    # Wilcoxon signed-rank test
    from scipy.stats import wilcoxon
    stat, p = wilcoxon(seq_delta_corrected, seq_delta_original)
    print(f"    Wilcoxon p: {p:.2e}")

    # Optional: compare corrected ProtGPT2 against ESM-2
    esm2_delta = None
    if args.esm2_layer_dir:
        esm2_dir = Path(args.esm2_layer_dir)
        esm2_ss = pd.read_csv(esm2_dir / "struct_seq_metrics.csv")
        esm2_delta = esm2_ss["seq_delta"].values

        print(f"\n  ESM-2 Sequential Δ: mean={esm2_delta.mean():.4f} "
              f"+/- {esm2_delta.std():.4f}")

        # Mann-Whitney: corrected ProtGPT2 vs ESM-2
        U, p_mw = mannwhitneyu(seq_delta_corrected, esm2_delta,
                                alternative="greater")
        pooled_std = np.sqrt((seq_delta_corrected.std()**2 +
                              esm2_delta.std()**2) / 2)
        d = (seq_delta_corrected.mean() - esm2_delta.mean()) / (pooled_std + 1e-8)
        print(f"  Corrected ProtGPT2 > ESM-2:")
        print(f"    Cohen's d: {d:+.4f}")
        print(f"    MW p:      {p_mw:.2e}")
        verdict = "STILL SUPPORTED" if p_mw < 0.05 else "NO LONGER SUPPORTED"
        print(f"    H2 after correction: {verdict}")

    # ------------------------------------------------------------------
    # 6. Save results
    # ------------------------------------------------------------------
    n_features = len(seq_delta_original)
    df = pd.DataFrame({
        "feature_idx": np.arange(n_features),
        "seq_delta_original": seq_delta_original,
        "seq_delta_corrected": seq_delta_corrected,
        "delta_change": seq_delta_corrected - seq_delta_original,
    })
    df.to_csv(save_dir / "bpe_correction_per_feature.csv", index=False)

    summary = {
        "original_mean": float(seq_delta_original.mean()),
        "original_std": float(seq_delta_original.std()),
        "corrected_mean": float(seq_delta_corrected.mean()),
        "corrected_std": float(seq_delta_corrected.std()),
        "pct_reduction": float(reduction * 100),
        "pct_neighbors_excluded": float(pct_excluded),
        "wilcoxon_p": float(p),
    }
    if esm2_delta is not None:
        summary["esm2_mean"] = float(esm2_delta.mean())
        summary["h2_corrected_cohens_d"] = float(d)
        summary["h2_corrected_p"] = float(p_mw)

    pd.DataFrame([summary]).to_csv(save_dir / "bpe_correction_summary.csv",
                                    index=False)

    # ------------------------------------------------------------------
    # 7. Plots
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel A: Histogram of original vs corrected
    ax = axes[0]
    bins = np.linspace(
        min(seq_delta_original.min(), seq_delta_corrected.min()),
        max(seq_delta_original.max(), seq_delta_corrected.max()),
        60)
    ax.hist(seq_delta_original, bins=bins, alpha=0.5, label="Original",
            color="#ff7f0e", density=True)
    ax.hist(seq_delta_corrected, bins=bins, alpha=0.5, label="BPE-corrected",
            color="#2ca02c", density=True)
    ax.axvline(seq_delta_original.mean(), color="#ff7f0e", ls="--", lw=2)
    ax.axvline(seq_delta_corrected.mean(), color="#2ca02c", ls="--", lw=2)
    ax.set_xlabel("Sequential Δ")
    ax.set_ylabel("Density")
    ax.set_title("ProtGPT2: Original vs BPE-corrected")
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel B: Scatter original vs corrected per feature
    ax = axes[1]
    ax.scatter(seq_delta_original, seq_delta_corrected, s=3, alpha=0.3,
               color="#ff7f0e")
    lims = [min(seq_delta_original.min(), seq_delta_corrected.min()),
            max(seq_delta_original.max(), seq_delta_corrected.max())]
    ax.plot(lims, lims, "k--", lw=1)
    ax.set_xlabel("Original Sequential Δ")
    ax.set_ylabel("BPE-corrected Sequential Δ")
    ax.set_title("Per-feature Comparison")
    ax.grid(alpha=0.3)

    # Panel C: Bar comparison (optionally with ESM-2)
    ax = axes[2]
    labels = ["ProtGPT2\n(original)", "ProtGPT2\n(BPE-corrected)"]
    means = [seq_delta_original.mean(), seq_delta_corrected.mean()]
    stds = [seq_delta_original.std(), seq_delta_corrected.std()]
    colors = ["#ff7f0e", "#2ca02c"]

    if esm2_delta is not None:
        labels.append("ESM-2")
        means.append(esm2_delta.mean())
        stds.append(esm2_delta.std())
        colors.append("#1f77b4")

    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.7,
                  edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Mean Sequential Δ")
    ax.set_title("H2 After BPE Correction")
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(save_dir / "bpe_correction_results.png", dpi=300)
    fig.savefig(save_dir / "bpe_correction_results.pdf")
    plt.close(fig)

    print(f"\n  Results saved to {save_dir}/")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
