#!/usr/bin/env python3
"""
experiment_stability.py — Seed replication & depth interpolation
================================================================

Addresses Limitations 1 (no seed replication) and 5 (sparse layer sampling).

Two sub-experiments:

A. SEED REPLICATION (Limitation 1):
   Train 3 SAEs with different random seeds at the 50% depth layer for each
   model. Compute pairwise cosine similarity of recovered feature dictionaries
   (decoder weight matrices). High mean cosine similarity (>0.85) demonstrates
   that the feature space is stable across initialisations.

   Method: For two decoder matrices D1, D2 each (F, d_model), we find the
   best-match feature in D2 for each feature in D1 via max cosine similarity,
   then report the mean of these best-match similarities.

B. DEPTH INTERPOLATION (Limitation 5):
   Sample ESM-2 at every 4th layer between 0 and 16 (layers 0, 4, 8, 12, 16)
   to find the inflection point where physicochemical biases (layer 0 Δ=0.059)
   transition to contextual structural representations (layer 8 Δ=0.018).

Usage:
  # Seed replication at 50% depth
  python experiment_stability.py seed \
    --device cuda \
    --save-dir results_stability

  # Depth interpolation for ESM-2
  python experiment_stability.py depth \
    --device cuda \
    --save-dir results_stability

  # Both
  python experiment_stability.py both \
    --device cuda \
    --save-dir results_stability

Prerequisites:
  - cache/sequences.json
  - cache/pdb_files/ (for structural locality)
  - cache/residue_features.csv (for interpretability)
"""

import argparse
import gc
import json
import time
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from extract_embeddings import (
    extract_esm2_embeddings,
    extract_protgpt2_embeddings,
    extract_prott5_encoder_embeddings,
    extract_prott5_decoder_embeddings,
)
from train_sae import train_sae, extract_sae_features
from sae import SparseAutoencoder

warnings.filterwarnings("ignore")


# =====================================================================
#                    CONFIGURATION
# =====================================================================

# 50% depth layers for each model
MIDPOINT_LAYERS = {
    "esm2": 16,
    "protgpt2": 18,
    "prott5_enc": 12,
    "prott5_dec": 12,
}

EXTRACTORS = {
    "esm2": extract_esm2_embeddings,
    "protgpt2": extract_protgpt2_embeddings,
    "prott5_enc": extract_prott5_encoder_embeddings,
    "prott5_dec": extract_prott5_decoder_embeddings,
}

# Depth interpolation layers for ESM-2 (every 4th layer, 0-16)
ESM2_DEPTH_LAYERS = [0, 4, 8, 12, 16]

# SAE hyperparameters (matching main analysis)
EXPANSION = 8
K_SPARSE = 1024
N_SEEDS = 3


# =====================================================================
#     FEATURE DICTIONARY SIMILARITY
# =====================================================================

def compute_dictionary_similarity(D1: np.ndarray, D2: np.ndarray) -> dict:
    """Compute similarity between two SAE decoder dictionaries.

    For each feature in D1, find the best-matching feature in D2 (highest
    cosine similarity), and vice versa. Report mean best-match similarity
    in both directions plus the symmetric mean.

    Args:
        D1: (F, d_model) decoder weights from seed 1
        D2: (F, d_model) decoder weights from seed 2

    Returns:
        dict with mean_d1_to_d2, mean_d2_to_d1, mean_symmetric, std_symmetric
    """
    # L2-normalize rows
    D1_norm = D1 / (np.linalg.norm(D1, axis=1, keepdims=True) + 1e-8)
    D2_norm = D2 / (np.linalg.norm(D2, axis=1, keepdims=True) + 1e-8)

    # Cosine similarity matrix: (F1, F2)
    # Process in chunks to avoid OOM for large dictionaries
    F1, F2 = D1_norm.shape[0], D2_norm.shape[0]
    chunk = 1024

    # D1 → D2: best match for each D1 feature
    best_d1_to_d2 = np.zeros(F1, dtype=np.float32)
    for i in range(0, F1, chunk):
        end = min(i + chunk, F1)
        sim = D1_norm[i:end] @ D2_norm.T  # (chunk, F2)
        best_d1_to_d2[i:end] = sim.max(axis=1)

    # D2 → D1: best match for each D2 feature
    best_d2_to_d1 = np.zeros(F2, dtype=np.float32)
    for i in range(0, F2, chunk):
        end = min(i + chunk, F2)
        sim = D2_norm[i:end] @ D1_norm.T  # (chunk, F1)
        best_d2_to_d1[i:end] = sim.max(axis=1)

    all_best = np.concatenate([best_d1_to_d2, best_d2_to_d1])

    return {
        "mean_d1_to_d2": float(best_d1_to_d2.mean()),
        "mean_d2_to_d1": float(best_d2_to_d1.mean()),
        "mean_symmetric": float(all_best.mean()),
        "std_symmetric": float(all_best.std()),
        "median_symmetric": float(np.median(all_best)),
        "pct_above_0.9": float(100 * (all_best > 0.9).mean()),
        "pct_above_0.8": float(100 * (all_best > 0.8).mean()),
    }


# =====================================================================
#     FAST STRUCTURAL LOCALITY (inline, no cpu_stage dependency)
# =====================================================================

def fast_struct_locality(Z: np.ndarray, uids: list, sequences: dict,
                          pdb_dir: Path, topk_frac: float = 0.10,
                          n_shuffles: int = 3) -> float:
    """Quick mean structural Δ for a set of SAE activations.

    Simplified version of cpu_stage.analyze_struct_seq for speed.
    Returns mean structural Δ across all features.
    """
    try:
        from cpu_stage import (
            build_neighbor_graphs_residue_parallel,
            adj_list_to_sparse,
            build_protein_permutations,
            _cohens_d_vectorized,
        )
    except ImportError:
        print("  WARNING: cpu_stage not available, returning 0")
        return 0.0

    res_lengths = np.array([len(sequences[uid]) for uid in uids], dtype=np.int32)
    n_res = int(res_lengths.sum())

    if Z.shape[0] != n_res:
        print(f"  WARNING: Z rows ({Z.shape[0]}) != residues ({n_res})")
        return 0.0

    # Build structural neighbor graph
    seq_adj, struct_adj = build_neighbor_graphs_residue_parallel(
        uids, res_lengths, sequences, pdb_dir, n_jobs=-1)

    from scipy import sparse as sp
    A_struct, deg_struct = adj_list_to_sparse(struct_adj, n_res)

    # Permutations
    perm_indices = build_protein_permutations(res_lengths, n_shuffles)

    # Compute Cohen's d for all features
    n_features = Z.shape[1]
    acts = np.asarray(Z, dtype=np.float32)
    gstds = np.std(acts, axis=0).astype(np.float32)

    d_obs = _cohens_d_vectorized(acts, A_struct, deg_struct, gstds, topk_frac)

    # Shuffled baseline
    d_sh = np.zeros(n_features, dtype=np.float32)
    for perm in perm_indices:
        d_sh += _cohens_d_vectorized(acts[perm], A_struct, deg_struct, gstds, topk_frac)
    d_sh /= max(len(perm_indices), 1)

    delta = d_obs - d_sh
    return float(delta.mean())


# =====================================================================
#     A. SEED REPLICATION
# =====================================================================

def run_seed_replication(sequences: List[str], uids: List[str],
                          device: str, save_dir: Path,
                          pdb_dir: Path = None, models: List[str] = None):
    """Train N_SEEDS SAEs at 50% depth for each model, compare dictionaries."""
    save_dir = Path(save_dir) / "seed_replication"
    save_dir.mkdir(parents=True, exist_ok=True)

    if models is None:
        models = list(MIDPOINT_LAYERS.keys())

    ref_seqs = {uid: seq for uid, seq in zip(uids, sequences)}
    all_results = []

    for model_key in models:
        layer = MIDPOINT_LAYERS[model_key]
        extractor = EXTRACTORS[model_key]

        print(f"\n{'=' * 60}")
        print(f"  {model_key.upper()} — Layer {layer} (50% depth)")
        print(f"{'=' * 60}")

        # Extract embeddings once
        print(f"  Extracting embeddings...")
        emb_dict = extractor(sequences, layers=[layer], device=device)
        X = emb_dict[layer]
        D_dim = X.shape[1]
        print(f"  Embeddings: {X.shape}")

        # Auto-scale epochs
        T = X.shape[0]
        if T < 20_000:
            epochs = 20
        elif T < 100_000:
            epochs = 40
        elif T < 300_000:
            epochs = 60
        else:
            epochs = 80

        # Train N_SEEDS SAEs
        decoders = []
        for seed in range(N_SEEDS):
            print(f"\n  --- Seed {seed} ---")
            torch.manual_seed(seed * 1000 + 42)
            np.random.seed(seed * 1000 + 42)

            sae = train_sae(
                X, input_dim=D_dim, device=device,
                epochs=epochs, lr=5e-5,
                expansion=EXPANSION, k_sparse=K_SPARSE,
            )

            # Extract decoder dictionary
            base = sae._orig_mod if hasattr(sae, "_orig_mod") else sae
            D = base.decoder.weight.detach().cpu().numpy().T  # (hidden, d_model)
            decoders.append(D.astype(np.float32))

            # Save
            seed_dir = save_dir / f"{model_key}_seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            np.save(seed_dir / "D.npy", D.astype(np.float16))
            torch.save(base.state_dict(), seed_dir / "sae_model.pt")

            del sae
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

        # Pairwise dictionary similarity
        print(f"\n  Computing pairwise dictionary similarities...")
        pair_results = []
        for i in range(N_SEEDS):
            for j in range(i + 1, N_SEEDS):
                sim = compute_dictionary_similarity(decoders[i], decoders[j])
                sim["model"] = model_key
                sim["seed_a"] = i
                sim["seed_b"] = j
                pair_results.append(sim)
                print(f"    Seed {i} ↔ {j}: "
                      f"mean cosine = {sim['mean_symmetric']:.4f} "
                      f"(>0.9: {sim['pct_above_0.9']:.1f}%)")

        all_results.extend(pair_results)

        # Clean up
        del emb_dict, X, decoders
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    # Save results
    df = pd.DataFrame(all_results)
    df.to_csv(save_dir / "seed_similarity.csv", index=False)

    # Summary
    print(f"\n{'=' * 60}")
    print("  SEED REPLICATION SUMMARY")
    print(f"{'=' * 60}")
    for model_key in models:
        sub = df[df["model"] == model_key]
        if len(sub) == 0:
            continue
        mean_sim = sub["mean_symmetric"].mean()
        print(f"  {model_key:15s}: mean cosine = {mean_sim:.4f} "
              f"(>0.9: {sub['pct_above_0.9'].mean():.1f}%)")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    model_names = df["model"].unique()
    x = np.arange(len(model_names))
    means = [df[df["model"] == m]["mean_symmetric"].mean() for m in model_names]
    stds = [df[df["model"] == m]["mean_symmetric"].std() for m in model_names]

    colors = {"esm2": "#1f77b4", "protgpt2": "#ff7f0e",
              "prott5_enc": "#2ca02c", "prott5_dec": "#d62728"}
    bar_colors = [colors.get(m, "grey") for m in model_names]

    ax.bar(x, means, yerr=stds, capsize=5, color=bar_colors, alpha=0.7,
           edgecolor="black", linewidth=0.5)
    ax.axhline(0.85, color="green", ls="--", lw=1.5, label="Stability threshold (0.85)")
    ax.set_xticks(x)
    display = {"esm2": "ESM-2", "protgpt2": "ProtGPT2",
               "prott5_enc": "ProtT5-enc", "prott5_dec": "ProtT5-dec"}
    ax.set_xticklabels([display.get(m, m) for m in model_names])
    ax.set_ylabel("Mean Best-Match Cosine Similarity")
    ax.set_title(f"SAE Feature Dictionary Stability ({N_SEEDS} Seeds, 50% Depth)")
    ax.set_ylim(0.5, 1.05)
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(save_dir / "seed_stability.png", dpi=300)
    fig.savefig(save_dir / "seed_stability.pdf")
    plt.close(fig)

    return df


# =====================================================================
#     B. DEPTH INTERPOLATION
# =====================================================================

def run_depth_interpolation(sequences: List[str], uids: List[str],
                             device: str, save_dir: Path,
                             pdb_dir: Path = None):
    """Train SAEs at ESM-2 layers 0,4,8,12,16 to find the inflection point."""
    save_dir = Path(save_dir) / "depth_interpolation"
    save_dir.mkdir(parents=True, exist_ok=True)

    ref_seqs = {uid: seq for uid, seq in zip(uids, sequences)}

    print(f"\n{'=' * 60}")
    print(f"  ESM-2 DEPTH INTERPOLATION")
    print(f"  Layers: {ESM2_DEPTH_LAYERS}")
    print(f"{'=' * 60}")

    # Extract embeddings at all target layers
    print(f"\n  Extracting embeddings at layers {ESM2_DEPTH_LAYERS}...")
    emb_dict = extract_esm2_embeddings(
        sequences, layers=ESM2_DEPTH_LAYERS, device=device)

    results = []

    for layer in ESM2_DEPTH_LAYERS:
        print(f"\n{'─' * 50}")
        print(f"  Layer {layer}")
        print(f"{'─' * 50}")

        X = emb_dict[layer]
        D_dim = X.shape[1]
        T = X.shape[0]

        # Auto epochs
        if T < 20_000:
            epochs = 20
        elif T < 100_000:
            epochs = 40
        elif T < 300_000:
            epochs = 60
        else:
            epochs = 80

        # Train SAE
        sae = train_sae(
            X, input_dim=D_dim, device=device,
            epochs=epochs, lr=5e-5,
            expansion=EXPANSION, k_sparse=K_SPARSE,
        )

        # Extract features
        layer_dir = save_dir / f"layer_{layer}"
        layer_dir.mkdir(parents=True, exist_ok=True)

        Z, D = extract_sae_features(sae, X, device=device,
                                     save_dir=str(layer_dir))

        # Compute structural locality (if pdb_dir available)
        struct_delta = 0.0
        if pdb_dir and Path(pdb_dir).exists():
            print(f"  Computing structural locality...")
            struct_delta = fast_struct_locality(
                Z, uids, ref_seqs, Path(pdb_dir))
            print(f"  Structural Δ: {struct_delta:.4f}")

        # Compute interpretability (quick: fraction of features with |r| > threshold)
        # We just check sparsity metrics as a proxy
        l0 = (Z > 0).sum(axis=1).mean()
        pct_dead = 100 * ((Z > 0).sum(axis=0) == 0).mean()

        results.append({
            "layer": layer,
            "relative_depth": layer / 32.0,
            "struct_delta": struct_delta,
            "mean_l0": float(l0),
            "pct_dead": float(pct_dead),
            "n_tokens": T,
            "embed_dim": D_dim,
        })

        print(f"  L0: {l0:.1f}, Dead: {pct_dead:.1f}%")

        # Save metadata
        (layer_dir / "META.json").write_text(json.dumps({
            "model": "ESM-2",
            "layer": layer,
            "embed_dim": D_dim,
            "expansion": EXPANSION,
            "k_sparse": K_SPARSE,
            "struct_delta": struct_delta,
        }, indent=2))

        del sae, Z, D
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    del emb_dict
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(save_dir / "depth_interpolation_results.csv", index=False)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Structural locality vs layer
    ax = axes[0]
    layers = df["layer"].values
    deltas = df["struct_delta"].values
    ax.plot(layers, deltas, "o-", color="#1f77b4", lw=2.5, markersize=10)
    ax.axhline(0, color="grey", ls="--", lw=1)
    ax.set_xlabel("ESM-2 Layer", fontsize=12)
    ax.set_ylabel("Structural Δ (mean)", fontsize=12)
    ax.set_title("Depth Interpolation: Structural Locality", fontsize=13)
    ax.set_xticks(layers)
    ax.grid(alpha=0.3)

    # Annotate the drop
    if len(deltas) >= 2:
        max_drop_idx = np.argmax(np.abs(np.diff(deltas)))
        l1, l2 = layers[max_drop_idx], layers[max_drop_idx + 1]
        d1, d2 = deltas[max_drop_idx], deltas[max_drop_idx + 1]
        ax.annotate(
            f"Largest change:\nL{l1}→L{l2}",
            xy=((l1 + l2) / 2, (d1 + d2) / 2),
            xytext=(l2 + 1, max(deltas) * 0.8),
            arrowprops=dict(arrowstyle="->", color="red"),
            fontsize=10, color="red",
        )

    # Panel B: L0 and dead features vs layer
    ax = axes[1]
    ax2 = ax.twinx()
    ax.plot(layers, df["mean_l0"].values, "s-", color="#2ca02c", lw=2,
            markersize=8, label="Mean L0")
    ax2.plot(layers, df["pct_dead"].values, "^--", color="#d62728", lw=2,
             markersize=8, label="% Dead")
    ax.set_xlabel("ESM-2 Layer", fontsize=12)
    ax.set_ylabel("Mean L0 (active features/token)", fontsize=12, color="#2ca02c")
    ax2.set_ylabel("% Dead Features", fontsize=12, color="#d62728")
    ax.set_title("SAE Health Metrics Across Depth", fontsize=13)
    ax.set_xticks(layers)
    ax.grid(alpha=0.3)

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    fig.tight_layout()
    fig.savefig(save_dir / "depth_interpolation.png", dpi=300)
    fig.savefig(save_dir / "depth_interpolation.pdf")
    plt.close(fig)

    # Report
    print(f"\n{'=' * 60}")
    print("  DEPTH INTERPOLATION RESULTS")
    print(f"{'=' * 60}")
    print(df[["layer", "struct_delta", "mean_l0", "pct_dead"]].to_string(index=False))

    # Find inflection
    if len(deltas) >= 3:
        diffs = np.diff(deltas)
        inflection_idx = np.argmax(np.abs(diffs))
        print(f"\n  Largest change: layer {layers[inflection_idx]} → "
              f"layer {layers[inflection_idx + 1]}")
        print(f"    Δ change: {diffs[inflection_idx]:+.4f}")

    return df


# =====================================================================
#                           MAIN
# =====================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Seed replication & depth interpolation experiments")
    ap.add_argument("mode", choices=["seed", "depth", "both"],
                    help="Which experiment to run")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--save-dir", default="results_stability")
    ap.add_argument("--pdb-dir", default="cache/pdb_files")
    ap.add_argument("--sequences-json", default="cache/sequences.json")
    ap.add_argument("--models", default="all",
                    help="Comma-separated model keys for seed replication "
                         "(default: all)")
    args = ap.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    pdb_dir = Path(args.pdb_dir)

    # Load sequences
    seq_path = Path(args.sequences_json)
    if not seq_path.exists():
        raise FileNotFoundError(f"Missing {seq_path}")

    data = json.loads(seq_path.read_text())
    uids = sorted(data.keys())
    sequences = [data[uid] for uid in uids]
    print(f"  Loaded {len(sequences)} proteins")

    if args.mode in ("seed", "both"):
        models = None
        if args.models != "all":
            models = [m.strip() for m in args.models.split(",")]
        run_seed_replication(sequences, uids, args.device, save_dir,
                              pdb_dir=pdb_dir, models=models)

    if args.mode in ("depth", "both"):
        run_depth_interpolation(sequences, uids, args.device, save_dir,
                                 pdb_dir=pdb_dir)

    print(f"\n  All results saved to {save_dir}/")


if __name__ == "__main__":
    main()
