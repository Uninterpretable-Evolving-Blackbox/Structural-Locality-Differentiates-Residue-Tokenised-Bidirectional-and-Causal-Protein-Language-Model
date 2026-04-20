#!/usr/bin/env python3
"""
run_unsupervised.py — GPU stage for protein SAE pipeline
=========================================================

Extracts PLM embeddings at matched relative depths, trains SAEs,
and exports Z.npy / D.npy for downstream CPU analysis.

Supports CUDA, MPS (Apple Silicon), and CPU backends.

Usage:
    DEVICE=mps  MODEL=esm2      python run_unsupervised.py
    DEVICE=cuda MODEL=all        python run_unsupervised.py
    DEVICE=cpu  MODEL=protgpt2   python run_unsupervised.py

Environment variables:
    DEVICE : cuda | mps | cpu  (auto-detected if omitted)
    MODEL  : esm2 | protgpt2 | prott5_enc | prott5_dec | all
"""

from extract_embeddings import (
    extract_esm2_embeddings,
    extract_protgpt2_embeddings,
    extract_prott5_encoder_embeddings,
    extract_prott5_decoder_embeddings,
)
from train_sae import (
    train_sae,
    extract_sae_features,
    compute_explained_variance,
    compute_norm_scale,
)
import matplotlib
matplotlib.use("Agg")
import torch
import os
import gc
import json
import numpy as np
from pathlib import Path
from typing import List, Tuple


# ============================================================
#                    CONFIGURATION
# ============================================================

cache_dir = Path("cache")

# Suffix lets us run multiple variants without clobbering each other.
# Examples:
#   RUN_SUFFIX=""        → outputs_layerwise/        (default seed=42, k=256)
#   RUN_SUFFIX="_seed43" → outputs_layerwise_seed43/ (different SAE init seed)
#   RUN_SUFFIX="_k128"   → outputs_layerwise_k128/   (different k_sparse)
#   RUN_SUFFIX="_n3000"  → outputs_layerwise_n3000/  (3000-protein dataset)
RUN_SUFFIX = os.environ.get("RUN_SUFFIX", "")
output_root = Path(f"outputs_layerwise{RUN_SUFFIX}")

# SAE hyperparameters (TopK SAE, Gao et al. 2024)
EXPANSION = int(os.environ.get("EXPANSION", 8))
# k_sparse selected from ESM-2 layer-16 ablation.  Override via K_SPARSE env var.
K_SPARSE = int(os.environ.get("K_SPARSE", 256))   # ~2.5% density at hidden=10240
K_AUX = 64           # auxiliary loss for dead-latent recovery
# Trigger AuxK ~every 4 epochs at 268k train tokens — fast enough that
# resurrection happens early in training, not after the LR has decayed.
DEAD_THRESHOLD = 1_000_000

# Train/Val split (protein-level).  Default is FIXED across all models and
# SAE_SEED values, so by default only SAE training stochasticity varies and
# all 4 models hold out the same 150 proteins.  Override SPLIT_SEED via env
# var to probe robustness to protein subset selection; RUN_SUFFIX MUST be
# set to a distinct value (e.g. _split99) or outputs will clobber the main
# seed=42 run.
SPLIT_SEED = int(os.environ.get("SPLIT_SEED", 42))
VAL_FRACTION = 0.10

# SAE training random seed — varies with SAE_SEED env var.  Default 42 is
# what produced the original outputs_layerwise/ run.
SAE_SEED = int(os.environ.get("SAE_SEED", 42))


def _auto_device() -> str:
    """Auto-detect best available device."""
    env = os.environ.get("DEVICE", "").lower()
    if env:
        return env
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ============================================================
#                  DATASET LOADING
# ============================================================

def load_dataset_from_json(path: Path) -> Tuple[List[str], List[str]]:
    """Load sequences and IDs from JSON."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run build_dataset.py first to create the sequence cache."
        )

    print(f"📥 Loading dataset from {path}...")
    data = json.loads(path.read_text())

    if isinstance(data, dict):
        uids_kept = sorted(data.keys())
        protein_sequences = [data[uid] for uid in uids_kept]
    elif isinstance(data, list):
        # Handle list-of-dicts format [{uid, sequence}, ...]
        uids_kept = [d["uid"] for d in data]
        protein_sequences = [d["sequence"] for d in data]
    else:
        raise ValueError(f"Unsupported sequences.json format: {type(data)}")

    return uids_kept, protein_sequences


# ============================================================
#              MODEL CONFIGURATION
# ============================================================

# Relative depth matching across architectures:
# | Depth | ESM-2 (33) | ProtGPT2 (36) | ProtT5 (24) |
# |-------|------------|---------------|-------------|
# | ~0%   | 0          | 0             | 0           |
# | ~25%  | 8          | 9             | 6           |
# | ~50%  | 16         | 18            | 12          |
# | ~75%  | 24         | 27            | 18          |
# | 100%  | 32         | 35            | 23          |

MODEL_PLANS = {
    "esm2": (
        "ESM-2 t33 (650M)",
        extract_esm2_embeddings,
        [0, 8, 16, 24, 32],
    ),
    "protgpt2": (
        "ProtGPT2 (~738M)",
        extract_protgpt2_embeddings,
        [0, 9, 18, 27, 35],
    ),
    "prott5_enc": (
        "ProtT5 encoder",
        extract_prott5_encoder_embeddings,
        [0, 6, 12, 18, 23],
    ),
    "prott5_dec": (
        "ProtT5 decoder",
        extract_prott5_decoder_embeddings,
        [0, 6, 12, 18, 23],
    ),
}


def auto_epochs(total_tokens: int) -> int:
    """Scale training epochs by dataset size."""
    if total_tokens < 20_000:   return 20
    if total_tokens < 100_000:  return 40
    if total_tokens < 300_000:  return 60
    if total_tokens < 800_000:  return 80
    return 100


def make_protein_split(n_proteins: int, val_fraction: float, seed: int):
    """
    Deterministic protein-level train/val split.

    Returns sorted (train_idx, val_idx) lists into the protein order.
    Seeded so that all four models hold out the SAME proteins, making
    cross-architecture holdout-EV comparisons valid.
    """
    n_val = max(1, int(round(n_proteins * val_fraction)))
    rng = np.random.RandomState(seed)
    val_idx = sorted(rng.choice(n_proteins, n_val, replace=False).tolist())
    val_set = set(val_idx)
    train_idx = [i for i in range(n_proteins) if i not in val_set]
    return train_idx, val_idx


def rows_for_proteins(token_offsets: np.ndarray, protein_indices: list) -> np.ndarray:
    """Concatenate token-row indices for a list of protein indices."""
    if not protein_indices:
        return np.array([], dtype=np.int64)
    return np.concatenate([
        np.arange(token_offsets[i], token_offsets[i + 1], dtype=np.int64)
        for i in protein_indices
    ])


def compute_protgpt2_token_lengths(protein_sequences: List[str]) -> np.ndarray:
    """
    Re-tokenize sequences with the ProtGPT2 BPE tokenizer to recover the
    per-protein token count. Needed only because the embedding extractor
    discards this. Run once per pipeline; cheap (<10s).
    """
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("nferruz/ProtGPT2", use_fast=True)
    counts = np.zeros(len(protein_sequences), dtype=np.int64)
    for i, seq in enumerate(protein_sequences):
        ids = tok(seq, add_special_tokens=False, truncation=True, max_length=1024)["input_ids"]
        counts[i] = len(ids)
    return counts


# ============================================================
#                      MAIN
# ============================================================

def main():
    device = _auto_device()
    # MPS works correctly for SAE training after the sae.py _init_weights
    # fix (replaced `weight.data = ...` with `weight.copy_(...)`).  Use
    # the auto-detected device for both PLM extraction and SAE training.
    sae_device = device
    model_key = os.environ.get("MODEL", "all").lower()

    cache_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    # Load dataset
    seq_json = cache_dir / "sequences.json"
    uids_kept, protein_sequences = load_dataset_from_json(seq_json)

    lengths = [len(s) for s in protein_sequences]
    offsets = np.cumsum([0] + lengths)
    total_tokens = int(offsets[-1])

    print(f"✅ Loaded {len(protein_sequences)} proteins ({total_tokens} residues)")

    # Select models
    if model_key == "all":
        selected_models = list(MODEL_PLANS.keys())
    elif model_key in MODEL_PLANS:
        selected_models = [model_key]
    else:
        raise ValueError(f"Unknown MODEL={model_key}. Use: {list(MODEL_PLANS.keys())} or 'all'")

    # Deterministic protein-level train/val split — fixed across all 4 models
    train_protein_idx, val_protein_idx = make_protein_split(
        n_proteins=len(protein_sequences),
        val_fraction=VAL_FRACTION,
        seed=SPLIT_SEED,
    )
    val_uids = [uids_kept[i] for i in val_protein_idx]
    train_residues = int(sum(lengths[i] for i in train_protein_idx))
    val_residues = int(sum(lengths[i] for i in val_protein_idx))
    print(f"📊 Protein-level split (seed={SPLIT_SEED}, val_fraction={VAL_FRACTION}):")
    print(f"   Train: {len(train_protein_idx)} proteins / {train_residues} residues")
    print(f"   Val:   {len(val_protein_idx)} proteins / {val_residues} residues")

    base_epochs = auto_epochs(train_residues)
    print(f"🛠️  SAE epochs: {base_epochs} (based on {train_residues} TRAIN tokens)")
    print(f"🖥️  Device: {device}")

    # Process each model
    for mk in selected_models:
        model_name, extractor_fn, layers_to_test = MODEL_PLANS[mk]

        print("\n" + "#" * 78)
        print(f"###  Model: {model_name}")
        print(f"###  Layers: {layers_to_test}")
        print(f"###  Device: {device}")
        print("#" * 78)

        model_out_root = output_root / mk
        model_out_root.mkdir(parents=True, exist_ok=True)

        # Resume support: if every layer for this model already has a per-layer
        # META.json, skip the model entirely — no PLM forward pass, no SAE
        # training.  This makes ./run_all.sh safe to re-run after a crash.
        all_done = all(
            (model_out_root / f"layer_{layer}" / "META.json").exists()
            for layer in layers_to_test
        )
        if all_done:
            print(f"   ✓ All {len(layers_to_test)} layers already complete — skipping {mk}")
            continue

        # Save metadata
        (model_out_root / "META.json").write_text(json.dumps({
            "model": model_name,
            "device": device,
            "layers": layers_to_test,
            "num_proteins": len(protein_sequences),
            "total_residues": total_tokens,
            "expansion": EXPANSION,
            "k_sparse": K_SPARSE,
            "k_aux": K_AUX,
            "dead_threshold": DEAD_THRESHOLD,
            "sae_seed": SAE_SEED,
            "run_suffix": RUN_SUFFIX,
            "split_seed": SPLIT_SEED,
            "val_fraction": VAL_FRACTION,
            "n_train_proteins": len(train_protein_idx),
            "n_val_proteins": len(val_protein_idx),
            "val_uids": val_uids,
        }, indent=2))

        # Extract embeddings for all layers
        print("\n🧮 Extracting embeddings for all layers...")
        embeddings_by_layer = extractor_fn(
            protein_sequences,
            layers=layers_to_test,
            device=device,
        )

        if not isinstance(embeddings_by_layer, dict):
            raise RuntimeError("Extractor must return dict[layer] -> embeddings")

        # Build per-protein TOKEN offsets for this model.
        #   - Residue models: 1 token per residue → use the residue offsets.
        #   - ProtGPT2 (BPE): re-tokenize each protein to recover token counts.
        if mk == "protgpt2":
            print("   Computing ProtGPT2 BPE token lengths for protein-level split...")
            tok_lengths = compute_protgpt2_token_lengths(protein_sequences)
            tok_offsets = np.cumsum([0] + tok_lengths.tolist()).astype(np.int64)
        else:
            tok_lengths = np.asarray(lengths, dtype=np.int64)
            tok_offsets = offsets.astype(np.int64)

        # Process each layer
        for layer in layers_to_test:
            print("\n" + "=" * 70)
            print(f"🔹 Processing layer {layer}")
            print("=" * 70)

            # Per-layer skip for resume after a mid-model crash
            layer_meta = model_out_root / f"layer_{layer}" / "META.json"
            if layer_meta.exists():
                print(f"   ✓ {layer_meta} already exists — skipping layer")
                # Still free this layer's embedding to prevent memory leak
                if layer in embeddings_by_layer:
                    embeddings_by_layer[layer] = None
                    del embeddings_by_layer[layer]
                continue

            if layer not in embeddings_by_layer:
                raise KeyError(f"Missing embeddings for layer {layer}")

            X_tokens = embeddings_by_layer[layer]

            if X_tokens.ndim != 2:
                raise RuntimeError(f"Expected 2D array, got {X_tokens.shape}")

            T, D = X_tokens.shape

            # Validate residue alignment for encoder models
            if mk in ("esm2", "prott5_enc", "prott5_dec"):
                if T != total_tokens:
                    raise RuntimeError(
                        f"{mk}: tokens ({T}) != residues ({total_tokens}). "
                        "Check special-token trimming."
                    )
            else:
                # ProtGPT2: validate against re-tokenized BPE token counts
                if T != int(tok_offsets[-1]):
                    raise RuntimeError(
                        f"{mk}: extracted tokens ({T}) != re-tokenized total "
                        f"({int(tok_offsets[-1])}). Tokenizer or truncation mismatch."
                    )
                print(f"⚠️ {mk}: BPE tokens ({T}) != residues ({total_tokens})")

            print(f"   Embeddings: {T} × {D}")

            # Protein-level split — same proteins held out across all models
            train_rows = rows_for_proteins(tok_offsets, train_protein_idx)
            val_rows = rows_for_proteins(tok_offsets, val_protein_idx)
            X_train = np.ascontiguousarray(X_tokens[train_rows])
            X_val = np.ascontiguousarray(X_tokens[val_rows])
            print(f"   Split: train {X_train.shape[0]} / val {X_val.shape[0]} tokens")

            # Bricken/Anthropic input normalization: rescale every token so the
            # mean L2 norm becomes √D.  Without this, ESM-2 / ProtGPT2 / ProtT5
            # outlier features (mid-layer dimensions with magnitudes ~50-100×
            # the typical scale, see Dettmers et al. 2022) destabilize TopK SAE
            # training, producing the negative-EV / loss-U-curve failure mode.
            # Compute scale on TRAIN only, apply to both, save in META.
            norm_scale = compute_norm_scale(X_train)
            X_train = (X_train * norm_scale).astype(np.float32)
            X_val = (X_val * norm_scale).astype(np.float32)
            print(f"   Norm scale: {norm_scale:.6f}  "
                  f"(post-scale mean L2 = {float(np.linalg.norm(X_train, axis=1).mean()):.2f}, "
                  f"target √D = {np.sqrt(D):.2f})")

            # Train SAE on TRAIN tokens only.
            # Seed torch BEFORE every SAE init so seeds are reproducible
            # per-(model,layer,SAE_SEED) instead of drifting with whatever
            # has happened in the process so far.
            print(f"\n🏋️ Training SAE on {sae_device} "
                  f"(seed={SAE_SEED}, expansion={EXPANSION}, k={K_SPARSE}, k_aux={K_AUX})...")
            torch.manual_seed(SAE_SEED)
            np.random.seed(SAE_SEED)
            sae_model = train_sae(
                X_train,
                input_dim=D,
                device=sae_device,
                epochs=base_epochs,
                lr=5e-5,
                expansion=EXPANSION,
                k_sparse=K_SPARSE,
                k_aux=K_AUX,
                dead_threshold=DEAD_THRESHOLD,
            )

            # Holdout vs train explained variance — the generalization check
            print(f"\n📐 Computing explained variance...")
            train_ev = float(compute_explained_variance(sae_model, X_train, device=sae_device))
            val_ev = float(compute_explained_variance(sae_model, X_val, device=sae_device))
            ev_gap = train_ev - val_ev
            warn = "  ⚠️ MEMORIZATION SUSPECTED" if ev_gap > 0.10 else ""
            print(f"   Train EV: {train_ev:.4f}")
            print(f"   Val EV:   {val_ev:.4f}{warn}")
            print(f"   Gap:      {ev_gap:+.4f}")

            # Extract features
            save_dir = model_out_root / f"layer_{layer}"
            save_dir.mkdir(parents=True, exist_ok=True)

            # Features for ALL tokens (train + val) so cpu_stage can analyse
            # every protein. The SAE was trained on normalized train tokens,
            # so we MUST apply the same normalization scale here. Otherwise
            # the encoder is fed inputs in a different range than it was
            # trained on and the Z values are meaningless.
            # Feature extraction also runs on CPU (forward only) to match
            # the training device.
            print(f"\n💾 Extracting features (full dataset, train+val, normalized)...")
            X_tokens_normalized = (X_tokens * norm_scale).astype(np.float32)
            Z_tokens, D_dict = extract_sae_features(
                sae_model,
                X_tokens_normalized,
                device=sae_device,
                save_dir=str(save_dir),
            )
            del X_tokens_normalized

            sparsity = (Z_tokens == 0).mean() * 100
            print(f"   Z: {Z_tokens.shape}, sparsity: {sparsity:.1f}%")

            # Save metadata for CPU stage.
            # CRITICAL: must save TOKEN-level lengths/offsets (matching Z rows),
            # not residue-level.  For ESM-2 / ProtT5 these are identical
            # (1 token == 1 residue), but for ProtGPT2 they differ (BPE tokens
            # cover ~3 residues each).  cpu_stage's load_layer asserts
            # sum(lengths.npy) == Z.shape[0], which trips for ProtGPT2 if we
            # save residue counts here.
            np.save(save_dir / "lengths.npy", np.asarray(tok_lengths, dtype=np.int32))
            np.save(save_dir / "offsets.npy", np.asarray(tok_offsets, dtype=np.int64))

            with open(save_dir / "sequences.json", "w") as f:
                json.dump(protein_sequences, f)
            with open(save_dir / "uids.json", "w") as f:
                json.dump(uids_kept, f)

            (save_dir / "META.json").write_text(json.dumps({
                "model": model_name,
                "layer": layer,
                "embed_dim": D,
                "sae_hidden_dim": D * EXPANSION,
                "k_sparse": K_SPARSE,
                "k_aux": K_AUX,
                "dead_threshold": DEAD_THRESHOLD,
                "num_proteins": len(protein_sequences),
                "total_tokens": T,
                "total_residues": total_tokens,
                "z_shape": list(Z_tokens.shape),
                # Bricken normalization scale (cpu_stage doesn't need this
                # since Z values are produced from normalized inputs, but
                # we save it for full reproducibility)
                "norm_scale": norm_scale,
                # Holdout metrics — load-bearing for the paper
                "sae_seed": SAE_SEED,
                "run_suffix": RUN_SUFFIX,
                "split_seed": SPLIT_SEED,
                "val_fraction": VAL_FRACTION,
                "n_train_proteins": len(train_protein_idx),
                "n_val_proteins": len(val_protein_idx),
                "val_uids": val_uids,
                "train_explained_variance": train_ev,
                "val_explained_variance": val_ev,
                "ev_gap": ev_gap,
            }, indent=2))

            del X_train, X_val

            # Clear memory — drop this layer's embedding tensor too, not
            # just the SAE.  The full embeddings dict can be 5–60 GB
            # depending on dataset size and PLM dim, and we don't need
            # this layer's chunk again.
            del sae_model, Z_tokens, D_dict
            embeddings_by_layer[layer] = None
            del embeddings_by_layer[layer]
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

        # Clear embeddings
        del embeddings_by_layer
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("✅ GPU stage complete! Ready for CPU pipeline.")
    print("=" * 70)


if __name__ == "__main__":
    main()
