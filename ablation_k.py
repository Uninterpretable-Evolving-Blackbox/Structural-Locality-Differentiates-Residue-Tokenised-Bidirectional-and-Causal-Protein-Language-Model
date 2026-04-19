#!/usr/bin/env python3
"""
ablation_k.py — k_sparse ablation on ESM-2 layer 16
====================================================

Purpose
-------
Pick / defend the SAE sparsity hyperparameter (k_sparse) *a priori*, before
launching the main 5–8 hour pipeline. Trains 3 SAEs at k_sparse ∈ {64, 128,
256} on ESM-2 layer-16 activations, using the SAME deterministic protein-level
train/val split as the main pipeline (seed 42, val_fraction 0.10) so the
ablation's held-out proteins are identical to the main run's.

Why ESM-2 layer 16
------------------
Middle layer (50% relative depth) of ESM-2 — the canonical "feature-rich"
layer in the protein language modelling literature, and the layer where the
sparsity / quality tradeoff is most informative for selecting k.

Outputs
-------
  analysis_results/ablation_k_sparse.csv
  analysis_results/ablation_k_sparse.png  (and .pdf)

Compute time
------------
~30–60 min on a single A100/H100 (mostly the one ESM-2 forward pass).
Slower on MPS / CPU. ESM-2 is loaded once; the three SAE training runs
share the same cached embedding matrix.

Usage
-----
    python ablation_k.py            # auto-detect device
    DEVICE=cuda python ablation_k.py
    DEVICE=mps  python ablation_k.py
"""

import os
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch

from extract_embeddings import extract_esm2_embeddings
from train_sae import train_sae, compute_explained_variance, compute_norm_scale
from run_unsupervised import (
    load_dataset_from_json,
    make_protein_split,
    rows_for_proteins,
    auto_epochs,
    EXPANSION, K_AUX, DEAD_THRESHOLD,
    SPLIT_SEED, VAL_FRACTION,
)


# Cache the extracted ESM-2 layer-16 embeddings on disk so re-running
# the ablation (e.g. after a fix) skips the 76 s extraction step.
EMBED_CACHE = Path("cache") / "ablation_esm2_layer16.npy"


# ---------------- ablation config ----------------
ABLATION_MODEL = "esm2"
ABLATION_LAYER = 16             # ESM-2 middle layer (50% relative depth)
K_VALUES = [64, 128, 256]
OUT_DIR = Path("analysis_results")


def _auto_device() -> str:
    env = os.environ.get("DEVICE", "").lower()
    if env:
        return env
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    device = _auto_device()
    # MPS now works for SAE training after the sae.py _init_weights fix
    # (replaced `weight.data = ...` with `weight.copy_(...)`; the .data
    # reassignment was corrupting Parameter bindings during .to('mps')
    # and producing wrong nn.Linear forward outputs even though weights
    # read back as bit-identical).  Both extraction and training run on
    # the same auto-detected device.
    sae_device = device
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 66)
    print(f"  k_sparse ablation — {ABLATION_MODEL} layer {ABLATION_LAYER}")
    print(f"  k values: {K_VALUES}")
    print(f"  expansion={EXPANSION}, k_aux={K_AUX}, dead_threshold={DEAD_THRESHOLD:,}")
    print(f"  Device: {device}  (PLM extraction + SAE training)")
    print("=" * 66)

    # ---- 1. Load the SAME dataset the main pipeline uses ----
    seq_json = Path("cache/sequences.json")
    uids, sequences = load_dataset_from_json(seq_json)
    n_proteins = len(sequences)
    lengths = np.array([len(s) for s in sequences], dtype=np.int64)
    offsets = np.cumsum([0] + lengths.tolist()).astype(np.int64)
    print(f"\n✅ Loaded {n_proteins} proteins / {int(offsets[-1])} residues")

    # ---- 2. Same seeded protein-level split → val set matches main run ----
    train_idx, val_idx = make_protein_split(n_proteins, VAL_FRACTION, SPLIT_SEED)
    print(f"📊 Protein split (seed={SPLIT_SEED}, val_fraction={VAL_FRACTION}):")
    print(f"   Train: {len(train_idx)} proteins")
    print(f"   Val:   {len(val_idx)} proteins  (identical to main pipeline's val set)")

    # ---- 3. Extract ESM-2 layer 16 (with disk cache) ----
    if EMBED_CACHE.exists():
        print(f"\n📥 Loading cached ESM-2 layer {ABLATION_LAYER} embeddings from {EMBED_CACHE}...")
        X_tokens = np.load(EMBED_CACHE)
        print(f"   Cache shape: {X_tokens.shape}")
    else:
        print(f"\n🧮 Extracting ESM-2 layer {ABLATION_LAYER} embeddings (one-time)...")
        emb = extract_esm2_embeddings(sequences, layers=[ABLATION_LAYER], device=device)
        X_tokens = emb[ABLATION_LAYER]
        del emb
        EMBED_CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.save(EMBED_CACHE, X_tokens)
        print(f"   Cached → {EMBED_CACHE}  ({X_tokens.nbytes/1e9:.2f} GB)")

    if X_tokens.shape[0] != int(offsets[-1]):
        raise RuntimeError(
            f"ESM-2 token count {X_tokens.shape[0]} != residue count {int(offsets[-1])}"
        )
    T, D = X_tokens.shape
    print(f"   Shape: {T} tokens × {D} dim")

    # ---- 3a.  Diagnostic stats so we can SEE what we're feeding the SAE ----
    print(f"   ESM-2 raw stats:")
    print(f"     per-element mean = {float(X_tokens.mean()):+.4f}")
    print(f"     per-element std  = {float(X_tokens.std()):.4f}")
    print(f"     min/max          = {float(X_tokens.min()):+.2f} / {float(X_tokens.max()):+.2f}")
    print(f"     mean L2(token)   = {float(np.linalg.norm(X_tokens, axis=1).mean()):.2f}")

    train_rows = rows_for_proteins(offsets, train_idx)
    val_rows = rows_for_proteins(offsets, val_idx)
    X_train = np.ascontiguousarray(X_tokens[train_rows])
    X_val = np.ascontiguousarray(X_tokens[val_rows])
    print(f"   train tokens: {X_train.shape[0]}  |  val tokens: {X_val.shape[0]}")
    del X_tokens
    gc.collect()

    # ---- 3b.  Bricken/Anthropic recipe: rescale so mean ||x||₂ = √D ----
    norm_scale = compute_norm_scale(X_train)
    X_train = (X_train * norm_scale).astype(np.float32)
    X_val = (X_val * norm_scale).astype(np.float32)
    print(f"\n📐 Input normalization (Bricken recipe):")
    print(f"   scale factor     = {norm_scale:.6f}")
    print(f"   post-scale mean  = {float(X_train.mean()):+.4f}")
    print(f"   post-scale std   = {float(X_train.std()):.4f}")
    print(f"   post-scale L2    = {float(np.linalg.norm(X_train, axis=1).mean()):.2f}  (target √D = {np.sqrt(D):.2f})")

    # Match main run's epoch budget (60 for ~241k train tokens)
    epochs = auto_epochs(X_train.shape[0])
    print(f"🛠️  Epochs (matching main run): {epochs}")

    # ---- 4. Train one SAE per k value ----
    rows = []
    for k in K_VALUES:
        print("\n" + "=" * 66)
        print(f"  Training k_sparse = {k}")
        print("=" * 66)

        sae = train_sae(
            X_train,
            input_dim=D,
            device=sae_device,
            epochs=epochs,
            lr=5e-5,
            expansion=EXPANSION,
            k_sparse=k,
            k_aux=K_AUX,
            dead_threshold=DEAD_THRESHOLD,
        )

        # Honest train vs val EV (val proteins were never touched by training)
        train_ev = float(compute_explained_variance(sae, X_train, device=sae_device))
        val_ev = float(compute_explained_variance(sae, X_val, device=sae_device))
        gap = train_ev - val_ev

        # Dead latents (post-training, from the SAE's own tracking buffer)
        base = sae._orig_mod if hasattr(sae, "_orig_mod") else sae
        n_dead = int(base._get_dead_latent_mask().sum().item())
        pct_dead = 100.0 * n_dead / base.hidden_dim

        warn = "  ⚠ memorization suspected" if gap > 0.10 else ""
        print(f"\n   train_ev = {train_ev:.4f}")
        print(f"   val_ev   = {val_ev:.4f}{warn}")
        print(f"   gap      = {gap:+.4f}")
        print(f"   dead     = {n_dead}/{base.hidden_dim} ({pct_dead:.1f}%)")

        rows.append({
            "k_sparse": k,
            "train_ev": train_ev,
            "val_ev": val_ev,
            "ev_gap": gap,
            "n_dead": n_dead,
            "pct_dead": pct_dead,
            "expansion": EXPANSION,
            "k_aux": K_AUX,
            "dead_threshold": DEAD_THRESHOLD,
            "epochs": epochs,
            "n_train_tokens": int(X_train.shape[0]),
            "n_val_tokens": int(X_val.shape[0]),
            "model": ABLATION_MODEL,
            "layer": ABLATION_LAYER,
        })

        del sae, base
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)

    # ---- 5. Save CSV ----
    csv_path = OUT_DIR / "ablation_k_sparse.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n💾 Saved {csv_path}")
    print("\n" + df.to_string(index=False))

    # ---- 6. Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ks = df["k_sparse"].values

    ax = axes[0]
    ax.plot(ks, df["train_ev"], "o-", color="#1f77b4", lw=2.2,
            markersize=11, label="Train EV")
    ax.plot(ks, df["val_ev"], "s--", color="#d62728", lw=2.2,
            markersize=11, label="Val EV (held-out proteins)")
    ax.set_xlabel("k_sparse  (active latents per token)", fontsize=11)
    ax.set_ylabel("Explained variance", fontsize=11)
    ax.set_title(f"ESM-2 layer {ABLATION_LAYER} — sparsity ablation", fontsize=12)
    ax.set_xticks(ks)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.bar([str(k) for k in ks], df["ev_gap"], color="#666666", alpha=0.85)
    ax.axhline(0.10, color="red", ls="--", lw=1.2, label="Memorization warning (>0.10)")
    ax.axhline(0.0, color="grey", lw=0.8)
    ax.set_xlabel("k_sparse", fontsize=11)
    ax.set_ylabel("Train EV  −  Val EV", fontsize=11)
    ax.set_title("Generalization gap", fontsize=12)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    png_path = OUT_DIR / "ablation_k_sparse.png"
    plt.savefig(png_path, dpi=220)
    plt.savefig(png_path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"💾 Saved {png_path}")

    # ---- 7. Recommendation summary ----
    print("\n" + "=" * 66)
    print("  How to read this:")
    print("=" * 66)
    print("  • Pick the k where val_ev plateaus (the 'knee').  Adding more")
    print("    capacity past the knee mostly inflates train_ev → memorization.")
    print("  • Reject any k whose ev_gap > 0.10 — the SAE is overfitting and")
    print("    the val_ev there is the only honest number.")
    print("  • If k=128 sits at or near the knee with gap < 0.10, you're done.")
    print("    Launch run_all.sh as planned.")
    print("  • If a different k wins clearly, edit K_SPARSE in run_unsupervised.py")
    print("    BEFORE launching the main pipeline.")
    print("=" * 66)


if __name__ == "__main__":
    main()
