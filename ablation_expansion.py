#!/usr/bin/env python3
"""
ablation_expansion.py — expansion factor ablation on ESM-2 layer 16
=====================================================================

Sweeps expansion ∈ {4, 8, 16, 32} with k_sparse=256 fixed.
Uses cached ESM-2 layer-16 embeddings (from ablation_k.py).
Reports: train_ev, val_ev, gap, %dead, mean_L0, %interpretable.

The interpretability metric is computed inline: for each SAE feature,
Pearson r with helix/strand/burial labels, BH FDR correction, then
% of features with any q < 0.05.

Usage:
    python ablation_expansion.py
"""

import os, gc, json, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import t as tdist

from train_sae import train_sae, compute_explained_variance, compute_norm_scale
from run_unsupervised import (
    load_dataset_from_json,
    make_protein_split,
    rows_for_proteins,
    auto_epochs,
    K_AUX, DEAD_THRESHOLD, SPLIT_SEED, VAL_FRACTION,
)

# ── Config ──
ABLATION_LAYER = 16
EXPANSION_VALUES = [4, 8, 16, 32]

# Two sweep strategies:
#   "fixed_k":     k=256 at every expansion (density varies: 5% → 0.6%)
#   "matched_density":  k = 2.5% of hidden_dim (density constant, k varies)
# Both run on the same embeddings.  The 8× row is shared (k=256 in both).
FIXED_K = 256
MATCHED_DENSITY = 0.025  # 2.5%, matching our main k=256 / 10240 setting

EMBED_CACHE = Path("cache/ablation_esm2_layer16.npy")
OUT_DIR = Path("analysis_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _auto_device():
    env = os.environ.get("DEVICE", "").lower()
    if env: return env
    if torch.cuda.is_available(): return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return "mps"
    return "cpu"


def bh_fdr(pvals):
    """Benjamini-Hochberg FDR correction."""
    p = np.asarray(pvals, dtype=np.float64)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(n) + 1.0)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.clip(q, 0.0, 1.0)
    return out.astype(np.float32)


def compute_interpretability(Z, uids, lengths, features_csv, fdr_threshold=0.05):
    """
    Compute % of SAE features that are interpretable (q < threshold for
    at least one of helix/strand/burial).  Simplified inline version of
    cpu_stage's analyze_feature_meanings_residue.
    """
    df_phys = pd.read_csv(features_csv)
    if "position_pdb" in df_phys.columns:
        df_phys = df_phys.rename(columns={"position_pdb": "position"})
    if "position" in df_phys.columns and df_phys["position"].min() == 1:
        df_phys["position"] -= 1

    # Build residue index
    offsets = np.cumsum([0] + [int(l) for l in lengths])
    res_rows = []
    for i, (uid, L) in enumerate(zip(uids, lengths)):
        L = int(L)
        pos = np.arange(L, dtype=np.int32)
        res_rows.append(pd.DataFrame({
            "uid": uid, "position": pos, "res_global": offsets[i] + pos
        }))
    res_idx = pd.concat(res_rows, ignore_index=True)

    # Merge with physical features
    dfm = df_phys.merge(res_idx, on=["uid", "position"], how="inner")
    if len(dfm) == 0:
        return 0.0, 0

    ridx = dfm["res_global"].astype(int).to_numpy()
    ss = dfm.get("ss_8class", pd.Series(["-"] * len(dfm))).astype(str)
    is_helix = ss.isin(["H", "G", "I"]).astype(np.float32).to_numpy()
    is_strand = ss.isin(["E", "B"]).astype(np.float32).to_numpy()
    burial = dfm.get("neighbor_count", pd.Series(np.zeros(len(dfm)))).astype(np.float32).to_numpy()

    n_features = Z.shape[1]
    Zr = np.asarray(Z[ridx, :], dtype=np.float32)
    n = float(Zr.shape[0])

    def corr_pvals(Z_chunk, y):
        y = y.astype(np.float32)
        y0 = y - y.mean()
        y_norm = np.linalg.norm(y0)
        if y_norm == 0:
            return np.ones(Z_chunk.shape[1], dtype=np.float32)
        z0 = Z_chunk - Z_chunk.mean(axis=0, keepdims=True)
        z_norm = np.linalg.norm(z0, axis=0)
        z_norm[z_norm == 0] = 1.0
        r = (z0.T @ y0) / (z_norm * y_norm)
        r = np.clip(r, -0.999999, 0.999999).astype(np.float32)
        df = max(int(n) - 2, 1)
        t_stat = np.abs(r) * np.sqrt(df / (1.0 - r * r))
        p = (2.0 * tdist.sf(t_stat, df=df)).astype(np.float32)
        return p

    pH = corr_pvals(Zr, is_helix)
    pS = corr_pvals(Zr, is_strand)
    pB = corr_pvals(Zr, burial)

    qH = bh_fdr(pH)
    qS = bh_fdr(pS)
    qB = bh_fdr(pB)

    n_interp = int(((qH < fdr_threshold) | (qS < fdr_threshold) | (qB < fdr_threshold)).sum())
    pct_interp = 100.0 * n_interp / n_features
    return pct_interp, n_interp


def main():
    device = _auto_device()
    sae_device = device

    print("=" * 66)
    print(f"  Expansion factor ablation — ESM-2 layer {ABLATION_LAYER}")
    print(f"  Expansion values: {EXPANSION_VALUES}")
    print(f"  k_sparse = {FIXED_K} (fixed-k strategy) / {MATCHED_DENSITY*100:.1f}% density (matched strategy)")
    print(f"  Device: {device}")
    print("=" * 66)

    # ── Load dataset ──
    uids, sequences = load_dataset_from_json(Path("cache/sequences.json"))
    n_proteins = len(sequences)
    lengths = np.array([len(s) for s in sequences], dtype=np.int64)
    offsets = np.cumsum([0] + lengths.tolist()).astype(np.int64)

    train_idx, val_idx = make_protein_split(n_proteins, VAL_FRACTION, SPLIT_SEED)

    # ── Load cached embeddings ──
    if not EMBED_CACHE.exists():
        print(f"\n  ERROR: {EMBED_CACHE} not found.")
        print(f"  Run ablation_k.py first to extract and cache ESM-2 layer 16 embeddings.")
        return

    print(f"\n  Loading cached embeddings from {EMBED_CACHE}...")
    X_tokens = np.load(EMBED_CACHE)
    T, D = X_tokens.shape
    print(f"  Shape: {T} x {D}")

    train_rows = rows_for_proteins(offsets, train_idx)
    val_rows = rows_for_proteins(offsets, val_idx)
    X_train_raw = np.ascontiguousarray(X_tokens[train_rows])
    X_val_raw = np.ascontiguousarray(X_tokens[val_rows])

    norm_scale = compute_norm_scale(X_train_raw)
    X_train = (X_train_raw * norm_scale).astype(np.float32)
    X_val = (X_val_raw * norm_scale).astype(np.float32)
    print(f"  Bricken scale: {norm_scale:.6f}")
    print(f"  train={len(X_train)}, val={len(X_val)}")

    # For interpretability: need Z on ALL tokens (train + val)
    X_all_norm = (X_tokens * norm_scale).astype(np.float32)

    epochs = auto_epochs(X_train.shape[0])

    # ── Build sweep configurations ──
    # Two strategies: fixed k vs matched density.
    # The 8× matched-density row (k=256) is the same as the 8× fixed-k row,
    # so we deduplicate to avoid redundant training.
    configs = []
    seen = set()
    for exp in EXPANSION_VALUES:
        hidden_dim = D * exp
        # Fixed-k strategy
        k_fixed = FIXED_K
        key = (exp, k_fixed)
        if key not in seen:
            configs.append({"expansion": exp, "k_sparse": k_fixed, "strategy": "fixed_k"})
            seen.add(key)
        # Matched-density strategy
        k_matched = max(1, int(round(hidden_dim * MATCHED_DENSITY)))
        key = (exp, k_matched)
        if key not in seen:
            configs.append({"expansion": exp, "k_sparse": k_matched, "strategy": "matched_density"})
            seen.add(key)

    print(f"\n  Sweep configurations ({len(configs)} unique SAE trainings):")
    for c in configs:
        hd = D * c["expansion"]
        dens = 100.0 * c["k_sparse"] / hd
        print(f"    {c['strategy']:17s}  expansion={c['expansion']:>2d}x  "
              f"hidden={hd:>5d}  k={c['k_sparse']:>4d}  density={dens:.2f}%")

    # ── Sweep ──
    import train_sae as ts
    from train_sae import extract_sae_features
    rows = []
    for c in configs:
        exp = c["expansion"]
        k = c["k_sparse"]
        strategy = c["strategy"]
        hidden_dim = D * exp
        density = 100.0 * k / hidden_dim

        print(f"\n{'='*66}")
        print(f"  [{strategy}]  expansion={exp}x  k={k}  "
              f"(hidden={hidden_dim}, density={density:.2f}%)")
        print(f"{'='*66}")

        # Clamp k_aux to at most k/4 to maintain the Gao et al. ratio
        k_aux = min(K_AUX, max(1, k // 4))

        ts._HW_CONFIG = None
        torch.manual_seed(42)
        np.random.seed(42)

        t0 = time.time()
        sae = train_sae(
            X_train, input_dim=D, device=sae_device,
            epochs=epochs, lr=5e-5,
            expansion=exp, k_sparse=k,
            k_aux=k_aux, dead_threshold=DEAD_THRESHOLD,
        )
        wall = time.time() - t0

        ts._HW_CONFIG = None
        train_ev = float(compute_explained_variance(sae, X_train, device=sae_device))
        val_ev = float(compute_explained_variance(sae, X_val, device=sae_device))
        gap = train_ev - val_ev

        # Dead latents
        base = sae._orig_mod if hasattr(sae, "_orig_mod") else sae
        n_dead = int(base._get_dead_latent_mask().sum().item())
        pct_dead = 100.0 * n_dead / hidden_dim

        # Extract Z for interpretability
        ts._HW_CONFIG = None
        Z, _ = extract_sae_features(sae, X_all_norm, device=sae_device)

        # Interpretability
        pct_interp, n_interp = compute_interpretability(
            Z, uids, lengths, "cache/residue_features.csv")

        print(f"\n  Results:")
        print(f"    train_ev      = {train_ev:+.4f}")
        print(f"    val_ev        = {val_ev:+.4f}")
        print(f"    gap           = {gap:+.4f}")
        print(f"    dead          = {n_dead}/{hidden_dim} ({pct_dead:.1f}%)")
        print(f"    interpretable = {n_interp}/{hidden_dim} ({pct_interp:.1f}%)")
        print(f"    wall time     = {wall:.0f}s")

        rows.append({
            "strategy": strategy,
            "expansion": exp,
            "hidden_dim": hidden_dim,
            "k_sparse": k,
            "density_pct": density,
            "k_aux": k_aux,
            "train_ev": train_ev,
            "val_ev": val_ev,
            "ev_gap": gap,
            "n_dead": n_dead,
            "pct_dead": pct_dead,
            "pct_interpretable": pct_interp,
            "n_interpretable": n_interp,
            "wall_seconds": wall,
        })

        del sae, Z, base
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        if hasattr(torch, "mps"):
            torch.mps.empty_cache()

    # ── Save ──
    df = pd.DataFrame(rows)
    csv_path = OUT_DIR / "ablation_expansion.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved {csv_path}")
    print()
    print(df.to_string(index=False))

    # ── Plot: two-panel (fixed-k vs matched-density) ──
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    for row_idx, (strategy, title_prefix) in enumerate([
        ("fixed_k", f"Fixed k={FIXED_K}"),
        ("matched_density", f"Matched density ~{MATCHED_DENSITY*100:.1f}%"),
    ]):
        sub = df[df["strategy"] == strategy].sort_values("expansion")
        if sub.empty:
            continue
        exps = sub["expansion"].values

        ax = axes[row_idx, 0]
        ax.plot(exps, sub["train_ev"], "o-", color="#1f77b4", lw=2, ms=10, label="Train EV")
        ax.plot(exps, sub["val_ev"], "s--", color="#d62728", lw=2, ms=10, label="Val EV")
        ax.set_xlabel("Expansion factor")
        ax.set_ylabel("Explained variance")
        ax.set_title(f"{title_prefix}: Reconstruction")
        ax.set_xticks(exps); ax.set_xticklabels([f"{e}x" for e in exps])
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        ax = axes[row_idx, 1]
        ax.plot(exps, sub["pct_interpretable"], "o-", color="#2ca02c", lw=2, ms=10)
        ax.set_xlabel("Expansion factor")
        ax.set_ylabel("% interpretable (q < 0.05)")
        ax.set_title(f"{title_prefix}: Interpretability")
        ax.set_xticks(exps); ax.set_xticklabels([f"{e}x" for e in exps])
        ax.grid(alpha=0.3)

        ax = axes[row_idx, 2]
        ax.plot(exps, sub["pct_dead"], "o-", color="#ff7f0e", lw=2, ms=10)
        ax.set_xlabel("Expansion factor")
        ax.set_ylabel("% dead latents")
        ax.set_title(f"{title_prefix}: Dead latents")
        ax.set_xticks(exps); ax.set_xticklabels([f"{e}x" for e in exps])
        ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "ablation_expansion.png", dpi=220)
    fig.savefig(OUT_DIR / "ablation_expansion.pdf")
    plt.close(fig)
    print(f"  Saved {OUT_DIR / 'ablation_expansion.png'}")

    print(f"\n{'='*66}")
    print("  EXPANSION ABLATION COMPLETE")
    print(f"{'='*66}")


if __name__ == "__main__":
    main()
