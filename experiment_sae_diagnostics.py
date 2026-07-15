#!/usr/bin/env python3
"""
experiment_sae_diagnostics.py — structural SAE quality metrics
==============================================================

Implements the standard structural-metric battery from the SAE-evaluation
literature (Gao 2024; Karvonen 2024; Bricken 2023; Templeton 2024), beyond the
single explained-variance number the paper currently reports:

  1. Feature geometry  — pairwise cosine similarity among decoder dictionary
     atoms (rows of D.npy); detects redundant / overlapping features.
     Reports per-atom max-cosine-with-any-other and the redundant fraction
     (> threshold). A cross-model contrast (ESM-2 vs RITA dictionary coherence).

  2. Latent firing frequency / feature density — fraction of tokens on which
     each latent is active (from Z.npy). Flags dead latents and ultra-frequent
     ("dense") latents; reports the density distribution.

  3. L2 ratio + reconstruction cosine — magnitude preservation and angular
     fidelity of the SAE reconstruction (needs cached raw_embeddings.npy;
     written by experiment_probe_baseline.py). Complements EV with behaviour-
     agnostic geometry.

All metrics run on cached artefacts (D.npy, Z.npy, raw_embeddings.npy) — CPU only.

Usage:
  python experiment_sae_diagnostics.py --layer-dir outputs_layerwise/esm2/layer_16 \
    --save-dir results_sae_diagnostics/esm2_l16
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

warnings.filterwarnings("ignore")


def feature_geometry(D, redundancy_threshold=0.9, chunk=512):
    """Pairwise cosine among decoder atoms (rows of D). Returns per-atom max-cos + stats."""
    D = np.asarray(D, dtype=np.float32)
    H = D.shape[0]
    norms = np.linalg.norm(D, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Dn = D / norms
    max_cos = np.zeros(H, dtype=np.float32)
    sum_abs = 0.0
    count = 0
    for s in range(0, H, chunk):
        e = min(s + chunk, H)
        S = Dn[s:e] @ Dn.T  # (cs, H)
        for i in range(e - s):
            S[i, s + i] = -np.inf  # exclude self
        max_cos[s:e] = S.max(axis=1)
        # accumulate mean abs off-diagonal (subsample columns for cost)
        block = S.copy()
        block[block == -np.inf] = 0.0
        sum_abs += np.abs(block).sum()
        count += block.size - (e - s)  # minus the self entries we zeroed
    return {
        "n_atoms": int(H),
        "mean_max_cosine": float(max_cos.mean()),
        "median_max_cosine": float(np.median(max_cos)),
        "frac_redundant_gt_%.2f" % redundancy_threshold: float((max_cos > redundancy_threshold).mean()),
        "mean_abs_offdiag_cosine": float(sum_abs / max(count, 1)),
        "_max_cos_per_atom": max_cos,
    }


def firing_frequency(Z, chunk=1024, dense_threshold=0.5):
    """Per-feature firing rate (fraction of tokens active). From Z.npy (mmap)."""
    n_tok, n_feat = Z.shape
    rate = np.zeros(n_feat, dtype=np.float64)
    for s in range(0, n_feat, chunk):
        e = min(s + chunk, n_feat)
        block = np.asarray(Z[:, s:e], dtype=np.float32)
        rate[s:e] = (block > 0).mean(axis=0)
    return {
        "n_features": int(n_feat),
        "n_tokens": int(n_tok),
        "n_dead": int((rate == 0).sum()),
        "n_dense_gt_%.2f" % dense_threshold: int((rate > dense_threshold).sum()),
        "median_firing_rate": float(np.median(rate)),
        "mean_firing_rate": float(rate.mean()),
        "_rate": rate,
    }


def recon_metrics(layer_dir, meta, device="cpu", batch=8192):
    """L2 ratio + reconstruction cosine + recomputed EV, from cached raw embeddings."""
    raw_path = Path(layer_dir) / "raw_embeddings.npy"
    if not raw_path.exists():
        return None
    from sae import SparseAutoencoder
    embed_dim, hidden_dim = meta["embed_dim"], meta["sae_hidden_dim"]
    ns = float(meta.get("norm_scale", 1.0))
    sae = SparseAutoencoder(input_dim=embed_dim, expansion=hidden_dim // embed_dim,
                            k_sparse=meta.get("k_sparse", 256), k_aux=meta.get("k_aux", 64))
    sae.load_state_dict(torch.load(Path(layer_dir) / "sae_model.pt", map_location="cpu"))
    sae = sae.to(device).float().eval()
    raw = np.load(raw_path, mmap_mode="r")
    n = raw.shape[0]
    l2_ratios, cosines = [], []
    tot_resid, tot_var = 0.0, 0.0
    # global mean for EV
    mean_vec = None
    with torch.no_grad():
        for s in range(0, n, batch):
            x = torch.from_numpy(np.asarray(raw[s:s + batch], dtype=np.float32)).to(device)
            xn = x * ns
            xhat_n, _, _ = sae(xn)
            xhat = xhat_n / ns
            l2_ratios.append((xhat.norm(dim=1) / x.norm(dim=1).clamp_min(1e-8)).cpu().numpy())
            cos = torch.nn.functional.cosine_similarity(x, xhat, dim=1)
            cosines.append(cos.cpu().numpy())
            tot_resid += ((x - xhat) ** 2).sum().item()
    raw_mean = np.asarray(raw, dtype=np.float32).mean(axis=0)
    # variance vs mean (streamed)
    for s in range(0, n, batch):
        x = np.asarray(raw[s:s + batch], dtype=np.float32)
        tot_var += ((x - raw_mean) ** 2).sum()
    l2_ratios = np.concatenate(l2_ratios)
    cosines = np.concatenate(cosines)
    return {
        "mean_L2_ratio": float(l2_ratios.mean()),
        "median_L2_ratio": float(np.median(l2_ratios)),
        "mean_recon_cosine": float(cosines.mean()),
        "median_recon_cosine": float(np.median(cosines)),
        "recomputed_EV": float(1.0 - tot_resid / max(tot_var, 1e-8)),
    }


def main():
    ap = argparse.ArgumentParser(description="Structural SAE diagnostics")
    ap.add_argument("--layer-dir", required=True)
    ap.add_argument("--redundancy-threshold", type=float, default=0.9)
    ap.add_argument("--dense-threshold", type=float, default=0.5)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--save-dir", required=True)
    args = ap.parse_args()

    layer_dir = Path(args.layer_dir)
    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    meta = json.loads((layer_dir / "META.json").read_text())

    print("=" * 70)
    print(f"  SAE DIAGNOSTICS — {layer_dir}")
    print("=" * 70)

    D = np.load(layer_dir / "D.npy", mmap_mode="r")
    Z = np.load(layer_dir / "Z.npy", mmap_mode="r")

    print("  [1/3] Feature geometry (decoder cosine redundancy)...")
    geom = feature_geometry(D, args.redundancy_threshold)
    print("  [2/3] Latent firing frequency / density...")
    fire = firing_frequency(Z, dense_threshold=args.dense_threshold)
    print("  [3/3] L2 ratio + reconstruction cosine (cached raw)...")
    recon = recon_metrics(layer_dir, meta, device=args.device)

    summary = {
        "layer_dir": str(layer_dir),
        "model": meta.get("model"), "layer": meta.get("layer"),
        "val_EV_meta": meta.get("val_explained_variance"),
        "geometry": {k: v for k, v in geom.items() if not k.startswith("_")},
        "firing": {k: v for k, v in fire.items() if not k.startswith("_")},
        "reconstruction": recon,
    }
    (save_dir / "diagnostics.json").write_text(json.dumps(summary, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(geom["_max_cos_per_atom"], bins=80, color="#673AB7", alpha=0.8)
    axes[0].axvline(args.redundancy_threshold, color="r", ls="--", label=f"redundant > {args.redundancy_threshold}")
    axes[0].set_xlabel("max cosine with any other decoder atom")
    axes[0].set_ylabel("# features"); axes[0].set_title("Feature geometry (redundancy)")
    axes[0].legend()
    rate = fire["_rate"]
    axes[1].hist(np.log10(rate[rate > 0] + 1e-9), bins=80, color="#009688", alpha=0.8)
    axes[1].set_xlabel("log10 firing rate"); axes[1].set_ylabel("# features")
    axes[1].set_title(f"Latent firing density (dead={fire['n_dead']})")
    fig.tight_layout(); fig.savefig(save_dir / "diagnostics.png", dpi=200); plt.close(fig)

    print("\n  Summary:"); print(json.dumps(summary, indent=2))
    print(f"  Saved to {save_dir}/")


if __name__ == "__main__":
    main()
