#!/usr/bin/env python3
"""
experiment_joint_sweep.py — JOINT k_sparse x expansion SAE sweep
================================================================

The paper swept k_sparse {64,128,256} (Appendix A) and expansion {4,8,16,32}
(Appendix B) as SEPARATE 1-D sweeps, each holding the other fixed, and only on
ESM-2 layer 16. This runs the full 2-D GRID (k x expansion) so interactions are
visible, and supports running it across multiple layers and both H1 models
(ESM-2 and RITA) to address the reviewer concern that the SAE configuration may
not be equally optimal across models/depths.

Per grid cell we report the same axes the paper uses:
  - reconstruction: train/val explained variance, EV gap
  - sparsity health: measured L0, % dead latents, dictionary density
  - interpretability: %Interp (fraction of features correlating with
    helix/strand/burial at BH-FDR q<0.05) and n_interp  [paper's metric]
  - geometry: mean max decoder-atom cosine (redundancy)

Training input is the cached raw_embeddings.npy (written by
experiment_probe_baseline.py); available for esm2 {0,16,32} and rita {0,12,23}.

Usage:
  python experiment_joint_sweep.py --layer-dir outputs_layerwise/esm2/layer_16 \
    --ks 64,128,256 --expansions 4,8,16,32 --save-dir results_joint_sweep/esm2_l16

  # smoke (tiny)
  python experiment_joint_sweep.py --layer-dir outputs_layerwise/esm2/layer_16 \
    --ks 64 --expansions 4 --epochs 2 --save-dir /tmp/joint_smoke
"""

import argparse
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from scipy.stats import t as tdist

from train_sae import (
    train_sae, compute_explained_variance, compute_norm_scale, extract_sae_features,
)
from cpu_stage import load_layer, load_ref_seqs, bh_fdr

warnings.filterwarnings("ignore")


def auto_epochs(total_tokens):
    if total_tokens < 20_000:  return 20
    if total_tokens < 100_000: return 40
    if total_tokens < 300_000: return 60
    if total_tokens < 800_000: return 80
    return 100


def pct_interpretable(Z, uids, lengths, features_csv, fdr=0.05, chunk=512):
    """%Interp (paper metric), chunked over features to bound memory on 32x."""
    df = pd.read_csv(features_csv)
    if "position_pdb" in df.columns:
        df = df.rename(columns={"position_pdb": "position"})
    if "position" in df.columns and df["position"].min() == 1:
        df["position"] -= 1
    offsets = np.cumsum([0] + [int(l) for l in lengths])
    res_rows = []
    for i, (uid, L) in enumerate(zip(uids, lengths)):
        L = int(L); pos = np.arange(L, dtype=np.int32)
        res_rows.append(pd.DataFrame({"uid": uid, "position": pos, "res_global": offsets[i] + pos}))
    res_idx = pd.concat(res_rows, ignore_index=True)
    dfm = df.merge(res_idx, on=["uid", "position"], how="inner")
    if len(dfm) == 0:
        return 0.0, 0
    ridx = dfm["res_global"].astype(int).to_numpy()
    ss = dfm.get("ss_8class", pd.Series(["-"] * len(dfm))).astype(str)
    ys = {
        "helix": ss.isin(["H", "G", "I"]).astype(np.float32).to_numpy(),
        "strand": ss.isin(["E", "B"]).astype(np.float32).to_numpy(),
        "burial": dfm.get("neighbor_count", pd.Series(np.zeros(len(dfm)))).astype(np.float32).to_numpy(),
    }
    n = float(len(ridx))
    n_feat = Z.shape[1]
    pmins = np.ones(n_feat, dtype=np.float32)  # min p across the 3 annotations
    y0s = {kk: (v - v.mean()) for kk, v in ys.items()}
    ynorms = {kk: np.linalg.norm(v) for kk, v in y0s.items()}
    df_t = max(int(n) - 2, 1)
    for s in range(0, n_feat, chunk):
        e = min(s + chunk, n_feat)
        Zc = np.asarray(Z[ridx, s:e], dtype=np.float32)
        z0 = Zc - Zc.mean(axis=0, keepdims=True)
        zn = np.linalg.norm(z0, axis=0); zn[zn == 0] = 1.0
        pc = np.ones(e - s, dtype=np.float32)
        for kk in ys:
            if ynorms[kk] == 0:
                continue
            r = (z0.T @ y0s[kk]) / (zn * ynorms[kk])
            r = np.clip(r, -0.999999, 0.999999)
            tstat = np.abs(r) * np.sqrt(df_t / (1.0 - r * r))
            p = (2.0 * tdist.sf(tstat, df=df_t)).astype(np.float32)
            pc = np.minimum(pc, p)
        pmins[s:e] = pc
    q = bh_fdr(pmins)
    n_interp = int((q < fdr).sum())
    return 100.0 * n_interp / n_feat, n_interp


def decoder_redundancy(sae, chunk=512):
    D = sae.decoder.weight.detach().cpu().numpy().T.astype(np.float32)  # (hidden, in)
    norms = np.linalg.norm(D, axis=1, keepdims=True); norms[norms == 0] = 1.0
    Dn = D / norms
    H = D.shape[0]
    mx = np.zeros(H, dtype=np.float32)
    for s in range(0, H, chunk):
        e = min(s + chunk, H)
        S = Dn[s:e] @ Dn.T
        for i in range(e - s):
            S[i, s + i] = -np.inf
        mx[s:e] = S.max(axis=1)
    return float(mx.mean())


def main():
    ap = argparse.ArgumentParser(description="Joint k x expansion SAE sweep")
    ap.add_argument("--layer-dir", required=True)
    ap.add_argument("--ks", default="64,128,256")
    ap.add_argument("--expansions", default="4,8,16,32")
    ap.add_argument("--features-csv", default="cache/residue_features.csv")
    ap.add_argument("--epochs", type=int, default=0, help="0 = auto (paper schedule)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-dir", required=True)
    args = ap.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    layer_dir = Path(args.layer_dir)
    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    ks = [int(x) for x in args.ks.split(",")]
    exps = [int(x) for x in args.expansions.split(",")]

    raw_path = layer_dir / "raw_embeddings.npy"
    if not raw_path.exists():
        raise SystemExit(f"need cached raw embeddings: {raw_path} (run experiment_probe_baseline first)")
    meta = json.loads((layer_dir / "META.json").read_text())
    val_uids = set(meta.get("val_uids", []))

    _, uids, lengths = load_layer(layer_dir)
    uids = [str(u) for u in uids]
    X = np.load(raw_path, mmap_mode="r")
    embed_dim = X.shape[1]
    offs = np.cumsum([0] + [int(l) for l in lengths])
    tr, va = [], []
    for i, u in enumerate(uids):
        idx = np.arange(offs[i], offs[i + 1])
        (va if u in val_uids else tr).append(idx)
    tr = np.concatenate(tr); va = np.concatenate(va)

    X_train = np.ascontiguousarray(X[tr])
    ns = compute_norm_scale(X_train)
    X_train = (X_train * ns).astype(np.float32)
    X_val = (np.asarray(X[va]) * ns).astype(np.float32)
    X_all = (np.asarray(X) * ns).astype(np.float32)
    epochs = args.epochs if args.epochs > 0 else auto_epochs(len(X_train))

    print("=" * 70)
    print(f"  JOINT SWEEP — {layer_dir} | k={ks} x exp={exps} | epochs={epochs} | {device}")
    print(f"  embed_dim={embed_dim} | train={len(X_train)} val={len(X_val)}")
    print("=" * 70)

    rows = []
    for exp in exps:
        for k in ks:
            hidden = embed_dim * exp
            print(f"\n  --- expansion={exp}x (hidden={hidden}) k={k} ---")
            torch.manual_seed(args.seed); np.random.seed(args.seed)
            sae = train_sae(X_train, input_dim=embed_dim, device=device, epochs=epochs,
                            lr=5e-5, expansion=exp, k_sparse=k, k_aux=64,
                            dead_threshold=1_000_000)
            train_ev = float(compute_explained_variance(sae, X_train, device=device))
            val_ev = float(compute_explained_variance(sae, X_val, device=device))
            Z, _ = extract_sae_features(sae, X_all, device=device, save_dir=None)
            l0 = float((Z > 0).sum(axis=1).mean())
            pct_dead = float(100.0 * (~(Z > 0).any(axis=0)).mean())
            pct_interp, n_interp = pct_interpretable(Z, uids, lengths, args.features_csv)
            redundancy = decoder_redundancy(sae)
            del Z
            row = {
                "k_sparse": k, "expansion": exp, "hidden_dim": hidden,
                "density_pct": 100.0 * k / hidden,
                "train_ev": train_ev, "val_ev": val_ev, "ev_gap": train_ev - val_ev,
                "measured_L0": l0, "pct_dead": pct_dead,
                "pct_interp": pct_interp, "n_interp": n_interp,
                "mean_max_decoder_cosine": redundancy,
            }
            rows.append(row)
            print(f"    val_ev={val_ev:.4f} gap={train_ev-val_ev:+.4f} L0={l0:.0f} "
                  f"dead={pct_dead:.1f}% %interp={pct_interp:.1f} redund={redundancy:.3f}")
            pd.DataFrame(rows).to_csv(save_dir / "joint_sweep.csv", index=False)  # checkpoint

    df = pd.DataFrame(rows)
    df.to_csv(save_dir / "joint_sweep.csv", index=False)

    # heatmaps: val_ev and pct_interp over (expansion x k)
    for metric, cmap in [("val_ev", "viridis"), ("pct_interp", "magma"), ("ev_gap", "coolwarm")]:
        piv = df.pivot(index="expansion", columns="k_sparse", values=metric)
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(piv.values, aspect="auto", cmap=cmap, origin="lower")
        ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns)
        ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
        ax.set_xlabel("k_sparse"); ax.set_ylabel("expansion")
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                ax.text(j, i, f"{piv.values[i,j]:.2f}", ha="center", va="center",
                        color="w", fontsize=8)
        fig.colorbar(im, ax=ax, label=metric)
        ax.set_title(f"{metric} — {layer_dir.name}")
        fig.tight_layout(); fig.savefig(save_dir / f"heatmap_{metric}.png", dpi=200); plt.close(fig)

    best = df.loc[df["val_ev"].idxmax()]
    (save_dir / "summary.json").write_text(json.dumps({
        "layer_dir": str(layer_dir), "ks": ks, "expansions": exps, "epochs": epochs,
        "best_val_ev_cell": {"k": int(best.k_sparse), "expansion": int(best.expansion),
                             "val_ev": float(best.val_ev), "pct_interp": float(best.pct_interp)},
    }, indent=2))
    print(f"\n  Grid complete. Best val_EV at k={int(best.k_sparse)}, exp={int(best.expansion)}x.")
    print(f"  Saved to {save_dir}/")


if __name__ == "__main__":
    main()
