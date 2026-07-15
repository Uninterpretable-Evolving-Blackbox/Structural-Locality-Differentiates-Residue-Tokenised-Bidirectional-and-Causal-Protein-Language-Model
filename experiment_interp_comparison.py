#!/usr/bin/env python3
"""
experiment_interp_comparison.py — match the metric to the annotation type
=========================================================================

Per review feedback: the paper mixes metric philosophies. The principled rule is
  - CONTINUOUS annotations (RSA, Cα coordination/burial)  -> Spearman rank corr
  - CATEGORICAL/span annotations (SS, SCOPe concepts)     -> domain-aware F1 (E0)
and L_struct is a separate GEOMETRIC property, not an interpretability metric.

This script computes, per SAE feature, the methodologically-correct interpretability
signal for each annotation type and lays them side by side so we can ask:
  - how many features are "interpretable" under each metric?
  - do the metrics AGREE (rank-correlate) or capture different feature populations?

Continuous Spearman is computed here (vectorised rank+Pearson, FDR-controlled);
categorical concept-F1 is read from results_concept_f1/<layer>/feature_concept_best.csv;
L_struct from the layer's struct_seq_metrics.csv.

Usage:
  python experiment_interp_comparison.py --layer-dir outputs_layerwise/esm2/layer_16 \
    --concept-csv results_concept_f1/esm2_l16/feature_concept_best.csv \
    --save-dir results_interp_comparison/esm2_l16
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
from scipy.stats import rankdata, t as tdist

from cpu_stage import load_layer, load_phys_features, bh_fdr

warnings.filterwarnings("ignore")


def spearman_chunk(Z_sub, y_rank, n):
    """Spearman rho + p for a chunk (n_res x cs) vs pre-ranked y, via rank+Pearson."""
    cs = Z_sub.shape[1]
    rho = np.zeros(cs, dtype=np.float32)
    pvals = np.ones(cs, dtype=np.float32)
    yr = y_rank - y_rank.mean()
    yn = np.linalg.norm(yr)
    if yn == 0:
        return rho, pvals
    for j in range(cs):
        col = Z_sub[:, j]
        if np.ptp(col) == 0:
            continue
        xr = rankdata(col)
        xr = xr - xr.mean()
        xn = np.linalg.norm(xr)
        if xn == 0:
            continue
        r = float(np.dot(xr, yr) / (xn * yn))
        rho[j] = r
        df = max(n - 2, 1)
        tval = abs(r) * np.sqrt(df / max(1.0 - r * r, 1e-12))
        pvals[j] = float(2.0 * tdist.sf(tval, df=df))
    return rho, pvals


def main():
    ap = argparse.ArgumentParser(description="Continuous-Spearman vs categorical-F1 vs L_struct")
    ap.add_argument("--layer-dir", required=True)
    ap.add_argument("--features-csv", default="cache/residue_features.csv")
    ap.add_argument("--concept-csv", default=None)
    ap.add_argument("--struct-csv", default=None)
    ap.add_argument("--max-residues", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-dir", required=True)
    args = ap.parse_args()

    layer_dir = Path(args.layer_dir)
    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)

    Z, uids, lengths = load_layer(layer_dir)
    uids = [str(u) for u in uids]
    n_features = int(Z.shape[1])
    offsets, off = {}, 0
    for u, L in zip(uids, lengths):
        offsets[u] = off; off += int(L)

    df = load_phys_features(Path(args.features_csv))
    # align continuous annotations to global residue rows
    rsa = np.full(int(Z.shape[0]), np.nan, dtype=np.float32)
    coord = np.full(int(Z.shape[0]), np.nan, dtype=np.float32)
    for u, g in df.groupby("uid"):
        u = str(u)
        if u not in offsets:
            continue
        base = offsets[u]
        pos = g["position"].to_numpy().astype(int)
        ok = (pos >= 0) & (pos < int(lengths[uids.index(u)]))
        gi = base + pos[ok]
        if "sasa" in g.columns:
            rsa[gi] = g["sasa"].to_numpy().astype(np.float32)[ok]
        if "neighbor_count" in g.columns:
            coord[gi] = g["neighbor_count"].to_numpy().astype(np.float32)[ok]

    valid = ~(np.isnan(rsa) | np.isnan(coord))
    res = np.where(valid)[0]
    rng = np.random.default_rng(args.seed)
    if len(res) > args.max_residues:
        res = np.sort(rng.choice(res, args.max_residues, replace=False))
    n = len(res)
    print(f"  Spearman on {n} residues x {n_features} features (continuous RSA + coordination)")

    rsa_rank = rankdata(rsa[res])
    coord_rank = rankdata(coord[res])

    rho_rsa = np.zeros(n_features, dtype=np.float32); p_rsa = np.ones(n_features, dtype=np.float32)
    rho_co = np.zeros(n_features, dtype=np.float32); p_co = np.ones(n_features, dtype=np.float32)
    chunk = 512
    for s in range(0, n_features, chunk):
        e = min(s + chunk, n_features)
        Zsub = np.asarray(Z[res, s:e], dtype=np.float32)
        rho_rsa[s:e], p_rsa[s:e] = spearman_chunk(Zsub, rsa_rank, n)
        rho_co[s:e], p_co[s:e] = spearman_chunk(Zsub, coord_rank, n)
    q_rsa, q_co = bh_fdr(p_rsa), bh_fdr(p_co)

    out = pd.DataFrame({
        "feature_idx": np.arange(n_features),
        "spearman_rsa": rho_rsa, "q_rsa": q_rsa,
        "spearman_coord": rho_co, "q_coord": q_co,
    })

    # merge categorical concept-F1 + L_struct
    concept_csv = args.concept_csv
    if concept_csv and Path(concept_csv).exists():
        cdf = pd.read_csv(concept_csv)[["feature_idx", "val_f1", "test_f1", "best_concept"]]
        out = out.merge(cdf.rename(columns={"val_f1": "concept_valF1", "test_f1": "concept_testF1"}),
                        on="feature_idx", how="left")
    struct_csv = args.struct_csv or (layer_dir / "struct_seq_metrics.csv")
    if Path(struct_csv).exists():
        sdf = pd.read_csv(struct_csv)[["feature_idx", "struct_delta"]]
        out = out.merge(sdf, on="feature_idx", how="left")
    out.to_csv(save_dir / "interp_comparison_per_feature.csv", index=False)

    # cross-metric agreement
    metrics = {}
    metrics["|spearman_rsa|"] = out["spearman_rsa"].abs()
    metrics["|spearman_coord|"] = out["spearman_coord"].abs()
    if "concept_valF1" in out:
        metrics["concept_valF1"] = out["concept_valF1"].fillna(0)
    if "struct_delta" in out:
        metrics["struct_delta"] = out["struct_delta"].fillna(0)
    M = pd.DataFrame(metrics)
    corr = M.corr(method="spearman")
    corr.to_csv(save_dir / "metric_agreement_spearman.csv")

    summary = {
        "layer_dir": str(layer_dir), "n_features": n_features, "n_residues": n,
        "n_interp_RSA_q<0.05": int((q_rsa < 0.05).sum()),
        "n_interp_coord_q<0.05": int((q_co < 0.05).sum()),
    }
    if "concept_valF1" in out:
        summary["n_concept_valF1>0.5"] = int((out["concept_valF1"].fillna(0) > 0.5).sum())
    summary["cross_metric_spearman"] = corr.round(3).to_dict()
    (save_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns, fontsize=8)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.7, label="Spearman ρ between metrics")
    ax.set_title(f"Do interpretability metrics agree?\n{layer_dir.name}")
    fig.tight_layout(); fig.savefig(save_dir / "metric_agreement.png", dpi=200); plt.close(fig)

    print("\n  Summary:"); print(json.dumps(summary, indent=2))
    print(f"  Saved to {save_dir}/")


if __name__ == "__main__":
    main()
