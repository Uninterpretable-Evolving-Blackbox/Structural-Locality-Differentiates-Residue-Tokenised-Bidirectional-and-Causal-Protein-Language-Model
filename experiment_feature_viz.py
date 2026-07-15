#!/usr/bin/env python3
"""
experiment_feature_viz.py — E6: visualise top features on 3D structures
=======================================================================

Reviewer 6P8E: "show qualitative examples of important features on real protein
structures." Renders, for the top structural-locality features, the protein on
which the feature fires most, as a 3D Cα trace coloured by feature activation,
alongside the structural contact map with active residues highlighted. Makes the
biology legible (e.g. a feature lighting up a β-sheet pairing or buried core).

Uses matplotlib 3D (no py3Dmol dependency).

Usage:
  python experiment_feature_viz.py --layer-dir outputs_layerwise/esm2/layer_16 \
    --pdb-dir cache/pdb_files --top-k 6 --save-dir results_feature_viz/esm2_l16
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
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.spatial import KDTree

from cpu_stage import load_ref_seqs, load_layer
from experiment_activation_clamping import extract_ca_coords

warnings.filterwarnings("ignore")


def render_feature(coords, acts, contacts_pairs, feat, uid, save_path, conceptC=None):
    valid = ~np.isnan(coords).any(axis=1)
    c = coords.copy()
    c[~valid] = np.nanmean(coords[valid], axis=0) if valid.any() else 0.0
    a = acts.copy()
    a = (a - a.min()) / (a.max() - a.min() + 1e-9)

    fig = plt.figure(figsize=(12, 5))
    # 3D Cα trace
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.plot(c[:, 0], c[:, 1], c[:, 2], color="lightgrey", lw=1, alpha=0.7)
    sc = ax.scatter(c[:, 0], c[:, 1], c[:, 2], c=a, cmap="hot_r", s=25,
                    vmin=0, vmax=1, edgecolors="none")
    fig.colorbar(sc, ax=ax, shrink=0.6, label="feature activation")
    ax.set_title(f"feature {feat} on {uid}\n3D Cα trace (hot = active)")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

    # contact map + active residues
    ax2 = fig.add_subplot(1, 2, 2)
    L = len(acts)
    cmap = np.zeros((L, L))
    for i, j in contacts_pairs:
        if i < L and j < L:
            cmap[i, j] = cmap[j, i] = 1
    ax2.imshow(cmap, cmap="Greys", origin="lower", alpha=0.5)
    active = np.where(a > 0.6)[0]
    ax2.scatter(active, active, c="red", s=12, label="active residue (>0.6)")
    ax2.set_xlabel("residue"); ax2.set_ylabel("residue")
    ttl = f"contacts + active residues"
    if conceptC:
        ttl += f"\nbest concept: {conceptC}"
    ax2.set_title(ttl)
    ax2.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="E6 feature visualisation")
    ap.add_argument("--layer-dir", required=True)
    ap.add_argument("--pdb-dir", default="cache/pdb_files")
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("--concept-csv", default=None,
                    help="optional feature_concept_best.csv from E0 for labels")
    ap.add_argument("--save-dir", required=True)
    args = ap.parse_args()

    layer_dir = Path(args.layer_dir)
    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)

    Z, uids, lengths = load_layer(layer_dir)
    uids = [str(u) for u in uids]
    ref = load_ref_seqs(layer_dir)
    offsets, off = [], 0
    for L in lengths:
        offsets.append(off); off += int(L)

    ss = pd.read_csv(layer_dir / "struct_seq_metrics.csv").sort_values("struct_delta", ascending=False)
    top_features = ss.head(args.top_k)["feature_idx"].astype(int).tolist()

    concept_map = {}
    if args.concept_csv and Path(args.concept_csv).exists():
        cdf = pd.read_csv(args.concept_csv)
        concept_map = dict(zip(cdf["feature_idx"], cdf["best_concept"]))

    print("=" * 70)
    print(f"  E6 FEATURE VISUALISATION — {layer_dir.name} | top {args.top_k} features")
    print("=" * 70)

    rendered = []
    for feat in top_features:
        # find protein where this feature fires most strongly
        best_gi, best_val = None, -1
        for gi, uid in enumerate(uids):
            base = offsets[gi]
            seg = np.asarray(Z[base:base + int(lengths[gi]), feat], dtype=np.float32)
            if seg.size and seg.max() > best_val:
                pdb = Path(args.pdb_dir) / f"{uid[1:5].lower()}.pdb"
                if pdb.exists():
                    best_val, best_gi = seg.max(), gi
        if best_gi is None:
            continue
        uid = uids[best_gi]
        seq = ref[uid]
        base = offsets[best_gi]
        acts = np.asarray(Z[base:base + len(seq), feat], dtype=np.float32)
        coords = extract_ca_coords(str(Path(args.pdb_dir) / f"{uid[1:5].lower()}.pdb"), seq)
        if coords is None:
            continue
        valid = ~np.isnan(coords).any(axis=1)
        idx = np.where(valid)[0]
        pairs = []
        if len(idx) >= 2:
            for a_i, b_i in KDTree(coords[valid]).query_pairs(r=8.0):
                ra, rb = int(idx[a_i]), int(idx[b_i])
                if abs(ra - rb) >= 12:
                    pairs.append((ra, rb))
        out_png = save_dir / f"feature_{feat}_{uid}.png"
        render_feature(coords, acts, pairs, feat, uid, out_png,
                       conceptC=concept_map.get(feat))
        rendered.append({"feature_idx": int(feat), "uid": uid,
                         "max_activation": float(best_val),
                         "struct_delta": float(ss.set_index("feature_idx").loc[feat, "struct_delta"]),
                         "best_concept": concept_map.get(feat, ""), "png": str(out_png)})
        print(f"  feature {feat}: rendered on {uid} (max act {best_val:.3f})")

    pd.DataFrame(rendered).to_csv(save_dir / "rendered_features.csv", index=False)
    print(f"\n  Saved {len(rendered)} figures to {save_dir}/")


if __name__ == "__main__":
    main()
