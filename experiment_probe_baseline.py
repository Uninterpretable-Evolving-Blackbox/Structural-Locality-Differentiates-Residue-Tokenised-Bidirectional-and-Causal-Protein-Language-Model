#!/usr/bin/env python3
"""
experiment_probe_baseline.py — "Do you even need SAEs?" linear-probe baseline
=============================================================================

Reviewer U2Yp / Adams et al. (ICML 2025) test: compare a linear probe trained on
RAW PLM activations against one trained on SAE features, for the same structural
readouts, at matched depths, for ESM-2 vs RITA.

Two outcomes both strengthen the paper:
  (a) raw probe reproduces the ESM-2 > RITA structural gap  -> H1 is robust to
      method, not an SAE artefact; OR
  (b) SAE features beat raw for structure decodability (Adams found this for
      structure) -> SAE-specific value, esp. at layer 0 where whole-vector
      cosine is blind (your Appendix L).

Readouts (per residue):  helix, strand, burial(binary).  Optional: long-range
contact probe on residue pairs (--with-contacts).

Raw activations are not persisted by the pipeline, so we re-extract them for the
requested (model, layer) and cache to <layer-dir>/raw_embeddings.npy.

Usage:
  python experiment_probe_baseline.py --layer-dir outputs_layerwise/esm2/layer_16 \
    --model esm2 --layer 16 --save-dir results_probe/esm2_l16

  # smoke test on 80 proteins
  python experiment_probe_baseline.py --layer-dir outputs_layerwise/esm2/layer_16 \
    --model esm2 --layer 16 --save-dir /tmp/probe_smoke --max-proteins 80
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

from cpu_stage import load_layer, load_ref_seqs, load_phys_features

warnings.filterwarnings("ignore")

MODEL_EXTRACTORS = {
    "esm2": ("facebook/esm2_t33_650M_UR50D", "extract_esm2_embeddings"),
    "rita": ("lightonai/RITA_l", "extract_rita_embeddings"),
    "esm2_small": ("facebook/esm2_t12_35M_UR50D", "extract_esm2_embeddings"),
    "rita_small": ("lightonai/RITA_s", "extract_rita_embeddings"),
}


def get_raw_embeddings(layer_dir, model, layer, uids, sequences, cache=True):
    """Re-extract raw PLM hidden states for (model, layer); cache to disk."""
    cache_path = Path(layer_dir) / "raw_embeddings.npy"
    if cache and cache_path.exists():
        arr = np.load(cache_path, mmap_mode="r")
        if arr.shape[0] == sum(len(s) for s in sequences):
            print(f"  Loaded cached raw embeddings {arr.shape} from {cache_path}")
            return arr
    import extract_embeddings as ee
    model_name, fn_name = MODEL_EXTRACTORS[model]
    fn = getattr(ee, fn_name)
    print(f"  Extracting raw {model} layer {layer} ({model_name})...")
    kwargs = dict(layers=[layer], model_name=model_name)
    emb = fn(list(sequences), **kwargs)
    arr = np.asarray(emb[layer], dtype=np.float32)
    if cache:
        np.save(cache_path, arr)
        print(f"  Cached raw embeddings {arr.shape} -> {cache_path}")
    return arr


def build_labels(uids, lengths, offsets, df_phys, burial_q=0.5):
    """Per-residue labels: helix, strand, burial(binary by global median)."""
    N = int(sum(int(l) for l in lengths))
    helix = np.zeros(N, dtype=np.int8)
    strand = np.zeros(N, dtype=np.int8)
    burial_raw = np.full(N, np.nan, dtype=np.float32)
    if df_phys is None or len(df_phys) == 0:
        return None
    by_uid = {u: g for u, g in df_phys.groupby("uid")}
    for uid, L in zip(uids, lengths):
        base = offsets[uid]
        g = by_uid.get(uid)
        if g is None:
            continue
        pos = g["position"].to_numpy().astype(int)
        ok = (pos >= 0) & (pos < int(L))
        pos = pos[ok]
        gi = base + pos
        if "ss_8class" in g.columns:
            ss = g["ss_8class"].astype(str).to_numpy()[ok]
            helix[gi] = np.isin(ss, ["H", "G", "I"]).astype(np.int8)
            strand[gi] = np.isin(ss, ["E", "B"]).astype(np.int8)
        if "neighbor_count" in g.columns:
            burial_raw[gi] = g["neighbor_count"].to_numpy().astype(np.float32)[ok]
    med = np.nanmedian(burial_raw)
    burial = (burial_raw > med).astype(np.int8)
    valid = ~np.isnan(burial_raw)
    return {"helix": helix, "strand": strand, "burial": burial, "valid_burial": valid}


def probe(X_tr, y_tr, X_te, y_te, max_iter=300):
    """Fit logistic probe, return (F1, AUROC) on test."""
    if y_tr.sum() < 5 or (len(y_tr) - y_tr.sum()) < 5:
        return float("nan"), float("nan")
    clf = LogisticRegression(max_iter=max_iter, C=1.0, class_weight="balanced",
                             solver="liblinear")
    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_te)
    try:
        proba = clf.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(y_te, proba) if len(np.unique(y_te)) > 1 else float("nan")
    except Exception:
        auc = float("nan")
    return f1_score(y_te, pred), auc


def main():
    ap = argparse.ArgumentParser(description="SAE-vs-raw linear probe baseline")
    ap.add_argument("--layer-dir", required=True)
    ap.add_argument("--model", required=True, choices=list(MODEL_EXTRACTORS))
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--features-csv", default="cache/residue_features.csv")
    ap.add_argument("--max-proteins", type=int, default=0, help="subsample for smoke test")
    ap.add_argument("--max-train-residues", type=int, default=60000)
    ap.add_argument("--split-level", choices=["protein", "fold", "superfamily", "family"],
                    default="fold", help="train/test split unit: fold-disjoint (default) or protein")
    ap.add_argument("--fasta", default="cache/scope_40.fa")
    ap.add_argument("--save-dir", required=True)
    args = ap.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    layer_dir = Path(args.layer_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    Z, uids, lengths = load_layer(layer_dir)
    uids = [str(u) for u in uids]
    ref = load_ref_seqs(layer_dir)
    sequences = [ref[u] for u in uids]

    if args.max_proteins and args.max_proteins < len(uids):
        uids = uids[:args.max_proteins]
        sequences = sequences[:args.max_proteins]
        lengths = lengths[:args.max_proteins]
        Z = Z[:int(sum(int(l) for l in lengths)), :]

    offsets, total = {}, 0
    for u, L in zip(uids, lengths):
        offsets[u] = total
        total += int(L)

    df_phys = load_phys_features(Path(args.features_csv))
    labels = build_labels(uids, lengths, offsets, df_phys)
    if labels is None:
        print("  No labels available; aborting.")
        return

    # avoid polluting the full-run cache during subsampled smoke tests
    raw = get_raw_embeddings(layer_dir, args.model, args.layer, uids, sequences,
                             cache=not bool(args.max_proteins))
    raw = np.asarray(raw[:total], dtype=np.float32)
    Zf = np.asarray(Z[:total], dtype=np.float32)

    # train/test split (70/30 by protein), fold-disjoint by default to close
    # fold-level leakage (a memorised fold would inflate both raw and SAE probes).
    rng = np.random.default_rng(42)
    if args.split_level != "protein" and Path(args.fasta).exists():
        from cluster_bootstrap import load_uid_clusters
        cl = load_uid_clusters(args.fasta, level=args.split_level)
        by_fold = {}
        for u in uids:
            by_fold.setdefault(cl.get(u, f"__singleton__{u}"), []).append(u)
        folds = list(by_fold); rng.shuffle(folds)
        target = int(0.7 * len(uids)); tr_uids, n = set(), 0
        for f in folds:
            if n < target:
                tr_uids.update(by_fold[f]); n += len(by_fold[f])
    else:
        order = rng.permutation(len(uids))
        tr_uids = {uids[i] for i in order[:int(0.7 * len(uids))]}
    tr_mask = np.zeros(total, dtype=bool)
    for u, L in zip(uids, lengths):
        if u in tr_uids:
            tr_mask[offsets[u]:offsets[u] + int(L)] = True
    te_mask = ~tr_mask

    # standardize raw (z-score on train); SAE features kept raw (already sparse/scaled)
    mu = raw[tr_mask].mean(0, keepdims=True)
    sd = raw[tr_mask].std(0, keepdims=True) + 1e-6
    raw_z = (raw - mu) / sd

    rows = []
    for task in ["helix", "strand", "burial"]:
        y = labels[task].astype(int)
        vmask = labels["valid_burial"] if task == "burial" else np.ones(total, bool)
        tr = tr_mask & vmask
        te = te_mask & vmask
        # subsample train residues for speed
        tr_idx = np.where(tr)[0]
        if len(tr_idx) > args.max_train_residues:
            tr_idx = rng.choice(tr_idx, args.max_train_residues, replace=False)
        te_idx = np.where(te)[0]
        for feat_name, X in [("raw", raw_z), ("sae", Zf)]:
            f1, auc = probe(X[tr_idx], y[tr_idx], X[te_idx], y[te_idx])
            rows.append({"task": task, "features": feat_name, "f1": f1, "auroc": auc,
                         "n_train": len(tr_idx), "n_test": len(te_idx)})
            print(f"  {task:8s} {feat_name:4s}  F1={f1:.3f}  AUROC={auc:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(save_dir / "probe_results.csv", index=False)

    # plot
    fig, ax = plt.subplots(figsize=(7, 4))
    tasks = ["helix", "strand", "burial"]
    x = np.arange(len(tasks))
    w = 0.35
    raw_auc = [df[(df.task == t) & (df.features == "raw")]["auroc"].values[0] for t in tasks]
    sae_auc = [df[(df.task == t) & (df.features == "sae")]["auroc"].values[0] for t in tasks]
    ax.bar(x - w / 2, raw_auc, w, label="raw activations", color="#888")
    ax.bar(x + w / 2, sae_auc, w, label="SAE features", color="#2196F3")
    ax.set_xticks(x); ax.set_xticklabels(tasks)
    ax.set_ylabel("test AUROC"); ax.set_ylim(0.5, 1.0)
    ax.set_title(f"Linear probe: raw vs SAE ({args.model} L{args.layer})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_dir / "probe_raw_vs_sae.png", dpi=200)
    plt.close(fig)

    summary = {"model": args.model, "layer": args.layer, "split_level": args.split_level,
               "results": rows, "n_proteins": len(uids)}
    (save_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n  Saved to {save_dir}/")


if __name__ == "__main__":
    main()
