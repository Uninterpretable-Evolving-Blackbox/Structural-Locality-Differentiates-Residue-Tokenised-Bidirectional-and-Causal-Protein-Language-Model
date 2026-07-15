#!/usr/bin/env python3
"""
experiment_concept_f1.py — InterPLM-style concept alignment for SAE features
============================================================================

Grounds SAE features in *curated biological concepts* using the domain-aware
precision/recall/F1 framework of InterPLM (Simon & Zou, Nature Methods 2025):

    precision = TruePositives / (TruePositives + FalsePositives)   # per amino acid
    recall    = DomainsWithTruePositive / TotalDomains             # per domain
    F1        = 2 * precision * recall / (precision + recall)

Why domain-aware recall? A feature that fires on 2 conserved positions of a
200-residue domain should not be penalised for "missing" the other 198 — it
still detects the domain. Standard residue-level recall over-penalises features
that are more specific than the annotation (InterPLM §Methods).

Concept sources (all derivable from artefacts already in this repo):
  - SCOPe class / fold / superfamily / family   (cache/scope_40.fa)   [domain = SCOPe domain = protein]
  - DSSP secondary structure (helix/strand/...)  (residue_features.csv) [domain = contiguous SS segment]
  - Relative solvent accessibility bins          (residue_features.csv) [domain = contiguous RSA segment]

Protocol (InterPLM): split proteins into a concept-VAL set (select the single
best feature + activation threshold per concept) and a concept-TEST set (report
that feature's F1). Count features with F1 > 0.5 to any concept in BOTH splits.

Headline cross-model use: run for ESM-2 and RITA at matched depths, then compare
the number of concept-aligned features and the mean best-F1 per concept. Also
merges with struct_seq_metrics.csv to test whether the homemade L_struct metric
correlates with concept-F1 (i.e. whether L_struct tracks real biology).

Usage:
  python experiment_concept_f1.py \
    --layer-dir outputs_layerwise/esm2/layer_16 \
    --fasta-path cache/scope_40.fa \
    --features-csv cache/residue_features.csv \
    --save-dir results_concept_f1/esm2_layer16

  # quick smoke test on a feature subset
  python experiment_concept_f1.py --layer-dir outputs_layerwise/esm2/layer_16 \
    --save-dir /tmp/cf1_smoke --max-features 512 --quick
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
from scipy import sparse
from scipy.stats import spearmanr

from cpu_stage import load_layer, load_ref_seqs, load_phys_features

warnings.filterwarnings("ignore")

DEFAULT_THRESHOLDS = [0.0, 0.15, 0.5, 0.6, 0.8]
SS_GROUPS = {
    "ss_helix": {"H", "G", "I"},
    "ss_strand": {"E", "B"},
    "ss_turn": {"T", "S"},
    "ss_coil": {"-", "C", "", " "},
}


# =====================================================================
#   Residue index + protein-level concept-eval split
# =====================================================================

def build_offsets(uids, lengths):
    """res_global offset per uid (Z rows are concatenated in uids order)."""
    offsets, off = {}, 0
    for uid, L in zip(uids, lengths):
        offsets[str(uid)] = off
        off += int(L)
    return offsets, off


def split_proteins(uids, seed=42, val_frac=0.5, fasta_path=None, level="fold"):
    """Deterministic split into (val, test) for concept eval.

    FOLD-DISJOINT (default, when a SCOPe FASTA is given): whole folds are
    assigned entirely to val OR test, so no fold -- and therefore no close
    structural homolog -- appears in both splits. This closes fold-level
    leakage (a fold memorised by a feature would otherwise inflate both the
    val-selected threshold and the test F1). Falls back to a protein-level
    random split when level='protein' or no FASTA is available.
    """
    rng = np.random.default_rng(seed)
    uids = [str(u) for u in uids]
    if fasta_path is None or level == "protein" or not Path(fasta_path).exists():
        order = rng.permutation(len(uids))
        n_val = int(round(len(uids) * val_frac))
        return ({uids[i] for i in order[:n_val]}, {uids[i] for i in order[n_val:]})
    from cluster_bootstrap import load_uid_clusters
    cl = load_uid_clusters(fasta_path, level=level)
    by_fold = {}
    for u in uids:
        by_fold.setdefault(cl.get(u, f"__singleton__{u}"), []).append(u)
    folds = list(by_fold)
    rng.shuffle(folds)
    target = int(round(len(uids) * val_frac))
    val_uids, n = set(), 0
    for f in folds:
        if n < target:
            val_uids.update(by_fold[f]); n += len(by_fold[f])
    test_uids = {u for u in uids if u not in val_uids}
    return val_uids, test_uids


# =====================================================================
#   Concept construction
# =====================================================================

def parse_scope_concepts(fasta_path: Path, levels):
    """uid -> {level: concept_string} from SCOPe sccs codes (e.g. a.1.1.1)."""
    level_depth = {"class": 1, "fold": 2, "superfamily": 3, "family": 4}
    uid_concepts = {}
    with open(fasta_path) as f:
        for line in f:
            if not line.startswith(">"):
                continue
            parts = line[1:].split()
            if len(parts) < 2:
                continue
            uid = parts[0]
            sccs = parts[1].split(".")
            d = {}
            for lv in levels:
                depth = level_depth[lv]
                if len(sccs) >= depth:
                    d[lv] = f"{lv}:" + ".".join(sccs[:depth])
            uid_concepts[uid] = d
    return uid_concepts


def build_concept_membership(uids, lengths, offsets, df_phys, uid_concepts,
                             levels, include_ss, include_rsa,
                             rsa_buried=0.1, rsa_exposed=0.4):
    """Build, for each concept, the residue list + per-residue domain id.

    Returns dict: concept -> (res_global array, domain_id array (int)).
    domain id is an arbitrary unique integer per domain (protein or segment).
    """
    concepts = {}
    next_dom = [0]

    def add_protein_concept(concept, uid, base, L):
        res = base + np.arange(L, dtype=np.int64)
        dom = np.full(L, next_dom[0], dtype=np.int64)
        next_dom[0] += 1
        if concept not in concepts:
            concepts[concept] = [[], []]
        concepts[concept][0].append(res)
        concepts[concept][1].append(dom)

    # --- SCOPe protein-level concepts (domain = protein) ---
    for uid, L in zip(uids, lengths):
        uid = str(uid)
        base = offsets[uid]
        L = int(L)
        cdict = uid_concepts.get(uid, {})
        for lv in levels:
            c = cdict.get(lv)
            if c is not None:
                add_protein_concept(c, uid, base, L)

    # --- Residue-level concepts (domain = contiguous segment) ---
    if (include_ss or include_rsa) and df_phys is not None and len(df_phys):
        phys = df_phys.set_index(["uid", "position"]) if "position" in df_phys.columns else None
        # Build per-protein arrays of SS / RSA in residue order
        by_uid = {u: g.sort_values("position") for u, g in df_phys.groupby("uid")} \
            if "uid" in df_phys.columns else {}
        for uid, L in zip(uids, lengths):
            uid = str(uid)
            base = offsets[uid]
            L = int(L)
            g = by_uid.get(uid)
            if g is None or len(g) == 0:
                continue
            pos = g["position"].to_numpy().astype(int)
            ok = (pos >= 0) & (pos < L)
            pos = pos[ok]
            if include_ss and "ss_8class" in g.columns:
                ss = g["ss_8class"].astype(str).to_numpy()[ok]
                for cname, members in SS_GROUPS.items():
                    mask = np.array([s in members for s in ss], dtype=bool)
                    _add_segments(concepts, cname, base, pos, mask, next_dom)
            if include_rsa and "sasa" in g.columns:
                rsa = g["sasa"].astype(float).to_numpy()[ok]
                _add_segments(concepts, "rsa_buried", base, pos, rsa < rsa_buried, next_dom)
                _add_segments(concepts, "rsa_exposed", base, pos, rsa > rsa_exposed, next_dom)

    # Concatenate per-concept lists into arrays
    out = {}
    for c, (res_lists, dom_lists) in concepts.items():
        res = np.concatenate(res_lists)
        dom = np.concatenate(dom_lists)
        out[c] = (res, dom)
    return out


def _add_segments(concepts, cname, base, pos, mask, next_dom):
    """Add contiguous True runs in `mask` (over positions `pos`) as domains."""
    if not mask.any():
        return
    if cname not in concepts:
        concepts[cname] = [[], []]
    # contiguous runs over the (sorted) positions
    idx = np.where(mask)[0]
    # split into runs where position is consecutive
    splits = np.where(np.diff(pos[idx]) != 1)[0] + 1
    for run in np.split(idx, splits):
        if run.size == 0:
            continue
        res = base + pos[run].astype(np.int64)
        dom = np.full(run.size, next_dom[0], dtype=np.int64)
        next_dom[0] += 1
        concepts[cname][0].append(res)
        concepts[cname][1].append(dom)


# =====================================================================
#   Per-feature normalisation (streamed)
# =====================================================================

def feature_maxima(Z, feat_idx, chunk=1024):
    """Per-feature max activation over all residues (for [0,1] normalisation)."""
    n = len(feat_idx)
    maxima = np.zeros(n, dtype=np.float32)
    for s in range(0, n, chunk):
        cols = feat_idx[s:s + chunk]
        block = np.asarray(Z[:, cols], dtype=np.float32)
        maxima[s:s + chunk] = block.max(axis=0)
    maxima[maxima <= 0] = 1.0  # dead features -> avoid div0 (they stay all-zero)
    return maxima


# =====================================================================
#   Core: precision / recall / F1 for a feature chunk vs all concepts
# =====================================================================

def eval_split_prf(Z, feat_idx, maxima, concepts, split_res_mask,
                   thresholds, chunk=1024):
    """For each (feature, concept) return best-threshold F1/prec/recall on a split.

    Returns dict concept -> DataFrame[feature_local, f1, precision, recall, threshold]
    where feature_local indexes into feat_idx.
    """
    n_feat = len(feat_idx)
    # map global residue id -> local split row; -1 if not in split
    split_res = np.where(split_res_mask)[0]
    n_split = split_res.size
    g2l = -np.ones(Z.shape[0], dtype=np.int64)
    g2l[split_res] = np.arange(n_split)

    # Precompute per-concept split-local membership + domain matrix
    concept_local = {}
    for c, (res, dom) in concepts.items():
        loc = g2l[res]
        keep = loc >= 0
        if keep.sum() == 0:
            continue
        loc = loc[keep]
        dom_k = dom[keep]
        uniq_dom, dom_idx = np.unique(dom_k, return_inverse=True)
        n_dom = uniq_dom.size
        Dmat = sparse.csr_matrix(
            (np.ones(loc.size, dtype=np.float32), (dom_idx, loc)),
            shape=(n_dom, n_split))
        concept_local[c] = {"loc": loc, "Dmat": Dmat, "n_dom": n_dom}

    # best F1 per (concept, feature)
    best = {c: {"f1": np.zeros(n_feat, dtype=np.float32),
                "prec": np.zeros(n_feat, dtype=np.float32),
                "rec": np.zeros(n_feat, dtype=np.float32),
                "thr": np.zeros(n_feat, dtype=np.float32)}
            for c in concept_local}

    for s in range(0, n_feat, chunk):
        cols = feat_idx[s:s + chunk]
        cs = len(cols)
        block = np.asarray(Z[:, cols], dtype=np.float32)[split_res, :]  # (n_split, cs)
        block /= maxima[s:s + cs][None, :]
        for t in thresholds:
            active = block > t if t > 0 else block > 0
            active_total = active.sum(axis=0).astype(np.float32)  # (cs,)
            active_total_safe = np.where(active_total > 0, active_total, 1.0)
            active_sp = sparse.csr_matrix(active.astype(np.float32))  # (n_split, cs)
            for c, cl in concept_local.items():
                loc = cl["loc"]
                # TP per feature = # concept residues active
                tp = active[loc, :].sum(axis=0).astype(np.float32)
                prec = tp / active_total_safe
                # recall per domain
                dom_has = (cl["Dmat"] @ active_sp)  # (n_dom, cs) sparse
                dom_has = (dom_has > 0).sum(axis=0)
                rec = np.asarray(dom_has).ravel().astype(np.float32) / cl["n_dom"]
                denom = prec + rec
                f1 = np.where(denom > 0, 2 * prec * rec / np.where(denom > 0, denom, 1.0), 0.0)
                # update best
                improve = f1 > best[c]["f1"][s:s + cs]
                bf = best[c]
                bf["f1"][s:s + cs] = np.where(improve, f1, bf["f1"][s:s + cs])
                bf["prec"][s:s + cs] = np.where(improve, prec, bf["prec"][s:s + cs])
                bf["rec"][s:s + cs] = np.where(improve, rec, bf["rec"][s:s + cs])
                bf["thr"][s:s + cs] = np.where(improve, t, bf["thr"][s:s + cs])
    return best, concept_local


def f1_at(Z, feat_local, feat_idx, maxima, concept_entry, threshold, split_res_mask):
    """Compute F1 for ONE feature at a FIXED threshold on a split (for test eval)."""
    split_res = np.where(split_res_mask)[0]
    n_split = split_res.size
    g2l = -np.ones(Z.shape[0], dtype=np.int64)
    g2l[split_res] = np.arange(n_split)
    col = feat_idx[feat_local]
    vec = np.asarray(Z[:, col], dtype=np.float32)[split_res] / maxima[feat_local]
    active = vec > threshold if threshold > 0 else vec > 0
    loc = concept_entry["loc_test"]
    if active.sum() == 0 or loc.size == 0:
        return 0.0, 0.0, 0.0
    tp = float(active[loc].sum())
    prec = tp / float(active.sum())
    Dmat = concept_entry["Dmat_test"]
    dom_has = (Dmat @ sparse.csr_matrix(active.astype(np.float32).reshape(-1, 1)))
    rec = float((dom_has.toarray().ravel() > 0).sum()) / concept_entry["n_dom_test"]
    denom = prec + rec
    f1 = 2 * prec * rec / denom if denom > 0 else 0.0
    return f1, prec, rec


# =====================================================================
#                              MAIN
# =====================================================================

def main():
    ap = argparse.ArgumentParser(description="InterPLM-style concept-F1 for SAE features")
    ap.add_argument("--layer-dir", required=True)
    ap.add_argument("--fasta-path", default="cache/scope_40.fa")
    ap.add_argument("--features-csv", default="cache/residue_features.csv")
    ap.add_argument("--struct-csv", default=None,
                    help="struct_seq_metrics.csv (defaults to layer-dir/struct_seq_metrics.csv)")
    ap.add_argument("--levels", default="class,fold,superfamily,family")
    ap.add_argument("--include-ss", action="store_true", default=True)
    ap.add_argument("--no-ss", dest="include_ss", action="store_false")
    ap.add_argument("--include-rsa", action="store_true", default=True)
    ap.add_argument("--no-rsa", dest="include_rsa", action="store_false")
    ap.add_argument("--min-domains", type=int, default=10,
                    help="Min domains for a concept to be evaluated (InterPLM uses >10)")
    ap.add_argument("--thresholds", default="0,0.15,0.5,0.6,0.8")
    ap.add_argument("--val-frac", type=float, default=0.5)
    ap.add_argument("--split-level", choices=["protein", "fold", "superfamily", "family"],
                    default="fold",
                    help="concept val/test split unit: fold-disjoint (default) closes "
                         "fold-level leakage; 'protein' = original random split")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-features", type=int, default=0,
                    help="0 = all features; >0 to subsample for smoke tests")
    ap.add_argument("--quick", action="store_true", help="fewer thresholds for smoke test")
    ap.add_argument("--save-dir", required=True)
    args = ap.parse_args()

    layer_dir = Path(args.layer_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    levels = [x.strip() for x in args.levels.split(",") if x.strip()]
    thresholds = [float(x) for x in args.thresholds.split(",")]
    if args.quick:
        thresholds = [0.0, 0.5]

    print("=" * 70)
    print("  CONCEPT-F1 (InterPLM-style domain-aware alignment)")
    print("=" * 70)

    Z, uids, lengths = load_layer(layer_dir)
    uids = [str(u) for u in uids]
    offsets, total_res = build_offsets(uids, lengths)
    n_features = int(Z.shape[1])
    feat_idx = np.arange(n_features, dtype=np.int64)
    if args.max_features and args.max_features < n_features:
        feat_idx = feat_idx[:args.max_features]
    print(f"  Z: {Z.shape} | features evaluated: {len(feat_idx)} | residues: {total_res}")

    df_phys = load_phys_features(Path(args.features_csv)) if Path(args.features_csv).exists() else None

    uid_concepts = parse_scope_concepts(Path(args.fasta_path), levels) \
        if Path(args.fasta_path).exists() else {}

    concepts = build_concept_membership(
        uids, lengths, offsets, df_phys, uid_concepts,
        levels, args.include_ss, args.include_rsa)

    # Filter concepts by min domains
    concepts = {c: v for c, v in concepts.items()
                if np.unique(v[1]).size >= args.min_domains}
    print(f"  Concepts evaluated (>= {args.min_domains} domains): {len(concepts)}")
    if not concepts:
        print("  No concepts pass the support filter; aborting.")
        return

    # Concept val/test split (fold-disjoint by default; closes fold-level leakage)
    val_uids, test_uids = split_proteins(uids, seed=args.seed, val_frac=args.val_frac,
                                         fasta_path=args.fasta_path, level=args.split_level)
    val_res_mask = np.zeros(Z.shape[0], dtype=bool)
    test_res_mask = np.zeros(Z.shape[0], dtype=bool)
    for uid, L in zip(uids, lengths):
        base = offsets[uid]
        if uid in val_uids:
            val_res_mask[base:base + int(L)] = True
        else:
            test_res_mask[base:base + int(L)] = True

    maxima = feature_maxima(Z, feat_idx)

    print("  [1/3] Selecting best feature+threshold per concept on VAL split...")
    best_val, concept_local_val = eval_split_prf(
        Z, feat_idx, maxima, concepts, val_res_mask, thresholds)

    # Precompute test-split concept matrices
    print("  [2/3] Building TEST-split concept matrices...")
    test_res = np.where(test_res_mask)[0]
    n_test = test_res.size
    g2l_test = -np.ones(Z.shape[0], dtype=np.int64)
    g2l_test[test_res] = np.arange(n_test)
    concept_test = {}
    for c, (res, dom) in concepts.items():
        loc = g2l_test[res]
        keep = loc >= 0
        if keep.sum() == 0:
            continue
        loc = loc[keep]
        dom_k = dom[keep]
        uniq, dom_idx = np.unique(dom_k, return_inverse=True)
        Dmat = sparse.csr_matrix(
            (np.ones(loc.size, dtype=np.float32), (dom_idx, loc)),
            shape=(uniq.size, n_test))
        concept_test[c] = {"loc_test": loc, "Dmat_test": Dmat, "n_dom_test": uniq.size}

    print("  [3/3] Scoring selected features on TEST split...")
    rows = []
    for c, bv in best_val.items():
        if c not in concept_test:
            continue
        f_local = int(np.argmax(bv["f1"]))
        val_f1 = float(bv["f1"][f_local])
        thr = float(bv["thr"][f_local])
        test_f1, test_p, test_r = f1_at(
            Z, f_local, feat_idx, maxima, concept_test[c], thr, test_res_mask)
        rows.append({
            "concept": c,
            "concept_kind": c.split(":")[0],
            "feature_idx": int(feat_idx[f_local]),
            "threshold": thr,
            "val_f1": val_f1,
            "val_precision": float(bv["prec"][f_local]),
            "val_recall": float(bv["rec"][f_local]),
            "test_f1": test_f1,
            "test_precision": test_p,
            "test_recall": test_r,
            "n_domains": int(np.unique(concepts[c][1]).size),
        })
    concept_df = pd.DataFrame(rows).sort_values("test_f1", ascending=False)
    concept_df.to_csv(save_dir / "concept_f1.csv", index=False)

    # Per-feature best concept (val) + count F1>0.5 in both splits
    feat_best_val = np.zeros(len(feat_idx), dtype=np.float32)
    feat_best_concept = np.array([""] * len(feat_idx), dtype=object)
    feat_best_thr = np.zeros(len(feat_idx), dtype=np.float32)
    for c, bv in best_val.items():
        improve = bv["f1"] > feat_best_val
        feat_best_val = np.where(improve, bv["f1"], feat_best_val)
        feat_best_thr = np.where(improve, bv["thr"], feat_best_thr)
        feat_best_concept[improve] = c

    # test F1 for each feature's best concept at the selected threshold
    feat_test_f1 = np.zeros(len(feat_idx), dtype=np.float32)
    for fl in range(len(feat_idx)):
        c = feat_best_concept[fl]
        if c == "" or c not in concept_test:
            continue
        tf1, _, _ = f1_at(Z, fl, feat_idx, maxima, concept_test[c],
                          float(feat_best_thr[fl]), test_res_mask)
        feat_test_f1[fl] = tf1

    feat_df = pd.DataFrame({
        "feature_idx": feat_idx,
        "best_concept": feat_best_concept,
        "best_threshold": feat_best_thr,
        "val_f1": feat_best_val,
        "test_f1": feat_test_f1,
    })
    feat_df.to_csv(save_dir / "feature_concept_best.csv", index=False)

    n_aligned = int(((feat_best_val > 0.5) & (feat_test_f1 > 0.5)).sum())

    # Correlate concept-F1 with L_struct (validates the homemade metric)
    struct_corr = None
    struct_path = Path(args.struct_csv) if args.struct_csv else (layer_dir / "struct_seq_metrics.csv")
    if struct_path.exists():
        sdf = pd.read_csv(struct_path)
        key = "feature_idx" if "feature_idx" in sdf.columns else sdf.columns[0]
        if "struct_delta" in sdf.columns:
            merged = feat_df.merge(sdf[[key, "struct_delta"]].rename(columns={key: "feature_idx"}),
                                   on="feature_idx", how="inner")
            if len(merged) > 10:
                rho, p = spearmanr(merged["val_f1"], merged["struct_delta"])
                struct_corr = {"spearman_rho": float(rho), "p": float(p), "n": int(len(merged))}
                fig, ax = plt.subplots(figsize=(6, 5))
                ax.scatter(merged["struct_delta"], merged["val_f1"], s=5, alpha=0.3)
                ax.set_xlabel("L_struct (struct_delta)")
                ax.set_ylabel("best concept F1 (val)")
                ax.set_title(f"Concept-F1 vs L_struct  (Spearman {rho:.3f})")
                fig.tight_layout()
                fig.savefig(save_dir / "conceptF1_vs_Lstruct.png", dpi=200)
                plt.close(fig)

    summary = {
        "layer_dir": str(layer_dir),
        "n_features_evaluated": int(len(feat_idx)),
        "n_concepts": int(len(concept_df)),
        "n_concepts_testF1_gt_0.5": int((concept_df["test_f1"] > 0.5).sum()),
        "mean_top_test_f1_per_concept": float(concept_df["test_f1"].mean()),
        "n_features_aligned_F1_gt_0.5_both": n_aligned,
        "split_level": args.split_level,
        "struct_corr": struct_corr,
    }
    (save_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n  Summary:")
    print(json.dumps(summary, indent=2))
    print(f"\n  Top concepts by test F1:")
    print(concept_df.head(12).to_string(index=False))
    print(f"\n  Saved to {save_dir}/")


if __name__ == "__main__":
    main()
