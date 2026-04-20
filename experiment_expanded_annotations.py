#!/usr/bin/env python3
"""
experiment_expanded_annotations.py — Richer interpretability ground truth
=========================================================================

Addresses Limitation 3: binary DSSP annotations (helix, strand, burial) are
too coarse, inflating the 99.6% interpretability rate and missing functional
nuance.

Three expansions:
  1. Continuous RSA: Replace binary burial with continuous Relative Solvent
     Accessibility from DSSP. Use Spearman rank correlation to test whether
     SAE features scale with exposure.

  2. Functional site annotations: Map SCOPe UIDs to UniProt accessions via
     the SIFTS resource, then query UniProt for:
       - Metal-binding sites
       - Active/catalytic sites
       - Disulfide bonds
       - Binding sites (ligand, DNA, etc.)
     These are sparse, specific annotations — a much harder test.

  3. Extended FDR pipeline: Run the same Benjamini–Hochberg pipeline from
     cpu_stage.py against the new annotations, measuring what fraction of
     SAE features correlate with specific biochemical functions.

Usage:
  python experiment_expanded_annotations.py \
    --layer-dir outputs_layerwise/esm2/layer_16 \
    --model-type residue \
    --pdb-dir cache/pdb_files \
    --save-dir results_expanded_annotations

  python experiment_expanded_annotations.py \
    --layer-dir outputs_layerwise/protgpt2/layer_18 \
    --model-type protgpt2 \
    --save-dir results_expanded_annotations_gpt2

Prerequisites:
  - Z.npy, uids.json, sequences.json, lengths.npy in layer-dir
  - PDB files for DSSP RSA extraction
  - Internet access for UniProt API queries (or pre-cached mapping)
"""

import argparse
import json
import time
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import spearmanr
from scipy.stats import t as tdist
from joblib import Parallel, delayed, cpu_count

from cpu_stage import (
    load_layer, load_ref_seqs, load_phys_features,
    build_residue_index, build_protgpt2_projection,
    bh_fdr, fill_ss8_from_dssp, have_dssp,
    DEFAULT_TOPK_FRAC,
)

warnings.filterwarnings("ignore")


# =====================================================================
#     1. CONTINUOUS RSA FROM DSSP
# =====================================================================

def extract_rsa_from_dssp(df_phys: pd.DataFrame, pdb_dir: Path,
                          n_jobs: int = -1) -> pd.DataFrame:
    """Ensure RSA (sasa column) is populated from DSSP.
    Returns updated df_phys with float 'rsa' column.
    """
    df = df_phys.copy()

    # fill_ss8_from_dssp already extracts RSA into 'sasa' column
    if "sasa" not in df.columns or df["sasa"].isna().all() or (df["sasa"] == 0).all():
        if have_dssp():
            print("  Running DSSP to extract RSA...")
            df = fill_ss8_from_dssp(df, pdb_dir, n_jobs=n_jobs)
        else:
            print("  WARNING: DSSP not available, RSA will be zeros")

    # Rename for clarity
    if "sasa" in df.columns:
        df["rsa"] = df["sasa"].astype(np.float32)
    else:
        df["rsa"] = 0.0

    return df


# =====================================================================
#     2. UNIPROT FUNCTIONAL SITE MAPPING
# =====================================================================

def map_scope_to_uniprot_sifts(uids: list, pdb_dir: Path) -> dict:
    """Map SCOPe UIDs (e.g., 'd1a2ba_') to UniProt accessions via SIFTS.

    SCOPe UIDs encode PDB ID at positions 1-4, so we extract PDB codes
    and query SIFTS for UniProt mappings.

    Returns dict: {uid: uniprot_accession} for successful mappings.
    """
    import urllib.request
    import csv
    import io

    # Extract unique PDB codes from SCOPe UIDs
    pdb_to_uids = defaultdict(list)
    for uid in uids:
        pdb_code = str(uid)[1:5].lower()
        pdb_to_uids[pdb_code].append(uid)

    print(f"  Mapping {len(pdb_to_uids)} PDB codes to UniProt via SIFTS...")

    # Download SIFTS PDB-UniProt mapping
    sifts_url = ("https://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/"
                 "csv/pdb_chain_uniprot.csv.gz")

    uid_to_uniprot = {}

    try:
        import gzip
        print(f"  Downloading SIFTS mapping...")
        req = urllib.request.Request(sifts_url)
        req.add_header("User-Agent", "Python-urllib/BIOL0044-SAE-project")
        response = urllib.request.urlopen(req, timeout=60)
        data = gzip.decompress(response.read()).decode("utf-8")

        reader = csv.DictReader(io.StringIO(data))
        for row in reader:
            pdb = row.get("PDB", "").lower()
            if pdb in pdb_to_uids:
                uniprot = row.get("SP_PRIMARY", "")
                if uniprot:
                    for uid in pdb_to_uids[pdb]:
                        uid_to_uniprot[uid] = uniprot

        print(f"  Mapped {len(uid_to_uniprot)} / {len(uids)} UIDs to UniProt")

    except Exception as e:
        print(f"  WARNING: SIFTS download failed ({e})")
        print(f"  Trying per-PDB UniProt lookup instead...")

        # Fallback: query PDB REST API per PDB code
        for pdb_code in tqdm(list(pdb_to_uids.keys())[:100],
                              desc="  PDB→UniProt"):
            try:
                url = (f"https://data.rcsb.org/rest/v1/core/uniprot/"
                       f"{pdb_code}/1")
                req = urllib.request.Request(url)
                req.add_header("User-Agent", "Python-urllib/BIOL0044")
                resp = urllib.request.urlopen(req, timeout=10)
                data = json.loads(resp.read().decode())
                if isinstance(data, list) and data:
                    acc = data[0].get("rcsb_uniprot_container_identifiers", {}).get(
                        "uniprot_id", "")
                    if acc:
                        for uid in pdb_to_uids[pdb_code]:
                            uid_to_uniprot[uid] = acc
                time.sleep(0.1)  # rate limiting
            except Exception:
                continue

        print(f"  Mapped {len(uid_to_uniprot)} / {len(uids)} UIDs via REST")

    return uid_to_uniprot


def fetch_uniprot_features(accession: str) -> list:
    """Fetch functional features for a UniProt accession.

    Returns list of dicts: [{type, start, end, description}, ...]
    """
    import urllib.request

    url = f"https://rest.uniprot.org/uniprotkb/{accession}.json"
    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "Python-urllib/BIOL0044-SAE-project")
        resp = urllib.request.urlopen(req, timeout=15)
        data = json.loads(resp.read().decode())
    except Exception:
        return []

    features = []
    for feat in data.get("features", []):
        ftype = feat.get("type", "")
        loc = feat.get("location", {})
        start = loc.get("start", {}).get("value")
        end = loc.get("end", {}).get("value")
        desc = feat.get("description", "")

        if start is not None and end is not None:
            features.append({
                "type": ftype,
                "start": int(start),
                "end": int(end),
                "description": desc,
            })

    return features


# Target feature types for interpretability
FUNCTIONAL_TYPES = {
    "Metal binding": "metal_binding",
    "Active site": "active_site",
    "Binding site": "binding_site",
    "Disulfide bond": "disulfide",
}


def build_functional_annotations(uids, sequences, uid_to_uniprot,
                                  max_queries=500):
    """Build per-residue functional annotation vectors.

    Returns DataFrame with columns: uid, position, metal_binding, active_site,
    binding_site, disulfide (all binary 0/1).
    """
    rows = []
    queried = 0

    # Deduplicate UniProt accessions
    acc_to_uids = defaultdict(list)
    for uid in uids:
        acc = uid_to_uniprot.get(uid)
        if acc:
            acc_to_uids[acc].append(uid)

    print(f"  Fetching functional features for {len(acc_to_uids)} "
          f"UniProt accessions (max {max_queries})...")

    acc_features_cache = {}
    for acc in tqdm(list(acc_to_uids.keys())[:max_queries],
                     desc="  UniProt features"):
        feats = fetch_uniprot_features(acc)
        acc_features_cache[acc] = feats
        queried += 1
        if queried % 50 == 0:
            time.sleep(1.0)  # rate limiting

    # Map features to residue positions
    for uid_idx, (uid, seq) in enumerate(zip(uids, sequences)):
        L = len(seq)
        acc = uid_to_uniprot.get(uid)
        feats = acc_features_cache.get(acc, []) if acc else []

        # Initialize binary annotations per residue
        annotations = {v: np.zeros(L, dtype=np.int8) for v in FUNCTIONAL_TYPES.values()}

        for feat in feats:
            ftype = feat["type"]
            col = FUNCTIONAL_TYPES.get(ftype)
            if col is None:
                continue
            # UniProt uses 1-based positions
            start = feat["start"] - 1  # convert to 0-based
            end = feat["end"]  # exclusive
            start = max(0, start)
            end = min(L, end)
            annotations[col][start:end] = 1

        for pos in range(L):
            row = {"uid": uid, "position": pos}
            for col in FUNCTIONAL_TYPES.values():
                row[col] = int(annotations[col][pos])
            rows.append(row)

    df = pd.DataFrame(rows)

    # Report annotation density
    for col in FUNCTIONAL_TYPES.values():
        n_pos = (df[col] == 1).sum()
        pct = 100 * n_pos / len(df) if len(df) > 0 else 0
        print(f"    {col}: {n_pos} residues ({pct:.2f}%)")

    return df


# =====================================================================
#     3. EXTENDED CORRELATION PIPELINE
# =====================================================================

def spearman_with_pvals(Z_chunk: np.ndarray, y: np.ndarray):
    """Spearman correlation + p-values for a chunk of features vs continuous y."""
    n_feat = Z_chunk.shape[1]
    rhos = np.zeros(n_feat, dtype=np.float32)
    pvals = np.ones(n_feat, dtype=np.float32)

    for j in range(n_feat):
        col = Z_chunk[:, j]
        if col.std() < 1e-8:
            continue
        rho, p = spearmanr(col, y)
        rhos[j] = rho
        pvals[j] = p

    return rhos, pvals


def pearson_with_pvals(Z_chunk: np.ndarray, y: np.ndarray):
    """Pearson correlation + p-values for binary annotations (same as cpu_stage)."""
    y = y.astype(np.float32)
    n = float(y.shape[0])
    y0 = y - y.mean()
    y_norm = np.linalg.norm(y0)
    if y_norm == 0:
        return (np.zeros(Z_chunk.shape[1], dtype=np.float32),
                np.ones(Z_chunk.shape[1], dtype=np.float32))
    Zm = Z_chunk.astype(np.float32)
    z0 = Zm - Zm.mean(axis=0, keepdims=True)
    z_norm = np.linalg.norm(z0, axis=0)
    z_norm[z_norm == 0] = 1.0
    r = (z0.T @ y0) / (z_norm * y_norm)
    r = np.clip(r, -0.999999, 0.999999).astype(np.float32)
    df = max(int(n) - 2, 1)
    t = np.abs(r) * np.sqrt(df / (1.0 - r * r))
    p = (2.0 * tdist.sf(t, df=df)).astype(np.float32)
    return r, p


def run_extended_correlation(Z, A_proj, df_merged, res_idx, save_dir: Path,
                              n_jobs: int = -1):
    """Run correlation pipeline with extended annotations."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    dfm = df_merged.merge(res_idx, on=["uid", "position"], how="inner")
    if len(dfm) == 0:
        print("  WARNING: No residues aligned for extended correlation")
        return

    ridx = dfm["res_global"].astype(int).to_numpy()
    n_features = int(Z.shape[1])

    print(f"  Running extended correlations on {len(dfm)} residues, "
          f"{n_features} features...")

    # Get Z at residue positions
    chunk_size = 512
    n_chunks = (n_features + chunk_size - 1) // chunk_size

    results = {}

    # --- Continuous RSA (Spearman) ---
    if "rsa" in dfm.columns and dfm["rsa"].std() > 1e-6:
        rsa = dfm["rsa"].astype(np.float32).to_numpy()
        print(f"  [1] Continuous RSA (Spearman)...")
        all_rho = np.zeros(n_features, dtype=np.float32)
        all_p = np.ones(n_features, dtype=np.float32)

        for ci in tqdm(range(n_chunks), desc="    RSA chunks"):
            i = ci * chunk_size
            end = min(i + chunk_size, n_features)
            if A_proj is None:
                Zr = np.asarray(Z[ridx, i:end], dtype=np.float32)
            else:
                Zr = (A_proj @ np.asarray(Z[:, i:end], dtype=np.float32))[ridx, :]
            rho, p = spearman_with_pvals(Zr, rsa)
            all_rho[i:end] = rho
            all_p[i:end] = p

        all_q = bh_fdr(all_p)
        results["rsa_spearman"] = {"rho": all_rho, "p": all_p, "q": all_q}
        n_sig = (all_q < 0.05).sum()
        print(f"    Significant: {n_sig} / {n_features} "
              f"({100*n_sig/n_features:.1f}%)")

    # --- Binary functional annotations (Pearson, same as thesis pipeline) ---
    functional_cols = [c for c in FUNCTIONAL_TYPES.values() if c in dfm.columns]

    for col in functional_cols:
        y = dfm[col].astype(np.float32).to_numpy()
        n_positive = (y > 0).sum()
        if n_positive < 10:
            print(f"  [{col}] Skipping: only {n_positive} positive residues")
            continue

        print(f"  [{col}] ({n_positive} positive residues)...")
        all_r = np.zeros(n_features, dtype=np.float32)
        all_p = np.ones(n_features, dtype=np.float32)

        for ci in tqdm(range(n_chunks), desc=f"    {col} chunks"):
            i = ci * chunk_size
            end = min(i + chunk_size, n_features)
            if A_proj is None:
                Zr = np.asarray(Z[ridx, i:end], dtype=np.float32)
            else:
                Zr = (A_proj @ np.asarray(Z[:, i:end], dtype=np.float32))[ridx, :]
            r, p = pearson_with_pvals(Zr, y)
            all_r[i:end] = r
            all_p[i:end] = p

        all_q = bh_fdr(all_p)
        results[col] = {"r": all_r, "p": all_p, "q": all_q}
        n_sig = (all_q < 0.05).sum()
        print(f"    Significant: {n_sig} / {n_features} "
              f"({100*n_sig/n_features:.1f}%)")

    # --- Save ---
    out_df = pd.DataFrame({"feature_idx": np.arange(n_features)})

    if "rsa_spearman" in results:
        out_df["rho_rsa"] = results["rsa_spearman"]["rho"]
        out_df["p_rsa"] = results["rsa_spearman"]["p"]
        out_df["q_rsa"] = results["rsa_spearman"]["q"]

    for col in functional_cols:
        if col in results:
            out_df[f"corr_{col}"] = results[col]["r"]
            out_df[f"p_{col}"] = results[col]["p"]
            out_df[f"q_{col}"] = results[col]["q"]

    out_df.to_csv(save_dir / "extended_interpretability.csv", index=False)

    # --- Summary ---
    summary_rows = []

    # Original annotations (from existing feature_interpretability.csv if available)
    orig_path = Path(save_dir).parent / "feature_interpretability.csv"
    # Try layer_dir
    orig_path2 = Path(args.layer_dir) / "feature_interpretability.csv"
    for op in [orig_path, orig_path2]:
        if op.exists():
            orig_df = pd.read_csv(op)
            n_orig = len(orig_df)
            any_orig = ((orig_df["q_helix"] < 0.05) |
                        (orig_df["q_strand"] < 0.05) |
                        (orig_df["q_burial"] < 0.05)).sum()
            summary_rows.append({
                "annotation": "original (helix/strand/burial)",
                "n_significant": int(any_orig),
                "pct_significant": float(100 * any_orig / n_orig),
            })
            break

    if "rsa_spearman" in results:
        n_sig = int((results["rsa_spearman"]["q"] < 0.05).sum())
        summary_rows.append({
            "annotation": "continuous RSA (Spearman)",
            "n_significant": n_sig,
            "pct_significant": float(100 * n_sig / n_features),
        })

    for col in functional_cols:
        if col in results:
            n_sig = int((results[col]["q"] < 0.05).sum())
            summary_rows.append({
                "annotation": col,
                "n_significant": n_sig,
                "pct_significant": float(100 * n_sig / n_features),
            })

    # Any functional site
    func_q_cols = [f"q_{c}" for c in functional_cols if c in results]
    if func_q_cols:
        any_func = np.zeros(n_features, dtype=bool)
        for qc in func_q_cols:
            any_func |= (out_df[qc].values < 0.05)
        n_any = int(any_func.sum())
        summary_rows.append({
            "annotation": "any functional site",
            "n_significant": n_any,
            "pct_significant": float(100 * n_any / n_features),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(save_dir / "extended_summary.csv", index=False)
    print(f"\n  Summary:")
    print(summary_df.to_string(index=False))

    # --- Plots ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: RSA Spearman distribution
    ax = axes[0]
    if "rsa_spearman" in results:
        rho_vals = results["rsa_spearman"]["rho"]
        q_vals = results["rsa_spearman"]["q"]
        sig_mask = q_vals < 0.05
        ax.hist(rho_vals[~sig_mask], bins=50, alpha=0.5, color="grey",
                label="Not significant", density=True)
        ax.hist(rho_vals[sig_mask], bins=50, alpha=0.7, color="#2196F3",
                label=f"q < 0.05 ({sig_mask.sum()})", density=True)
        ax.axvline(0, color="black", ls="--", lw=1)
        ax.set_xlabel("Spearman ρ (feature vs RSA)")
        ax.set_ylabel("Density")
        ax.set_title("Continuous RSA Correlations")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "RSA not available", ha="center", va="center",
                transform=ax.transAxes)
    ax.grid(alpha=0.3)

    # Panel B: Bar chart comparing annotation types
    ax = axes[1]
    if summary_rows:
        sdf = pd.DataFrame(summary_rows)
        colors_bar = ["#4CAF50"] + ["#2196F3"] * (len(sdf) - 1)
        if len(sdf) > 1:
            colors_bar[0] = "#4CAF50"  # original in green
        ax.barh(range(len(sdf)), sdf["pct_significant"], color=colors_bar,
                alpha=0.7, edgecolor="black", linewidth=0.5)
        ax.set_yticks(range(len(sdf)))
        ax.set_yticklabels(sdf["annotation"], fontsize=9)
        ax.set_xlabel("% Features with q < 0.05")
        ax.set_title("Interpretability by Annotation Type")
        ax.grid(alpha=0.3, axis="x")

    fig.tight_layout()
    fig.savefig(save_dir / "extended_annotations_results.png", dpi=300)
    fig.savefig(save_dir / "extended_annotations_results.pdf")
    plt.close(fig)

    print(f"\n  Saved to {save_dir}/")


# =====================================================================
#                           MAIN
# =====================================================================

def main():
    global args  # needed for orig_path lookup in run_extended_correlation
    ap = argparse.ArgumentParser(
        description="Expanded interpretability annotations")
    ap.add_argument("--layer-dir", required=True)
    ap.add_argument("--model-type", choices=["residue", "protgpt2"], required=True)
    ap.add_argument("--features-csv", default="cache/residue_features.csv")
    ap.add_argument("--pdb-dir", default="cache/pdb_files")
    ap.add_argument("--tokenizer", default="nferruz/ProtGPT2")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--max-uniprot-queries", type=int, default=500,
                    help="Max UniProt accessions to query (rate limiting)")
    ap.add_argument("--save-dir", default="results_expanded_annotations")
    ap.add_argument("--skip-uniprot", action="store_true",
                    help="Skip UniProt queries (RSA only)")
    args = ap.parse_args()

    layer_dir = Path(args.layer_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    pdb_dir = Path(args.pdb_dir)

    print("=" * 70)
    print("  EXPANDED INTERPRETABILITY ANNOTATIONS")
    print("=" * 70)

    # Load data
    Z, uids, tok_lengths = load_layer(layer_dir)
    ref_seqs = load_ref_seqs(layer_dir)
    sequences = [ref_seqs[uid] for uid in uids]

    print(f"  Z: {Z.shape}")
    print(f"  Proteins: {len(uids)}")

    # Load physical features
    features_csv = Path(args.features_csv)
    df_phys = load_phys_features(features_csv) if features_csv.exists() else pd.DataFrame()

    # Build residue index
    if args.model_type == "residue":
        res_lengths = np.array([len(ref_seqs[uid]) for uid in uids], dtype=np.int32)
        res_offsets = {}
        off = 0
        for uid, Lr in zip(uids, res_lengths):
            res_offsets[uid] = off
            off += int(Lr)
        A_proj = None
    else:
        A_proj, res_offsets, res_lengths = build_protgpt2_projection(
            uids, ref_seqs, tok_lengths, args.tokenizer)

    res_idx = build_residue_index(uids, res_lengths, res_offsets)

    # Step 1: Extract continuous RSA
    print("\n  [Step 1] Extracting continuous RSA from DSSP...")
    if len(df_phys) > 0:
        df_phys = extract_rsa_from_dssp(df_phys, pdb_dir, n_jobs=args.n_jobs)
    else:
        print("  No features CSV found, building RSA from scratch...")
        # Build minimal df with uid, position, rsa
        rows = []
        for uid in uids:
            for pos in range(len(ref_seqs[uid])):
                rows.append({"uid": uid, "position": pos, "rsa": 0.0,
                             "ss_8class": "-", "neighbor_count": 0})
        df_phys = pd.DataFrame(rows)
        if have_dssp():
            df_phys = fill_ss8_from_dssp(df_phys, pdb_dir, n_jobs=args.n_jobs)
        if "sasa" in df_phys.columns:
            df_phys["rsa"] = df_phys["sasa"].astype(np.float32)

    # Step 2: UniProt functional annotations
    if not args.skip_uniprot:
        print("\n  [Step 2] Mapping to UniProt functional annotations...")
        uid_to_uniprot = map_scope_to_uniprot_sifts(uids, pdb_dir)

        if uid_to_uniprot:
            df_func = build_functional_annotations(
                uids, sequences, uid_to_uniprot,
                max_queries=args.max_uniprot_queries)

            # Merge functional annotations into df_phys
            df_phys = df_phys.merge(df_func, on=["uid", "position"], how="left")
            for col in FUNCTIONAL_TYPES.values():
                if col in df_phys.columns:
                    df_phys[col] = df_phys[col].fillna(0).astype(np.int8)
        else:
            print("  No UniProt mappings found, skipping functional annotations")
    else:
        print("\n  [Step 2] Skipping UniProt (--skip-uniprot)")

    # Step 3: Run extended correlation pipeline
    print("\n  [Step 3] Running extended correlation pipeline...")
    run_extended_correlation(Z, A_proj, df_phys, res_idx, save_dir,
                              n_jobs=args.n_jobs)

    print(f"\n{'=' * 70}")
    print(f"  Complete! Results in {save_dir}/")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
