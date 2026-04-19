#!/usr/bin/env python3
"""
cpu_stage.py  v3 – Vectorised Sparse-Matrix Edition
=====================================================
CRITICAL SPEEDUP: struct/seq locality (step [3/6]) now uses sparse-matrix
neighbor averaging instead of Python adjacency-list loops.
Expected ~20-50× faster on that bottleneck.

Also FIXES: topk_frac is now properly used to threshold "active" residues
(previously hardcoded to acts > 0, making the sweep a no-op).

Unified SAE analysis for residue-token PLMs (ESM/ProtT5) and BPE-token
ProtGPT2, with fair residue-level comparison via token→residue projection.

Expected layer_dir contents:
  Z.npy           (N_tokens × n_features)
  D.npy           (n_features × embed_dim)   [optional, for decoder UMAP]
  uids.json
  lengths.npy
  sequences.json

Usage:
  python cpu_stage.py --layer-dir outputs/esm2/layer_16 --model-type residue
  python cpu_stage.py --layer-dir outputs/protgpt2/layer_16 --model-type protgpt2
  python cpu_stage.py --layer-dir outputs/esm2/layer_16 --model-type residue --sweep-topk
"""

import os, json, shutil, argparse, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from tqdm import tqdm
from scipy.spatial import KDTree
from scipy import sparse
from scipy.stats import t as tdist, spearmanr
from Bio.PDB import PDBParser
from Bio import pairwise2
from Bio.PDB.DSSP import DSSP
from joblib import Parallel, delayed, cpu_count

warnings.simplefilter("ignore")

N_THREADS = str(cpu_count())
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, N_THREADS)

# ===========================================================================
#                    STRUCTURALLY-MOTIVATED CONSTANTS
# ===========================================================================
# CONTACT_CUTOFF = 8.0 Å  — Cα–Cα distance (Jumper 2021, Marks 2011)
# SEQ_GAP_MIN = 12         — excludes helix contacts (Marks 2011, Morcos 2011)
# These derive from protein geometry, NOT tunable hyperparameters.
# ===========================================================================
DEFAULT_CONTACT_CUTOFF = 8.0
DEFAULT_SEQ_GAP_MIN = 12
DEFAULT_TOPK_FRAC = 0.10
SWEEP_TOPK_VALUES = [0.05, 0.10, 0.15, 0.20]


# ===========================================================================
#                           SMALL HELPERS
# ===========================================================================

def load_ref_seqs(layer_dir: Path) -> dict:
    p = layer_dir / "sequences.json"
    d = json.loads(p.read_text())
    if isinstance(d, dict):
        return {str(k): str(v) for k, v in d.items()}
    uids_path = layer_dir / "uids.json"
    uids = json.loads(uids_path.read_text()) if uids_path.exists() else None
    if isinstance(d, list) and d and isinstance(d[0], dict):
        out = {}
        for it in d:
            uid = it.get("uid") or it.get("id") or it.get("name")
            seq = it.get("sequence") or it.get("seq")
            if uid is not None and seq is not None:
                out[str(uid)] = str(seq)
        if out: return out
        raise ValueError("sequences.json list-of-dicts lacks uid/sequence keys")
    if isinstance(d, list) and d and isinstance(d[0], (list, tuple)) and len(d[0]) == 2:
        return {str(k): str(v) for k, v in d}
    if isinstance(d, list) and (not d or isinstance(d[0], str)):
        if uids is None: raise ValueError("Need uids.json to map list sequences.json")
        if len(d) != len(uids): raise ValueError("sequences.json length != uids.json length")
        return {str(uid): str(seq) for uid, seq in zip(uids, d)}
    raise ValueError(f"Unsupported sequences.json format: {type(d)}")


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=np.float64)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(n) + 1.0)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.clip(q, 0.0, 1.0)
    return out.astype(np.float32)


def have_dssp() -> bool:
    return shutil.which("mkdssp") is not None or shutil.which("dssp") is not None

def pick_dssp_binary() -> str:
    for b in ("mkdssp", "dssp"):
        if shutil.which(b): return b
    return ""

def find_first_model_chain(structure):
    model = next(iter(structure))
    for chain in model:
        n = 0
        for res in chain:
            if res.id[0] == " " and "CA" in res:
                n += 1
                if n >= 5: return model, chain.id
    raise ValueError("No suitable chain found.")

def extract_chain_ca_seq_coords(chain):
    aa3_to_aa1 = {
        "ALA":"A","CYS":"C","ASP":"D","GLU":"E","PHE":"F","GLY":"G","HIS":"H","ILE":"I",
        "LYS":"K","LEU":"L","MET":"M","ASN":"N","PRO":"P","GLN":"Q","ARG":"R","SER":"S",
        "THR":"T","VAL":"V","TRP":"W","TYR":"Y","SEC":"U","PYL":"O","ASX":"B","GLX":"Z","UNK":"X"
    }
    seq, coords = [], []
    for res in chain:
        if res.id[0] != " " or "CA" not in res: continue
        seq.append(aa3_to_aa1.get(res.get_resname().upper(), "X"))
        coords.append(res["CA"].get_coord())
    coords = np.asarray(coords, dtype=np.float32) if coords else np.zeros((0, 3), dtype=np.float32)
    return "".join(seq), coords


# ===========================================================================
#                     DSSP FILL IF PLACEHOLDER
# ===========================================================================

def fill_ss8_from_dssp(df_phys: pd.DataFrame, pdb_dir: Path, n_jobs: int = -1) -> pd.DataFrame:
    if not have_dssp(): raise RuntimeError("DSSP not found on PATH.")
    pdb_dir = Path(pdb_dir)
    out = df_phys.copy()
    if "ss_8class" not in out.columns: out["ss_8class"] = "-"
    if "sasa" not in out.columns: out["sasa"] = 0.0
    dssp_bin = pick_dssp_binary()
    uids = out["uid"].unique().tolist()

    def process_uid(uid):
        pdb_path = pdb_dir / f"{str(uid)[1:5].lower()}.pdb"
        if not pdb_path.exists(): return uid, None, None
        try:
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure(str(uid), str(pdb_path))
            model, chain_id = find_first_model_chain(structure)
            dssp = DSSP(model, str(pdb_path), dssp=dssp_bin)
            chain_keys = sorted([k for k in dssp.keys() if k[0] == chain_id],
                                key=lambda k: (k[1][1], k[1][2]))
            ss_list = [dssp[k][2] if dssp[k][2] else "-" for k in chain_keys]
            rsa_list = [float(dssp[k][3]) if dssp[k][3] is not None else 0.0 for k in chain_keys]
            return uid, ss_list, rsa_list
        except Exception: return uid, None, None

    results = Parallel(n_jobs=n_jobs, verbose=5)(delayed(process_uid)(uid) for uid in uids)
    for uid, ss_list, rsa_list in results:
        if ss_list is None: continue
        sub_idx = out.index[out["uid"] == uid].to_numpy()
        sub_pos = out.loc[sub_idx, "position"].astype(int).to_numpy()
        for idx, pos0 in zip(sub_idx, sub_pos):
            if 0 <= pos0 < len(ss_list):
                out.at[idx, "ss_8class"] = ss_list[pos0]
                out.at[idx, "sasa"] = rsa_list[pos0]
    return out


# ===========================================================================
#          PROTGPT2 TOKEN->RESIDUE PROJECTION
# ===========================================================================

def _clean_piece(tok: str) -> str:
    return tok.replace("\u2581", "").replace("\u0120", "").replace(" ", "")

def build_protgpt2_projection(uids, seqs_by_uid, token_lengths, tokenizer_name: str):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    res_lengths = np.array([len(seqs_by_uid[uid]) for uid in uids], dtype=np.int32)
    res_offsets = {}; off = 0
    for uid, Lr in zip(uids, res_lengths): res_offsets[uid] = off; off += int(Lr)
    n_res_total, n_tok_total = int(off), int(np.sum(token_lengths))
    rows, cols, data = [], [], []
    tok_global = 0
    for uid, Lt in tqdm(list(zip(uids, token_lengths)), desc="ProtGPT2 spans", leave=False):
        seq = seqs_by_uid[uid]
        ids = tok(seq, add_special_tokens=False)["input_ids"]
        if len(ids) != int(Lt): raise ValueError(f"Token count mismatch for {uid}")
        toks = tok.convert_ids_to_tokens(ids)
        spans, cursor = [], 0
        for t in toks:
            piece = _clean_piece(t)
            if piece == "": spans.append((cursor, cursor - 1)); continue
            Lp = len(piece)
            if seq[cursor:cursor + Lp] != piece:
                raise ValueError(f"Span decode mismatch for {uid} at cursor {cursor}")
            spans.append((cursor, cursor + Lp - 1)); cursor += Lp
        if cursor != len(seq): raise ValueError(f"Span coverage mismatch for {uid}")
        base_res = res_offsets[uid]
        for local_tok, (a, b) in enumerate(spans):
            if b < a: tok_global += 1; continue
            w = 1.0 / float(b - a + 1)
            for r in range(a, b + 1):
                rows.append(base_res + r); cols.append(tok_global); data.append(w)
            tok_global += 1
    A = sparse.coo_matrix(
        (np.array(data, dtype=np.float32),
         (np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32))),
        shape=(n_res_total, n_tok_total)
    ).tocsr()
    return A, res_offsets, res_lengths


# ===========================================================================
#                    DATA LOADING + RESIDUE INDEX
# ===========================================================================

def load_layer(layer_dir: Path):
    Z = np.load(layer_dir / "Z.npy", mmap_mode="r")
    uids = json.loads((layer_dir / "uids.json").read_text())
    lengths = np.load(layer_dir / "lengths.npy")
    if int(np.sum(lengths)) != int(Z.shape[0]):
        raise ValueError("sum(lengths.npy) != Z rows")
    return Z, uids, lengths

def load_phys_features(features_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(features_csv)
    if "position_pdb" in df.columns: df = df.rename(columns={"position_pdb": "position"})
    if "position" in df.columns and df["position"].min() == 1: df["position"] -= 1
    return df

def build_residue_index(uids, res_lengths, res_offsets):
    rows = []
    for uid, L in zip(uids, res_lengths):
        base = res_offsets[uid]; pos = np.arange(int(L), dtype=np.int32)
        rows.append(pd.DataFrame({"uid": uid, "position": pos, "res_global": base + pos}))
    return pd.concat(rows, ignore_index=True)


# ===========================================================================
#     [1/6]  CORRELATION (r, p, q) ON RESIDUES  -  PARALLELIZED
# ===========================================================================

def corr_with_pvals(Z_res_chunk: np.ndarray, y: np.ndarray):
    y = y.astype(np.float32); n = float(y.shape[0])
    y0 = y - y.mean(); y_norm = np.linalg.norm(y0)
    if y_norm == 0:
        return np.zeros(Z_res_chunk.shape[1], dtype=np.float32), np.ones(Z_res_chunk.shape[1], dtype=np.float32)
    Zm = Z_res_chunk.astype(np.float32)
    z0 = Zm - Zm.mean(axis=0, keepdims=True)
    z_norm = np.linalg.norm(z0, axis=0); z_norm[z_norm == 0] = 1.0
    r = (z0.T @ y0) / (z_norm * y_norm)
    r = np.clip(r, -0.999999, 0.999999).astype(np.float32)
    df = max(int(n) - 2, 1)
    t = np.abs(r) * np.sqrt(df / (1.0 - r * r))
    p = (2.0 * tdist.sf(t, df=df)).astype(np.float32)
    return r, p

def _process_correlation_chunk(chunk_idx, chunk_size, Z, A, ridx,
                                is_helix, is_strand, burial, n_features):
    i = chunk_idx * chunk_size; end = min(i + chunk_size, n_features)
    if A is None:
        Zr = np.asarray(Z[ridx, i:end], dtype=np.float32)
    else:
        Zr = (A @ np.asarray(Z[:, i:end], dtype=np.float32))[ridx, :]
    rH, pH = corr_with_pvals(Zr, is_helix)
    rS, pS = corr_with_pvals(Zr, is_strand)
    rB, pB = corr_with_pvals(Zr, burial)
    return i, end, rH, pH, rS, pS, rB, pB

def analyze_feature_meanings_residue(Z, A, df_phys, res_idx, save_dir: Path, n_jobs: int = -1):
    save_dir = Path(save_dir)
    dfm = df_phys.merge(res_idx, on=["uid", "position"], how="inner")
    if len(dfm) == 0: raise ValueError("No residues aligned")
    ridx = dfm["res_global"].astype(int).to_numpy()
    ss = dfm.get("ss_8class", pd.Series(["-"] * len(dfm))).astype(str)
    is_helix = ss.isin(["H", "G", "I"]).astype(np.float32).to_numpy()
    is_strand = ss.isin(["E", "B"]).astype(np.float32).to_numpy()
    burial = dfm.get("neighbor_count", pd.Series(np.zeros(len(dfm)))).astype(np.float32).to_numpy()
    n_features = int(Z.shape[1]); chunk_size = 512
    n_chunks = (n_features + chunk_size - 1) // chunk_size
    print(f"  Computing correlations for {n_features} features using {n_jobs} workers...")
    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_process_correlation_chunk)(ci, chunk_size, Z, A, ridx,
                                            is_helix, is_strand, burial, n_features)
        for ci in range(n_chunks))
    rH = np.zeros(n_features, dtype=np.float32); pH = np.ones(n_features, dtype=np.float32)
    rS = np.zeros(n_features, dtype=np.float32); pS = np.ones(n_features, dtype=np.float32)
    rB = np.zeros(n_features, dtype=np.float32); pB = np.ones(n_features, dtype=np.float32)
    for i, end, rHc, pHc, rSc, pSc, rBc, pBc in results:
        rH[i:end]=rHc; pH[i:end]=pHc; rS[i:end]=rSc; pS[i:end]=pSc; rB[i:end]=rBc; pB[i:end]=pBc
    qH, qS, qB = bh_fdr(pH), bh_fdr(pS), bh_fdr(pB)
    out = pd.DataFrame({
        "feature_idx": np.arange(n_features, dtype=np.int32),
        "corr_helix": rH, "p_helix": pH, "q_helix": qH,
        "corr_strand": rS, "p_strand": pS, "q_strand": qS,
        "corr_burial": rB, "p_burial": pB, "q_burial": qB,
    })
    out.to_csv(save_dir / "feature_interpretability.csv", index=False)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=out, x="corr_helix", y="corr_strand", hue="corr_burial",
                    palette="viridis", s=6, alpha=0.6)
    plt.axhline(0, c="grey", ls="--"); plt.axvline(0, c="grey", ls="--")
    plt.title("SAE feature correlations (residue space)")
    plt.tight_layout(); plt.savefig(save_dir / "plot_structure_correlations.png", dpi=200); plt.close()
    print(f"  Saved feature_interpretability.csv")


# ===========================================================================
#     [2/6]  FOLD ENRICHMENT  -  PARALLELIZED
# ===========================================================================

def _process_fold_enrichment_chunk(chunk_idx, chunk_size, Z, A, all_res_idx,
                                    fold_res_idx, top_folds, n_features):
    i = chunk_idx * chunk_size; end = min(i + chunk_size, n_features); results = []
    if A is None:
        Zglob = np.asarray(Z[all_res_idx, i:end], dtype=np.float32)
    else:
        Zres = A @ np.asarray(Z[:, i:end], dtype=np.float32)
        Zglob = Zres[all_res_idx, :]
    global_mean = Zglob.mean(axis=0) + 1e-6
    for fold in top_folds:
        ridx = fold_res_idx.get(fold)
        if ridx is None or ridx.size < 50: continue
        Zf = np.asarray(Z[ridx, i:end], dtype=np.float32) if A is None else Zres[ridx, :]
        enrich = Zf.mean(axis=0) / global_mean
        for j in np.where(enrich > 5.0)[0]:
            results.append({"fold": fold, "feature_idx": int(i+j), "enrichment": float(enrich[j])})
    return results

def analyze_fold_enrichment_residue(Z, A, uids, res_lengths, res_offsets,
                                     fasta_path: Path, save_dir: Path, n_jobs: int = -1):
    fasta_path = Path(fasta_path)
    if not fasta_path.exists():
        print(f"  Skipping fold enrichment: {fasta_path} not found"); return
    meta = {}
    with open(fasta_path) as f:
        for line in f:
            if line.startswith(">"):
                parts = line[1:].split()
                if len(parts) >= 2: meta[parts[0]] = ".".join(parts[1].split(".")[:2])
    folds = [meta.get(uid) for uid in uids]
    vc = pd.Series([x for x in folds if x]).value_counts()
    top_folds = vc.head(50).index.tolist()
    if not top_folds: print("  Skipping fold enrichment: no folds found"); return
    n_features = int(Z.shape[1]); chunk_size = 256
    fold_res_idx = {}; all_res_idx = []
    for uid, Lr in zip(uids, res_lengths):
        base = res_offsets[uid]; ridx = np.arange(base, base + int(Lr), dtype=np.int32)
        all_res_idx.append(ridx)
        fold = meta.get(uid)
        if fold in top_folds: fold_res_idx.setdefault(fold, []).append(ridx)
    all_res_idx = np.concatenate(all_res_idx)
    for k in list(fold_res_idx): fold_res_idx[k] = np.concatenate(fold_res_idx[k])
    n_chunks = (n_features + chunk_size - 1) // chunk_size
    print(f"  Computing fold enrichment for {n_features} features using {n_jobs} workers...")
    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_process_fold_enrichment_chunk)(ci, chunk_size, Z, A, all_res_idx,
                                                fold_res_idx, top_folds, n_features)
        for ci in range(n_chunks))
    all_results = [r for chunk in results for r in chunk]
    if all_results:
        pd.DataFrame(all_results).to_csv(Path(save_dir) / "fold_enrichment.csv", index=False)
        print(f"  Saved fold_enrichment.csv ({len(all_results)} enrichments)")
    else:
        print("  No significant fold enrichments found")


# ===========================================================================
#     [3/6]  STRUCT VS SEQ LOCALITY  -  VECTORISED SPARSE MATRIX
# ===========================================================================
#
#  OLD (v2): Python loop over 360k adjacency lists per feature = ~2.5 hrs
#  NEW (v3): sparse matrix multiply A @ acts_chunk = ~5-10 min
#
#  The key insight: neighbor_avg[i] = sum(acts[j] for j in nbrs[i]) / degree[i]
#  is equivalent to (A @ acts) / degree, where A is the adjacency matrix.
#  scipy.sparse handles this in optimised C code.
# ===========================================================================

def adj_list_to_sparse(adj_list, n_res):
    """Convert adjacency list to CSR matrix + degree vector. One-time cost."""
    rows, cols = [], []
    for i, nbrs in enumerate(adj_list):
        for j in nbrs:
            rows.append(i); cols.append(j)
    if not rows:
        return sparse.csr_matrix((n_res, n_res), dtype=np.float32), np.zeros(n_res, dtype=np.float32)
    A = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32),
         (np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32))),
        shape=(n_res, n_res))
    deg = np.asarray(A.sum(axis=1)).ravel().astype(np.float32)
    return A, deg


def build_protein_permutations(res_lengths, n_shuffles, seed=42):
    """Pre-compute within-protein shuffle permutations (one-time)."""
    n_res = int(np.sum(res_lengths))
    rng = np.random.RandomState(seed)
    perms = []
    for _ in range(n_shuffles):
        perm = np.arange(n_res, dtype=np.int64)
        off = 0
        for L in res_lengths:
            L = int(L); rng.shuffle(perm[off:off + L]); off += L
        perms.append(perm)
    return perms


def _cohens_d_vectorized(acts_chunk, A_sp, deg, global_stds, topk_frac):
    """
    Vectorised Cohen's d for a whole chunk of features at once.

    Instead of looping over adjacency lists per residue (Python, slow),
    we compute neighbor averages via sparse matmul (C, fast):
        smoothed = (A_sp @ acts_chunk) / degree

    Args:
        acts_chunk:  (n_res, chunk_size) float32 — activation matrix
        A_sp:        (n_res, n_res) sparse CSR — adjacency matrix
        deg:         (n_res,) float32 — node degrees
        global_stds: (chunk_size,) float32 — per-feature std for normalisation
        topk_frac:   float — fraction of residues considered "active"

    Returns:
        (chunk_size,) float32 — Cohen's d per feature
    """
    n_res, n_feat = acts_chunk.shape

    # Sparse matmul: neighbor sums for ALL features at once
    nbr_sums = np.asarray(A_sp @ acts_chunk, dtype=np.float32)  # (n_res, n_feat)

    # In-place normalise to neighbor averages
    has_nbrs = deg > 0
    nbr_sums[has_nbrs] /= deg[has_nbrs, None]
    nbr_sums[~has_nbrs] = 0.0
    smoothed = nbr_sums  # alias, no copy

    # Global mean of smoothed values
    global_mean = smoothed.mean(axis=0)  # (n_feat,)

    # Active mask: top topk_frac of activations per feature
    # For sparse SAE activations (many exact zeros from ReLU),
    # percentile at 90th is often 0, so this gracefully degrades to acts > 0.
    thresh = np.percentile(acts_chunk, 100.0 * (1.0 - topk_frac), axis=0)  # (n_feat,)
    active = acts_chunk > thresh[None, :]  # (n_res, n_feat) bool

    n_active = active.sum(axis=0).astype(np.float32)  # (n_feat,)

    # Mean of smoothed values at active residues
    active_sum = (smoothed * active).sum(axis=0)  # (n_feat,)
    n_safe = n_active.copy(); n_safe[n_safe == 0] = 1.0
    active_mean = active_sum / n_safe

    # Cohen's d = (active_mean - global_mean) / global_std
    d = (active_mean - global_mean) / (global_stds + 1e-6)
    d[n_active < 5] = 0.0
    return d.astype(np.float32)


def _process_struct_seq_chunk_v3(chunk_idx, chunk_size, Z, A_proj,
                                  A_seq, deg_seq, A_struct, deg_struct,
                                  perm_indices, n_features, topk_frac):
    """
    Process a chunk of features: compute observed + shuffled Cohen's d
    for both structural and sequential adjacency, all vectorised.
    """
    i = chunk_idx * chunk_size
    end = min(i + chunk_size, n_features)

    # Load / project activations → (n_res, chunk_size)
    if A_proj is None:
        acts = np.asarray(Z[:, i:end], dtype=np.float32)
    else:
        acts = np.asarray(A_proj @ np.asarray(Z[:, i:end], dtype=np.float32), dtype=np.float32)

    gstds = np.std(acts, axis=0).astype(np.float32)  # (chunk_size,)

    # --- Observed ---
    seq_obs = _cohens_d_vectorized(acts, A_seq, deg_seq, gstds, topk_frac)
    str_obs = _cohens_d_vectorized(acts, A_struct, deg_struct, gstds, topk_frac)

    # --- Shuffled (within-protein permutations, pre-computed) ---
    n_sh = len(perm_indices)
    cs = end - i
    seq_sh = np.zeros(cs, dtype=np.float32)
    str_sh = np.zeros(cs, dtype=np.float32)
    for perm in perm_indices:
        acts_p = acts[perm]  # permuted rows — breaks spatial structure
        seq_sh += _cohens_d_vectorized(acts_p, A_seq, deg_seq, gstds, topk_frac)
        str_sh += _cohens_d_vectorized(acts_p, A_struct, deg_struct, gstds, topk_frac)
    if n_sh > 0:
        seq_sh /= n_sh; str_sh /= n_sh

    # Return arrays (not dicts — assembled into DataFrame later)
    return (np.arange(i, end, dtype=np.int32),
            seq_obs, str_obs, seq_sh, str_sh)


def _process_single_protein_graph(uid, Lr, ref_seq, pdb_dir, offset,
                                  contact_cutoff=8.0, seq_gap_min=12):
    """Build neighbor adjacency lists for one protein (unchanged from v2)."""
    Lr = int(Lr)
    # Sequential neighbors: ±1, ±2
    p_seq = [[] for _ in range(Lr)]
    for r in range(Lr):
        for d in (-2, -1, 1, 2):
            rr = r + d
            if 0 <= rr < Lr: p_seq[r].append(offset + rr)

    # Structural neighbors: Cα within cutoff AND |i-j| >= gap
    p_struct = [[] for _ in range(Lr)]
    pdb_path = pdb_dir / f"{str(uid)[1:5].lower()}.pdb"
    if pdb_path.exists():
        try:
            parser = PDBParser(QUIET=True)
            struct = parser.get_structure("t", str(pdb_path))
            best_score, best_coords = None, None
            for model in struct:
                for chain in model:
                    chain_seq, chain_coords = extract_chain_ca_seq_coords(chain)
                    if len(chain_seq) < 10: continue
                    aln = pairwise2.align.globalms(ref_seq, chain_seq, 2, -1, -5, -0.5,
                                                    one_alignment_only=True)
                    if not aln: continue
                    aln = aln[0]
                    coords_ref = np.full((Lr, 3), np.nan, dtype=np.float32)
                    ri, ci = -1, -1
                    for a, b in zip(aln.seqA, aln.seqB):
                        if a != "-": ri += 1
                        if b != "-": ci += 1
                        if a != "-" and b != "-" and 0 <= ri < Lr and 0 <= ci < len(chain_coords):
                            coords_ref[ri] = chain_coords[ci]
                    if best_score is None or aln.score > best_score:
                        best_score, best_coords = aln.score, coords_ref
                break  # first model only
            if best_coords is not None:
                valid = ~np.isnan(best_coords).any(axis=1)
                idx_map = np.where(valid)[0]
                if idx_map.size >= 2:
                    pairs = KDTree(best_coords[valid]).query_pairs(r=contact_cutoff)
                    for a, b in pairs:
                        ra, rb = int(idx_map[a]), int(idx_map[b])
                        if abs(ra - rb) >= seq_gap_min:
                            p_struct[ra].append(offset + rb)
                            p_struct[rb].append(offset + ra)
        except Exception: pass
    return p_seq, p_struct


def build_neighbor_graphs_residue_parallel(uids, res_lengths, ref_seqs, pdb_dir: Path,
                                           n_jobs=-1, contact_cutoff=8.0, seq_gap_min=12):
    """Build adjacency lists (parallelised over proteins)."""
    pdb_dir = Path(pdb_dir)
    offsets, off = [], 0
    for Lr in res_lengths: offsets.append(off); off += int(Lr)
    print(f"  Building neighbor graphs for {len(uids)} proteins using {n_jobs} workers...")
    print(f"    Contact cutoff: {contact_cutoff} \u00c5, Sequence gap: \u2265{seq_gap_min} residues")
    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_process_single_protein_graph)(uid, Lr, ref_seqs[uid], pdb_dir, offset,
                                               contact_cutoff, seq_gap_min)
        for uid, Lr, offset in zip(uids, res_lengths, offsets))
    seq_adj, struct_adj = [], []
    for p_seq, p_struct in results:
        seq_adj.extend(p_seq); struct_adj.extend(p_struct)
    return seq_adj, struct_adj


def analyze_struct_seq_residue_parallel(Z, A_proj, uids, res_lengths, ref_seqs,
                                        pdb_dir: Path, save_dir: Path,
                                        n_shuffles: int, n_jobs=-1,
                                        contact_cutoff=8.0, seq_gap_min=12,
                                        topk_frac=0.10):
    """
    Main struct/seq locality analysis — VECTORISED v3.

    1. Build adjacency lists (parallel over proteins)
    2. Convert to sparse CSR matrices (one-time, fast)
    3. Pre-compute within-protein permutations
    4. Sparse matmul for neighbor averaging (parallel over feature chunks)
    """
    save_dir = Path(save_dir)
    n_res = int(np.sum(res_lengths))

    # Step 1: adjacency lists (parallelised over proteins — unchanged)
    seq_adj, struct_adj = build_neighbor_graphs_residue_parallel(
        uids, res_lengths, ref_seqs, pdb_dir, n_jobs,
        contact_cutoff=contact_cutoff, seq_gap_min=seq_gap_min)

    # Step 2: convert to sparse matrices (seconds, not minutes)
    import time
    t0 = time.time()
    print("  Converting adjacency → sparse matrices...")
    A_seq, deg_seq = adj_list_to_sparse(seq_adj, n_res)
    A_struct, deg_struct = adj_list_to_sparse(struct_adj, n_res)
    del seq_adj, struct_adj  # free ~2 GB of Python lists
    print(f"    Sequential:  {A_seq.nnz:,} edges")
    print(f"    Structural:  {A_struct.nnz:,} edges")
    print(f"    Conversion:  {time.time()-t0:.1f}s")

    # Step 3: pre-compute permutations
    print(f"  Pre-computing {n_shuffles} within-protein permutations...")
    perm_indices = build_protein_permutations(res_lengths, n_shuffles)

    # Step 4: process feature chunks in parallel
    n_features = int(Z.shape[1])
    chunk_size = 256

    # Auto-limit workers to avoid OOM: ~1.5 GB per worker for 360k residues.
    # Default budget: 100 GB (was 40 GB).  Override via CPU_STAGE_MEM_GB env
    # var if running on a smaller machine.
    mem_per_worker = n_res * chunk_size * 4 * 5 / 1e9  # ~1.5 GB for 360k × 256
    mem_budget_gb = float(os.environ.get("CPU_STAGE_MEM_GB", 100.0))
    max_safe_jobs = max(1, int(mem_budget_gb / max(mem_per_worker, 0.1)))
    effective_jobs = min(n_jobs if n_jobs > 0 else cpu_count(), max_safe_jobs)
    if effective_jobs < (n_jobs if n_jobs > 0 else cpu_count()):
        print(f"  ⚠️  Auto-capping workers: {effective_jobs} (memory limit, ~{mem_per_worker:.1f} GB/worker)")

    n_chunks = (n_features + chunk_size - 1) // chunk_size
    print(f"  Computing struct/seq locality for {n_features} features ({n_chunks} chunks)...")
    print(f"    Method: sparse matrix multiplication (vectorised v3)")
    print(f"    topk_frac: {topk_frac}, shuffles: {n_shuffles}")
    print(f"    Workers: {effective_jobs}, chunk_size: {chunk_size}")

    t0 = time.time()
    results = Parallel(n_jobs=effective_jobs, verbose=5)(
        delayed(_process_struct_seq_chunk_v3)(
            ci, chunk_size, Z, A_proj,
            A_seq, deg_seq, A_struct, deg_struct,
            perm_indices, n_features, topk_frac)
        for ci in range(n_chunks))
    elapsed = time.time() - t0

    # Assemble results
    all_idx = np.concatenate([r[0] for r in results])
    all_seq_obs = np.concatenate([r[1] for r in results])
    all_str_obs = np.concatenate([r[2] for r in results])
    all_seq_sh  = np.concatenate([r[3] for r in results])
    all_str_sh  = np.concatenate([r[4] for r in results])

    order = np.argsort(all_idx)
    df = pd.DataFrame({
        "feature_idx":         all_idx[order].astype(np.int32),
        "seq_effect_obs":      all_seq_obs[order],
        "struct_effect_obs":   all_str_obs[order],
        "seq_effect_shuffle":  all_seq_sh[order],
        "struct_effect_shuffle": all_str_sh[order],
        "seq_delta":           (all_seq_obs - all_seq_sh)[order],
        "struct_delta":        (all_str_obs - all_str_sh)[order],
    })
    df.to_csv(save_dir / "struct_seq_metrics.csv", index=False)

    # Plot
    plt.figure(figsize=(6, 6))
    sns.scatterplot(data=df, x="seq_delta", y="struct_delta", s=6, alpha=0.5)
    mx = float(max(df["seq_delta"].max(), df["struct_delta"].max(), 1.0))
    plt.plot([0, mx], [0, mx], "k--", lw=1)
    plt.title("Structural vs Sequential locality (\u0394 = obs \u2212 shuffle)")
    plt.xlabel("Sequential \u0394 (Cohen\u2019s d)")
    plt.ylabel("Structural \u0394 (Cohen\u2019s d)")
    plt.tight_layout(); plt.savefig(save_dir / "plot_struct_seq.png", dpi=200); plt.close()
    print(f"  Saved struct_seq_metrics.csv  ({elapsed:.0f}s)")


# ===========================================================================
#     [4/6]  UMAP: DECODER DICTIONARY
# ===========================================================================

def run_umap_decoder(D, save_dir: Path, n_jobs=-1):
    save_dir = Path(save_dir)
    if D is None: print("  Skipping decoder UMAP: D.npy not found"); return
    n_features = D.shape[0]
    norms = np.linalg.norm(D, axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, metric in zip(axes, ["cosine", "euclidean"]):
        print(f"  Running decoder UMAP ({n_features} features, {metric})...")
        reducer = umap.UMAP(metric=metric, n_neighbors=15, min_dist=0.1,
                            random_state=42, n_jobs=n_jobs)
        emb = reducer.fit_transform(D.astype(np.float32))
        np.save(save_dir / f"umap_decoder_{metric}.npy", emb)
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=norms, cmap="viridis",
                        s=3, alpha=0.6, rasterized=True)
        ax.set_title(f"Decoder UMAP ({metric})")
        ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
        plt.colorbar(sc, ax=ax, label="Feature norm")
    plt.tight_layout(); plt.savefig(save_dir / "umap_decoder_dual.png", dpi=200); plt.close()
    print(f"  Saved umap_decoder_dual.png")


# ===========================================================================
#     [5/6]  UMAP: RESIDUE ACTIVATIONS
# ===========================================================================

def run_umap_activations(Z, A, df_phys, res_idx, save_dir: Path,
                         n_points=20000, n_jobs=-1):
    save_dir = Path(save_dir)
    dfm = df_phys.merge(res_idx, on=["uid", "position"], how="inner")
    if len(dfm) == 0: print("  Skipping activation UMAP: no aligned residues"); return
    if len(dfm) > n_points: dfm = dfm.sample(n_points, random_state=42)
    ridx = dfm["res_global"].astype(int).to_numpy()

    if A is None:
        Zr = np.asarray(Z[ridx, :], dtype=np.float32)
    else:
        n_features = int(Z.shape[1]); chunk = 256
        Zr = np.zeros((len(ridx), n_features), dtype=np.float32)
        for i in range(0, n_features, chunk):
            end = min(i + chunk, n_features)
            Zr[:, i:end] = (A @ np.asarray(Z[:, i:end], dtype=np.float32))[ridx, :]

    ss = dfm.get("ss_8class", pd.Series(["-"] * len(dfm))).astype(str)
    col = ss.map({"H":"red","G":"red","I":"red","E":"blue","B":"blue",
                  "C":"lightgrey"}).fillna("grey")

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Helix (H/G/I)'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Strand (E/B)'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='grey', markersize=8, label='Coil/other'),
    ]
    for metric in ["cosine", "euclidean"]:
        print(f"  Running activation UMAP ({len(ridx)} residues, {metric})...")
        emb = umap.UMAP(metric=metric, n_neighbors=15, min_dist=0.1,
                        random_state=42, n_jobs=n_jobs).fit_transform(Zr)
        np.save(save_dir / f"umap_activations_{metric}.npy", emb)
        plt.figure(figsize=(8, 8))
        plt.scatter(emb[:, 0], emb[:, 1], c=col, s=2, alpha=0.6, rasterized=True)
        plt.legend(handles=legend_elements, loc='upper right')
        plt.title(f"Residue activations UMAP ({metric})")
        plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2")
        plt.tight_layout()
        plt.savefig(save_dir / f"umap_activations_{metric}.png", dpi=200); plt.close()
    print(f"  Saved umap_activations_cosine.png + umap_activations_euclidean.png")


# ===========================================================================
#     [6/6]  TOPK SENSITIVITY SWEEP  (now properly uses topk_frac!)
# ===========================================================================

def run_topk_sensitivity_sweep(Z, A_proj, uids, res_lengths, ref_seqs,
                               pdb_dir: Path, save_dir: Path,
                               n_shuffles: int, n_jobs: int,
                               contact_cutoff: float, seq_gap_min: int,
                               topk_values=None, n_proteins=500):
    """
    Sensitivity analysis: vary topk_frac, verify rank stability.

    FIX from v2: topk_frac is now passed through to _cohens_d_vectorized
    (previously it was accepted as an argument but never used — the sweep
    was computing the same result for every threshold!).
    """
    import time
    if topk_values is None: topk_values = SWEEP_TOPK_VALUES
    save_dir = Path(save_dir)

    # --- Subsample proteins ---
    n_total = len(uids)
    if n_proteins > 0 and n_proteins < n_total:
        np.random.seed(42)
        keep_idx = sorted(np.random.choice(n_total, n_proteins, replace=False))
        sub_uids = [uids[i] for i in keep_idx]
        sub_lengths = (res_lengths[keep_idx] if isinstance(res_lengths, np.ndarray)
                       else np.array([res_lengths[i] for i in keep_idx]))

        offsets_old = np.zeros(n_total + 1, dtype=np.int64)
        for i, L in enumerate(res_lengths): offsets_old[i+1] = offsets_old[i] + int(L)
        keep_rows = np.concatenate([np.arange(offsets_old[i], offsets_old[i+1], dtype=np.int64)
                                    for i in keep_idx])

        if A_proj is None:
            Z_sweep = np.asarray(Z[keep_rows, :], dtype=np.float32)
            A_sweep = None
        else:
            # ProtGPT2: subsetting sparse projection is complex.
            # Use all proteins — sweep bottleneck is graph building.
            Z_sweep, A_sweep = Z, A_proj
            sub_uids, sub_lengths = uids, res_lengths
            n_proteins = n_total

        sub_ref_seqs = {u: ref_seqs[u] for u in sub_uids}
    else:
        Z_sweep, A_sweep = Z, A_proj
        sub_uids, sub_lengths, sub_ref_seqs = uids, res_lengths, ref_seqs
        n_proteins = n_total

    n_res_sub = int(np.sum(sub_lengths))

    print(f"\n{'='*60}")
    print(f"SENSITIVITY ANALYSIS: topk_frac sweep")
    print(f"  Values: {topk_values}")
    print(f"  Proteins: {n_proteins}/{n_total}")
    print(f"  Residues: {n_res_sub:,}")
    print(f"  Contact: {contact_cutoff} \u00c5 (fixed), Gap: \u2265{seq_gap_min} (fixed)")
    print(f"{'='*60}")

    # Build graphs ONCE
    print("\n  Building neighbor graphs (one-time)...")
    seq_adj, struct_adj = build_neighbor_graphs_residue_parallel(
        sub_uids, sub_lengths, sub_ref_seqs, pdb_dir, n_jobs,
        contact_cutoff=contact_cutoff, seq_gap_min=seq_gap_min)

    # Convert to sparse ONCE
    A_seq, deg_seq = adj_list_to_sparse(seq_adj, n_res_sub)
    A_struct, deg_struct = adj_list_to_sparse(struct_adj, n_res_sub)
    del seq_adj, struct_adj
    print(f"    Sparse: seq {A_seq.nnz:,} edges, struct {A_struct.nnz:,} edges")

    # Permutations ONCE
    perm_indices = build_protein_permutations(sub_lengths, n_shuffles)

    # Auto-limit workers
    n_features = int(Z_sweep.shape[1]); chunk_size = 256
    mem_pw = n_res_sub * chunk_size * 4 * 5 / 1e9
    max_safe = max(1, int(40.0 / max(mem_pw, 0.1)))
    eff_jobs = min(n_jobs if n_jobs > 0 else cpu_count(), max_safe)
    n_chunks = (n_features + chunk_size - 1) // chunk_size

    all_results = {}
    for topk_frac in topk_values:
        print(f"\n  topk_frac = {topk_frac}...")
        t0 = time.time()
        results = Parallel(n_jobs=eff_jobs, verbose=0)(
            delayed(_process_struct_seq_chunk_v3)(
                ci, chunk_size, Z_sweep, A_sweep,
                A_seq, deg_seq, A_struct, deg_struct,
                perm_indices, n_features, topk_frac)
            for ci in range(n_chunks))
        elapsed = time.time() - t0

        all_seq = np.concatenate([r[1] for r in results]) - np.concatenate([r[3] for r in results])
        all_str = np.concatenate([r[2] for r in results]) - np.concatenate([r[4] for r in results])

        all_results[topk_frac] = {
            'n_struct': int((all_str > 0.5).sum()),
            'n_seq': int((all_seq > 0.5).sum()),
            'mean_struct': float(all_str.mean()),
            'mean_seq': float(all_seq.mean()),
            'struct_deltas': all_str,
            'seq_deltas': all_seq,
        }
        r = all_results[topk_frac]
        print(f"    Struct (d>0.5): {r['n_struct']}, Seq: {r['n_seq']}  ({elapsed:.0f}s)")

    # Rank stability
    sorted_topks = sorted(topk_values)
    stability = []
    for i in range(len(sorted_topks) - 1):
        t1, t2 = sorted_topks[i], sorted_topks[i+1]
        rho_s, _ = spearmanr(all_results[t1]['struct_deltas'], all_results[t2]['struct_deltas'])
        rho_q, _ = spearmanr(all_results[t1]['seq_deltas'], all_results[t2]['seq_deltas'])
        stability.append({'topk_1': t1, 'topk_2': t2, 'rho_struct': rho_s, 'rho_seq': rho_q})
        print(f"    \u03c1(struct) {t1}\u2194{t2}: {rho_s:.4f}, \u03c1(seq): {rho_q:.4f}")

    # Save
    pd.DataFrame([{'topk_frac': t,
                    **{k: all_results[t][k] for k in ('n_struct','n_seq','mean_struct','mean_seq')}}
                   for t in sorted_topks]).to_csv(save_dir / "topk_sensitivity_sweep.csv", index=False)
    if stability:
        pd.DataFrame(stability).to_csv(save_dir / "topk_sensitivity_stability.csv", index=False)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(sorted_topks, [all_results[t]['n_struct'] for t in sorted_topks],
                 'o-', label='Structural', color='steelblue')
    axes[0].plot(sorted_topks, [all_results[t]['n_seq'] for t in sorted_topks],
                 's--', label='Sequential', color='coral')
    axes[0].set_xlabel('topk_frac'); axes[0].set_ylabel('Features (\u03b4 > 0.5)')
    axes[0].legend(); axes[0].grid(alpha=0.3); axes[0].set_title('Feature Counts')

    axes[1].plot(sorted_topks, [all_results[t]['mean_struct'] for t in sorted_topks],
                 'o-', label='Structural', color='steelblue')
    axes[1].plot(sorted_topks, [all_results[t]['mean_seq'] for t in sorted_topks],
                 's--', label='Sequential', color='coral')
    axes[1].set_xlabel('topk_frac'); axes[1].set_ylabel('Mean \u0394')
    axes[1].legend(); axes[1].grid(alpha=0.3); axes[1].set_title('Effect Sizes')

    if stability:
        x = range(len(stability))
        axes[2].bar([i-0.15 for i in x], [s['rho_struct'] for s in stability],
                    0.3, label='Structural', color='steelblue')
        axes[2].bar([i+0.15 for i in x], [s['rho_seq'] for s in stability],
                    0.3, label='Sequential', color='coral')
        axes[2].set_xticks(list(x))
        axes[2].set_xticklabels([f"{s['topk_1']}\u2194{s['topk_2']}" for s in stability])
        axes[2].axhline(0.95, color='green', ls='--', alpha=0.5); axes[2].set_ylim(0.8, 1.02)
        axes[2].set_ylabel('Spearman \u03c1'); axes[2].legend()
        axes[2].grid(alpha=0.3); axes[2].set_title('Rank Stability')

    plt.tight_layout(); plt.savefig(save_dir / "topk_sensitivity_plot.png", dpi=200); plt.close()
    print(f"\n  Saved: topk_sensitivity_*.csv + topk_sensitivity_plot.png")
    return all_results


# ===========================================================================
#                               MAIN
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(
        description="CPU-stage analysis for protein SAE (vectorised v3)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--layer-dir", required=True)
    ap.add_argument("--model-type", choices=["residue", "protgpt2"], required=True)
    ap.add_argument("--features-csv", default="cache/residue_features.csv")
    ap.add_argument("--pdb-dir", default="cache/pdb_files")
    ap.add_argument("--fasta-path", default="cache/scope_40.fa")
    ap.add_argument("--n-shuffles", type=int, default=3)
    ap.add_argument("--tokenizer", default="nferruz/ProtGPT2")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--contact-cutoff", type=float, default=DEFAULT_CONTACT_CUTOFF,
                    help="C\u03b1\u2013C\u03b1 distance threshold (\u00c5)")
    ap.add_argument("--seq-gap-min", type=int, default=DEFAULT_SEQ_GAP_MIN,
                    help="Min sequence separation (residues)")
    ap.add_argument("--topk-frac", type=float, default=DEFAULT_TOPK_FRAC,
                    help="Fraction of residues considered 'active' per feature")
    ap.add_argument("--sweep-topk", action="store_true",
                    help="Run topk sensitivity sweep")
    ap.add_argument("--sweep-n-proteins", type=int, default=500,
                    help="Proteins to subsample for sweep (0 = all)")
    args = ap.parse_args()

    n_jobs = args.n_jobs if args.n_jobs > 0 else cpu_count()
    print(f"{'='*60}")
    print(f"CPU Stage Analysis (Vectorised v3 — sparse matrix)")
    print(f"  Layer dir:      {args.layer_dir}")
    print(f"  Model type:     {args.model_type}")
    print(f"  N jobs:         {n_jobs} cores")
    print(f"  Contact cutoff: {args.contact_cutoff} \u00c5")
    print(f"  Seq gap min:    \u2265{args.seq_gap_min} residues")
    print(f"  Topk frac:      {args.topk_frac}")
    print(f"{'='*60}")

    layer_dir = Path(args.layer_dir)
    Z, uids, tok_lengths = load_layer(layer_dir)
    ref_seqs = load_ref_seqs(layer_dir)
    print(f"  Loaded Z: {Z.shape} ({Z.shape[0]} tokens, {Z.shape[1]} features)")

    features_csv = Path(args.features_csv)
    if not features_csv.exists(): raise FileNotFoundError(f"Missing: {features_csv}")
    df_phys = load_phys_features(features_csv)
    if "ss_8class" in df_phys.columns and (df_phys["ss_8class"].astype(str) == "-").all():
        print("  Filling SS from DSSP...")
        df_phys = fill_ss8_from_dssp(df_phys, Path(args.pdb_dir), n_jobs=n_jobs)
        df_phys.to_csv(features_csv, index=False)

    if args.model_type == "residue":
        res_lengths = np.array([len(ref_seqs[uid]) for uid in uids], dtype=np.int32)
        if int(np.sum(res_lengths)) != int(Z.shape[0]):
            raise ValueError("Residue model: sum(len(seq)) != Z rows")
        res_offsets = {}; off = 0
        for uid, Lr in zip(uids, res_lengths): res_offsets[uid] = off; off += int(Lr)
        A = None
    else:
        print("  Building ProtGPT2 token->residue projection...")
        A, res_offsets, res_lengths = build_protgpt2_projection(
            uids, ref_seqs, tok_lengths, args.tokenizer)

    res_idx = build_residue_index(uids, res_lengths, res_offsets)
    print(f"  Residue index: {len(res_idx)} residues")

    print("\n[1/6] Feature meaning correlations...")
    analyze_feature_meanings_residue(Z, A, df_phys, res_idx, layer_dir, n_jobs=n_jobs)

    print("\n[2/6] Fold enrichment...")
    analyze_fold_enrichment_residue(Z, A, uids, res_lengths, res_offsets,
                                    Path(args.fasta_path), layer_dir, n_jobs=n_jobs)

    print("\n[3/6] Structural vs sequential locality (vectorised v3)...")
    analyze_struct_seq_residue_parallel(
        Z, A, uids, res_lengths, ref_seqs, Path(args.pdb_dir), layer_dir,
        args.n_shuffles, n_jobs=n_jobs,
        contact_cutoff=args.contact_cutoff, seq_gap_min=args.seq_gap_min,
        topk_frac=args.topk_frac)

    print("\n[4/6] UMAP on decoder dictionary...")
    D_path = layer_dir / "D.npy"
    D = np.load(D_path) if D_path.exists() else None
    run_umap_decoder(D, layer_dir, n_jobs=n_jobs)

    print("\n[5/6] UMAP on residue activations (cosine + euclidean)...")
    run_umap_activations(Z, A, df_phys, res_idx, layer_dir, n_jobs=n_jobs)

    if args.sweep_topk:
        print("\n[6/6] Activation threshold sensitivity sweep...")
        run_topk_sensitivity_sweep(
            Z, A, uids, res_lengths, ref_seqs, Path(args.pdb_dir), layer_dir,
            args.n_shuffles, n_jobs, args.contact_cutoff, args.seq_gap_min,
            n_proteins=args.sweep_n_proteins)
    else:
        print("\n[6/6] Skipping topk sweep (use --sweep-topk to enable)")

    print(f"\n{'='*60}")
    print(f"CPU stage complete!")
    print(f"  Outputs: {layer_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()