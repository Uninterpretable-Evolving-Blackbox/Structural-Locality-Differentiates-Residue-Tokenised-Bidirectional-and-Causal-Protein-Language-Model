#!/usr/bin/env python3
"""
build_dataset.py — Reproducible protein dataset preparation
=============================================================

Downloads SCOPe 40% identity-filtered domains, fetches PDB structures,
runs DSSP for secondary structure / solvent accessibility, computes
burial (neighbour count), and saves everything to cache/.

Outputs:
    cache/sequences.json        — {uid: sequence} dict
    cache/residue_features.csv  — uid, position, ss_8class, sasa, neighbor_count
    cache/pdb_files/            — one PDB per domain (named by PDB ID)
    cache/scope_40.fa           — raw SCOPe FASTA (downloaded)

Usage:
    python build_dataset.py
    python build_dataset.py --min-length 50 --max-length 512 --max-proteins 1500

Design choices (justify in paper):
    - SCOPe 2.08, 40% sequence identity filter: removes redundancy while
      retaining fold diversity. 40% is standard for structural studies
      (Murzin 1995; Chandonia 2022).
    - Length filter 50–1024: excludes fragments (<50) and sequences that
      exceed typical PLM context windows (>1024).
    - One domain per PDB chain to avoid double-counting structures.
"""

import argparse
import json
import os
import re
import shutil
import sys
import warnings
from pathlib import Path

import ssl
import certifi

import numpy as np
import pandas as pd
import requests
from Bio import pairwise2

# Fix macOS SSL certificate issue
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
from Bio.PDB import PDBParser, PDBList
from Bio.PDB.DSSP import DSSP
from scipy.spatial import KDTree
from tqdm import tqdm

warnings.simplefilter("ignore")

# ============================================================
#                    CONSTANTS
# ============================================================

# SCOPe 2.08, genetic domain, 40% identity (astral)
SCOPE_URL = (
    "https://scop.berkeley.edu/downloads/scopeseq-2.08/"
    "astral-scopedom-seqres-gd-sel-gs-bib-40-2.08.fa"
)
SCOPE_FILENAME = "scope_40.fa"

CACHE_DIR = Path("cache")
PDB_DIR = CACHE_DIR / "pdb_files"

# Standard amino acids
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


# ============================================================
#                  1. DOWNLOAD SCOPe FASTA
# ============================================================

def download_scope_fasta(cache_dir: Path) -> Path:
    """Download SCOPe 40% identity-filtered FASTA if not present."""
    fasta_path = cache_dir / SCOPE_FILENAME
    if fasta_path.exists():
        print(f"✅ SCOPe FASTA already present: {fasta_path}")
        return fasta_path

    print(f"📥 Downloading SCOPe 40% FASTA from {SCOPE_URL}...")
    try:
        resp = requests.get(SCOPE_URL, timeout=120)
    except requests.exceptions.SSLError:
        warnings.warn("SSL verification failed — retrying without verify (macOS cert issue)")
        resp = requests.get(SCOPE_URL, timeout=120, verify=False)
    resp.raise_for_status()
    fasta_path.write_bytes(resp.content)
    print(f"   Saved to {fasta_path} ({len(resp.content) / 1024:.0f} KB)")
    return fasta_path


# ============================================================
#                  2. PARSE AND FILTER
# ============================================================

def parse_scope_fasta(fasta_path: Path):
    """Parse SCOPe FASTA into (uid, fold, sequence) tuples."""
    entries = []
    uid, fold, seq_lines = None, None, []

    with open(fasta_path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if uid is not None:
                    entries.append((uid, fold, "".join(seq_lines)))
                parts = line[1:].split()
                uid = parts[0]  # e.g. d1a2ba_
                fold = ".".join(parts[1].split(".")[:2]) if len(parts) >= 2 else "unknown"
                seq_lines = []
            else:
                seq_lines.append(line.strip())
        if uid is not None:
            entries.append((uid, fold, "".join(seq_lines)))

    return entries


def filter_entries(entries, min_length=50, max_length=1024, max_proteins=0):
    """Filter by length, valid AAs, and optionally cap count."""
    filtered = []
    reasons = {"short": 0, "long": 0, "non_standard": 0, "duplicate_pdb": 0}
    seen_pdbs = set()

    for uid, fold, seq in entries:
        seq = seq.upper()  # SCOPe FASTA uses lowercase
        if len(seq) < min_length:
            reasons["short"] += 1
            continue
        if len(seq) > max_length:
            reasons["long"] += 1
            continue
        if not all(c in VALID_AA for c in seq):
            reasons["non_standard"] += 1
            continue

        # One domain per PDB ID to avoid structural double-counting
        pdb_id = uid[1:5].lower()
        if pdb_id in seen_pdbs:
            reasons["duplicate_pdb"] += 1
            continue
        seen_pdbs.add(pdb_id)

        filtered.append((uid, fold, seq))  # seq is already uppercased above

    print(f"   Filtering: {len(entries)} → {len(filtered)} domains")
    for reason, count in reasons.items():
        if count > 0:
            print(f"     Removed {count} ({reason})")

    if 0 < max_proteins < len(filtered):
        # Stratified subsample: preserve fold diversity
        np.random.seed(42)
        fold_groups = {}
        for uid, fold, seq in filtered:
            fold_groups.setdefault(fold, []).append((uid, fold, seq))

        # Round-robin from each fold
        sampled = []
        folds = sorted(fold_groups.keys())
        idx = {f: 0 for f in folds}
        while len(sampled) < max_proteins:
            added_any = False
            for f in folds:
                if len(sampled) >= max_proteins:
                    break
                if idx[f] < len(fold_groups[f]):
                    sampled.append(fold_groups[f][idx[f]])
                    idx[f] += 1
                    added_any = True
            if not added_any:
                break

        print(f"   Subsampled to {len(sampled)} proteins ({len(set(s[1] for s in sampled))} folds)")
        filtered = sampled

    return filtered


# ============================================================
#               3. DOWNLOAD PDB FILES
# ============================================================

def download_pdbs(entries, pdb_dir: Path, max_retries=2):
    """Download PDB files for each domain."""
    pdb_dir.mkdir(parents=True, exist_ok=True)
    pdb_ids = sorted(set(uid[1:5].lower() for uid, _, _ in entries))

    existing = set(p.stem for p in pdb_dir.glob("*.pdb"))
    to_download = [pid for pid in pdb_ids if pid not in existing]

    if not to_download:
        print(f"✅ All {len(pdb_ids)} PDB files present")
        return

    print(f"📥 Downloading {len(to_download)} PDB files...")
    failed = []
    for pdb_id in tqdm(to_download, desc="PDB download"):
        success = False
        for attempt in range(max_retries):
            try:
                url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
                try:
                    resp = requests.get(url, timeout=30)
                except requests.exceptions.SSLError:
                    resp = requests.get(url, timeout=30, verify=False)
                if resp.status_code == 200:
                    (pdb_dir / f"{pdb_id}.pdb").write_bytes(resp.content)
                    success = True
                    break
            except requests.RequestException:
                continue
        if not success:
            failed.append(pdb_id)

    if failed:
        print(f"   ⚠️ Failed to download {len(failed)} PDBs: {failed[:10]}...")


# ============================================================
#            4. STRUCTURAL FEATURE EXTRACTION
# ============================================================

def _find_dssp_binary():
    for b in ("mkdssp", "dssp"):
        if shutil.which(b):
            return b
    return None


def _extract_chain_ca(chain):
    """Extract Cα sequence and coordinates from a PDB chain."""
    aa3 = {
        "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
        "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
        "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
        "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
    }
    seq, coords = [], []
    for res in chain:
        if res.id[0] != " " or "CA" not in res:
            continue
        seq.append(aa3.get(res.get_resname().upper(), "X"))
        coords.append(res["CA"].get_coord())
    return "".join(seq), np.array(coords, dtype=np.float32) if coords else np.zeros((0, 3))


def extract_structural_features(entries, pdb_dir: Path, dssp_bin=None):
    """Extract SS, SASA, burial for each residue."""
    parser = PDBParser(QUIET=True)
    rows = []

    if dssp_bin is None:
        dssp_bin = _find_dssp_binary()
    has_dssp = dssp_bin is not None

    if not has_dssp:
        print("⚠️ DSSP not found — SS and SASA will be placeholder values.")
        print("   Install mkdssp: conda install -c salilab dssp  OR  brew install dssp")

    for uid, fold, seq in tqdm(entries, desc="Structural features"):
        pdb_id = uid[1:5].lower()
        pdb_path = pdb_dir / f"{pdb_id}.pdb"
        L = len(seq)

        # Default: placeholder values
        ss_list = ["-"] * L
        sasa_list = [0.0] * L
        neighbor_count = [0] * L

        if pdb_path.exists():
            try:
                structure = parser.get_structure(uid, str(pdb_path))
                model = next(iter(structure))

                # Find best-matching chain
                best_chain, best_score, best_coords = None, -1, None
                for chain in model:
                    cseq, ccoords = _extract_chain_ca(chain)
                    if len(cseq) < 10:
                        continue
                    aln = pairwise2.align.globalms(
                        seq, cseq, 2, -1, -5, -0.5, one_alignment_only=True
                    )
                    if aln and aln[0].score > best_score:
                        best_chain = chain
                        best_score = aln[0].score
                        best_coords = ccoords

                if best_chain is not None and best_coords is not None:
                    # Align coordinates to sequence
                    cseq, _ = _extract_chain_ca(best_chain)
                    aln = pairwise2.align.globalms(
                        seq, cseq, 2, -1, -5, -0.5, one_alignment_only=True
                    )[0]
                    coords_aligned = np.full((L, 3), np.nan, dtype=np.float32)
                    ri, ci = -1, -1
                    for a, b in zip(aln.seqA, aln.seqB):
                        if a != "-":
                            ri += 1
                        if b != "-":
                            ci += 1
                        if a != "-" and b != "-" and ri < L and ci < len(best_coords):
                            coords_aligned[ri] = best_coords[ci]

                    # Burial: count Cα within 10Å
                    valid = ~np.isnan(coords_aligned).any(axis=1)
                    valid_idx = np.where(valid)[0]
                    if len(valid_idx) >= 2:
                        tree = KDTree(coords_aligned[valid])
                        for i, vi in enumerate(valid_idx):
                            nbrs = tree.query_ball_point(coords_aligned[vi], r=10.0)
                            neighbor_count[vi] = len(nbrs) - 1  # exclude self

                    # DSSP
                    if has_dssp:
                        try:
                            dssp = DSSP(model, str(pdb_path), dssp=dssp_bin)
                            chain_id = best_chain.id
                            dssp_keys = sorted(
                                [k for k in dssp.keys() if k[0] == chain_id],
                                key=lambda k: (k[1][1], k[1][2]),
                            )
                            dssp_ss = [dssp[k][2] if dssp[k][2] else "-" for k in dssp_keys]
                            dssp_rsa = [float(dssp[k][3]) if dssp[k][3] is not None else 0.0
                                       for k in dssp_keys]

                            # Map DSSP residues back to sequence positions via alignment
                            ri, ci = -1, -1
                            for a, b in zip(aln.seqA, aln.seqB):
                                if a != "-":
                                    ri += 1
                                if b != "-":
                                    ci += 1
                                if a != "-" and b != "-" and ri < L and ci < len(dssp_ss):
                                    ss_list[ri] = dssp_ss[ci]
                                    sasa_list[ri] = dssp_rsa[ci]
                        except Exception:
                            pass

            except Exception:
                pass

        for pos in range(L):
            rows.append({
                "uid": uid,
                "position": pos,
                "ss_8class": ss_list[pos],
                "sasa": sasa_list[pos],
                "neighbor_count": neighbor_count[pos],
            })

    return pd.DataFrame(rows)


# ============================================================
#                    5. SAVE OUTPUTS
# ============================================================

def save_outputs(entries, df_features, cache_dir: Path):
    """Save sequences.json and residue_features.csv."""
    # sequences.json: {uid: sequence}
    seq_dict = {uid: seq for uid, _, seq in entries}
    seq_path = cache_dir / "sequences.json"
    seq_path.write_text(json.dumps(seq_dict, indent=2))
    print(f"✅ Saved {len(seq_dict)} sequences → {seq_path}")

    # residue_features.csv
    csv_path = cache_dir / "residue_features.csv"
    df_features.to_csv(csv_path, index=False)
    print(f"✅ Saved {len(df_features)} residue features → {csv_path}")

    # Summary statistics
    n_proteins = len(entries)
    lengths = [len(seq) for _, _, seq in entries]
    folds = set(fold for _, fold, _ in entries)
    print(f"\n📊 Dataset summary:")
    print(f"   Proteins:  {n_proteins}")
    print(f"   Folds:     {len(folds)}")
    print(f"   Residues:  {sum(lengths)}")
    print(f"   Length:     {np.min(lengths)}–{np.max(lengths)} (mean {np.mean(lengths):.0f})")

    # Save summary for paper
    summary = {
        "scope_version": "2.08",
        "identity_filter": "40%",
        "n_proteins": n_proteins,
        "n_folds": len(folds),
        "n_residues": sum(lengths),
        "length_min": int(np.min(lengths)),
        "length_max": int(np.max(lengths)),
        "length_mean": float(np.mean(lengths)),
        "length_median": float(np.median(lengths)),
    }
    (cache_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2))


# ============================================================
#                        MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser(
        description="Build protein dataset from SCOPe for SAE analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--min-length", type=int, default=50,
                    help="Minimum sequence length")
    ap.add_argument("--max-length", type=int, default=1024,
                    help="Maximum sequence length")
    ap.add_argument("--max-proteins", type=int, default=0,
                    help="Cap protein count (0 = no cap). Uses fold-stratified sampling.")
    ap.add_argument("--skip-pdb", action="store_true",
                    help="Skip PDB download (use existing files)")
    ap.add_argument("--skip-dssp", action="store_true",
                    help="Skip DSSP — fill SS/SASA with placeholders")
    args = ap.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Download SCOPe FASTA
    fasta_path = download_scope_fasta(CACHE_DIR)

    # 2. Parse and filter
    print(f"\n📖 Parsing SCOPe FASTA...")
    entries = parse_scope_fasta(fasta_path)
    print(f"   Found {len(entries)} domains")

    entries = filter_entries(
        entries,
        min_length=args.min_length,
        max_length=args.max_length,
        max_proteins=args.max_proteins,
    )

    if not entries:
        print("❌ No proteins passed filtering!")
        sys.exit(1)

    # 3. Download PDB files
    if not args.skip_pdb:
        download_pdbs(entries, PDB_DIR)
    else:
        print("⏭️  Skipping PDB download")

    # 4. Extract structural features
    dssp_bin = None if args.skip_dssp else _find_dssp_binary()
    print(f"\n🔬 Extracting structural features...")
    df_features = extract_structural_features(entries, PDB_DIR, dssp_bin=dssp_bin)

    # 5. Save
    save_outputs(entries, df_features, CACHE_DIR)

    print(f"\n{'='*60}")
    print(f"✅ Dataset build complete!")
    print(f"   Next: DEVICE=mps MODEL=esm2 python run_unsupervised.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
