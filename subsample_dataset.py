#!/usr/bin/env python3
"""
subsample_dataset.py — deterministic stratified subsample of SCOPe40 to N proteins
====================================================================================

Why
---
build_dataset.py was run without --max-proteins, producing the full SCOPe40
set (~11.7k proteins, ~2.24M residues). For the workshop paper we want a
1,500-protein subset that:
  • is deterministic (seed=42, fixed selection across runs);
  • is stratified by SCOPe fold so rare folds aren't lost;
  • prefers proteins with high DSSP coverage (cleaner labels);
  • excludes the ~2.5k proteins where DSSP completely failed;
  • leaves the ORIGINAL 11.7k cache intact (backed up to cache/full_*).

The protein-level train/val split (seed 42, 90/10) is then applied to the
subsampled set inside run_unsupervised.py.

Idempotent: re-running this script restores the originals first, then
re-subsamples. Safe to run multiple times.

Usage
-----
    python subsample_dataset.py --n 1500
    python subsample_dataset.py --n 1500 --min-coverage 0.80 --seed 42
    python subsample_dataset.py --restore       # put the full cache back
"""

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path("cache")
SEQ_PATH = CACHE / "sequences.json"
FEAT_PATH = CACHE / "residue_features.csv"
FASTA_PATH = CACHE / "scope_40.fa"
PDB_DIR = CACHE / "pdb_files"

# Backups (left untouched once written)
SEQ_BACKUP = CACHE / "full_sequences.json"
FEAT_BACKUP = CACHE / "full_residue_features.csv"


def restore_full_cache():
    """Restore originals from backup, no-op if no backup."""
    if SEQ_BACKUP.exists():
        shutil.copy2(SEQ_BACKUP, SEQ_PATH)
        print(f"  Restored {SEQ_PATH} from {SEQ_BACKUP}")
    if FEAT_BACKUP.exists():
        shutil.copy2(FEAT_BACKUP, FEAT_PATH)
        print(f"  Restored {FEAT_PATH} from {FEAT_BACKUP}")


def parse_fold_map(fasta_path: Path) -> dict:
    """uid → SCOPe fold (e.g. 'a.1') from FASTA headers."""
    fold = {}
    with open(fasta_path) as f:
        for line in f:
            if line.startswith(">"):
                parts = line[1:].split()
                if len(parts) >= 2:
                    uid = parts[0]
                    fold[uid] = ".".join(parts[1].split(".")[:2])
    return fold


def coverage_per_uid(df_feat: pd.DataFrame) -> pd.Series:
    """DSSP coverage fraction per UID."""
    ss = df_feat["ss_8class"].astype(str)
    df_feat = df_feat.assign(_filled=(ss != "-").astype(np.int8))
    cov = df_feat.groupby("uid")["_filled"].agg(["sum", "count"])
    return (cov["sum"] / cov["count"]).rename("coverage")


def stratified_sample(uids_pool: list, fold_of: dict, n_target: int, seed: int) -> list:
    """
    Stratified sample of `uids_pool` to size `n_target`, proportional to fold size.

    Each fold contributes round(n_target * fold_size / pool_size) proteins,
    rounded so the total is exactly n_target. Within a fold, proteins are
    sampled with a deterministic seed.
    """
    rng = np.random.RandomState(seed)
    by_fold = defaultdict(list)
    for u in uids_pool:
        by_fold[fold_of.get(u, "_unknown")].append(u)

    pool_n = len(uids_pool)
    # Target per fold (real-valued, then largest-remainder rounded)
    real_targets = {f: n_target * len(us) / pool_n for f, us in by_fold.items()}
    int_targets = {f: int(np.floor(t)) for f, t in real_targets.items()}
    fractional = sorted(
        [(real_targets[f] - int_targets[f], f) for f in by_fold],
        reverse=True,
    )
    deficit = n_target - sum(int_targets.values())
    for _, f in fractional[:deficit]:
        int_targets[f] += 1

    chosen = []
    for fold, members in by_fold.items():
        k = min(int_targets[fold], len(members))
        if k <= 0:
            continue
        idx = rng.choice(len(members), k, replace=False)
        chosen.extend(members[i] for i in idx)
    chosen.sort()
    return chosen


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("--n", type=int, default=1500, help="target #proteins")
    ap.add_argument("--min-coverage", type=float, default=0.80,
                    help="minimum DSSP coverage to be eligible")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed")
    ap.add_argument("--restore", action="store_true",
                    help="restore the full cache from backup and exit")
    args = ap.parse_args()

    if args.restore:
        print("Restoring full cache from backups...")
        restore_full_cache()
        print("Done.")
        return

    # ---- Snapshot originals once ----
    if not SEQ_BACKUP.exists():
        print(f"Backing up {SEQ_PATH} → {SEQ_BACKUP}")
        shutil.copy2(SEQ_PATH, SEQ_BACKUP)
    else:
        print(f"Backup already exists: {SEQ_BACKUP}  (re-using)")
    if not FEAT_BACKUP.exists():
        print(f"Backing up {FEAT_PATH} → {FEAT_BACKUP}")
        shutil.copy2(FEAT_PATH, FEAT_BACKUP)
    else:
        print(f"Backup already exists: {FEAT_BACKUP}  (re-using)")

    # Always read from the backup so re-running is idempotent
    sequences_full = json.loads(SEQ_BACKUP.read_text())
    if not isinstance(sequences_full, dict):
        raise RuntimeError(f"Expected dict in {SEQ_BACKUP}, got {type(sequences_full)}")
    df_feat = pd.read_csv(FEAT_BACKUP)

    print(f"\nLoaded full cache:")
    print(f"  proteins: {len(sequences_full)}")
    print(f"  residue rows: {len(df_feat)}")

    # ---- Compute eligibility ----
    cov = coverage_per_uid(df_feat)
    fold_map = parse_fold_map(FASTA_PATH)
    pdb_present = lambda u: (PDB_DIR / f"{str(u)[1:5].lower()}.pdb").exists()

    eligible = []
    n_no_pdb = n_no_fold = n_low_cov = 0
    for uid in sequences_full:
        if uid not in cov.index:
            continue
        if cov.loc[uid] < args.min_coverage:
            n_low_cov += 1
            continue
        if uid not in fold_map:
            n_no_fold += 1
            continue
        if not pdb_present(uid):
            n_no_pdb += 1
            continue
        eligible.append(uid)

    print(f"\nEligibility filter (coverage >= {args.min_coverage:.2f}):")
    print(f"  eligible:        {len(eligible)}")
    print(f"  excluded (low DSSP coverage): {n_low_cov}")
    print(f"  excluded (no fold info):       {n_no_fold}")
    print(f"  excluded (no PDB on disk):     {n_no_pdb}")

    if len(eligible) < args.n:
        raise RuntimeError(
            f"Eligible pool ({len(eligible)}) is smaller than target ({args.n}). "
            f"Lower --min-coverage."
        )

    # ---- Stratified sample ----
    chosen = stratified_sample(eligible, fold_map, args.n, args.seed)
    print(f"\nStratified sample: {len(chosen)} proteins (seed={args.seed})")

    folds_in_chosen = {fold_map.get(u, "?") for u in chosen}
    classes_in_chosen = {f.split(".")[0] for f in folds_in_chosen}
    print(f"  unique folds preserved: {len(folds_in_chosen)} / "
          f"{len({fold_map.get(u, '?') for u in eligible})}")
    print(f"  SCOPe classes preserved: {sorted(classes_in_chosen)}")

    # ---- Write subsampled cache files ----
    chosen_set = set(chosen)
    sub_sequences = {uid: sequences_full[uid] for uid in chosen}
    SEQ_PATH.write_text(json.dumps(sub_sequences, indent=2))

    sub_feat = df_feat[df_feat["uid"].isin(chosen_set)].copy()
    sub_feat.to_csv(FEAT_PATH, index=False)

    sub_residues = sum(len(s) for s in sub_sequences.values())
    sub_filled = (sub_feat["ss_8class"].astype(str) != "-").sum()
    sub_cov_pct = 100 * sub_filled / len(sub_feat)

    print(f"\n✅ Wrote subsampled cache:")
    print(f"  {SEQ_PATH}            ({len(sub_sequences)} proteins)")
    print(f"  {FEAT_PATH}  ({len(sub_feat)} residue rows)")
    print(f"  total residues: {sub_residues}")
    print(f"  DSSP coverage in subset: {sub_cov_pct:.1f}%")
    print(f"\nBackups remain at:")
    print(f"  {SEQ_BACKUP}")
    print(f"  {FEAT_BACKUP}")
    print(f"\nTo restore the full set later:  python subsample_dataset.py --restore")


if __name__ == "__main__":
    main()
