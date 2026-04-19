#!/usr/bin/env python3
"""
smoke_test.py — fast offline pipeline sanity check
====================================================

CPU only. No GPU, no network, no real data, no PDB files.
Runs in ~30–60 seconds. Safe to run in parallel with build_dataset.py.

Verifies the post-edit pipeline by:
  1. Importing every active module without crashing.
  2. Checking that today's hyperparameter constants are the post-edit values.
  3. Exercising helper functions (split, row extraction, epoch scaling).
  4. Training a tiny SparseAutoencoder on synthetic low-rank data and
     checking that train + val explained variance both look sane.
  5. Round-tripping extract_sae_features (Z shape + L0 ≈ k_sparse).
  6. Building a synthetic all_data dict with a known positive ESM-2 trend
     and running analyze_hypotheses.run_cross_model end-to-end. Confirms
     the new per-feature H5 Spearman path recovers the planted trend.
  7. Calling --help on cpu_stage.py and analyze_hypotheses.py.

Exits non-zero on any failure.
"""

import os
import sys
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
import pandas as pd

# Force deterministic small CPU run
torch.manual_seed(0)
np.random.seed(0)

OK = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"


def section(title: str):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


# ---------------------------------------------------------------
# 1. Constants & helpers
# ---------------------------------------------------------------
section("[1] Constants and helpers")

from run_unsupervised import (
    make_protein_split, rows_for_proteins, auto_epochs,
    K_SPARSE, K_AUX, DEAD_THRESHOLD, SPLIT_SEED, VAL_FRACTION, EXPANSION,
)

assert K_SPARSE == int(os.environ.get("K_SPARSE", 256)), f"K_SPARSE = {K_SPARSE}"
assert K_AUX == 64, f"K_AUX = {K_AUX}, expected 64"
assert DEAD_THRESHOLD == 1_000_000, f"DEAD_THRESHOLD = {DEAD_THRESHOLD}"
assert SPLIT_SEED == 42 and VAL_FRACTION == 0.10
assert EXPANSION == int(os.environ.get("EXPANSION", 8)), f"EXPANSION = {EXPANSION}"
print(f"  {OK} K_SPARSE={K_SPARSE}  K_AUX={K_AUX}  DEAD={DEAD_THRESHOLD:,}")
print(f"  {OK} SPLIT_SEED={SPLIT_SEED}  VAL_FRACTION={VAL_FRACTION}  EXPANSION={EXPANSION}")

tr1, va1 = make_protein_split(1500, 0.10, 42)
tr2, va2 = make_protein_split(1500, 0.10, 42)
assert tr1 == tr2 and va1 == va2, "make_protein_split not deterministic"
assert len(tr1) == 1350 and len(va1) == 150
assert set(tr1).isdisjoint(set(va1))
print(f"  {OK} make_protein_split: deterministic, 1350/150, no overlap")

lengths = np.array([10, 20, 15, 5, 30])
offsets = np.cumsum([0] + lengths.tolist()).astype(np.int64)
rows = rows_for_proteins(offsets, [1, 3])
assert rows.tolist() == list(range(10, 30)) + list(range(45, 50))
print(f"  {OK} rows_for_proteins: 25 rows extracted correctly")

assert auto_epochs(241_000) == 60
assert auto_epochs(50_000) == 40
print(f"  {OK} auto_epochs: 241k→60, 50k→40")


# ---------------------------------------------------------------
# 2. Train SAE on synthetic low-rank data
# ---------------------------------------------------------------
section("[2] SparseAutoencoder training (CPU, synthetic)")

from sae import SparseAutoencoder
from train_sae import train_sae, compute_explained_variance, extract_sae_features

DIM = 64
N = 4000
N_BASIS = 8

# Synthetic dictionary: 8 atoms in R^64, sparse mixture per token (k=2)
dictionary = np.random.randn(N_BASIS, DIM).astype(np.float32)
codes = np.zeros((N, N_BASIS), dtype=np.float32)
for i in range(N):
    active = np.random.choice(N_BASIS, 2, replace=False)
    codes[i, active] = np.abs(np.random.randn(2)).astype(np.float32)
X_full = (codes @ dictionary + 0.02 * np.random.randn(N, DIM)).astype(np.float32)

# Honest train/val split (already protein-level analog: separate rows)
X_train = X_full[:3000]
X_val = X_full[3000:]

print(f"  Training tiny SAE: {X_train.shape[0]} train tokens, {X_val.shape[0]} val tokens")
sae = train_sae(
    X_train,
    input_dim=DIM,
    device="cpu",
    epochs=12,
    lr=1e-3,
    expansion=4,        # hidden_dim = 256
    k_sparse=8,
    k_aux=4,
    dead_threshold=500,
    batch_size=256,
    log_interval=6,
)
assert isinstance(sae, SparseAutoencoder)
assert sae.k_sparse == 8, f"k_sparse not threaded: {sae.k_sparse}"
assert sae.k_aux == 4, f"k_aux not threaded: {sae.k_aux}"
assert sae.dead_threshold == 500, f"dead_threshold not threaded: {sae.dead_threshold}"
print(f"  {OK} train_sae returned SparseAutoencoder with k_sparse=8 k_aux=4 dead_threshold=500")


# ---------------------------------------------------------------
# 3. Explained variance — train vs val
# ---------------------------------------------------------------
section("[3] compute_explained_variance on train + val")

train_ev = compute_explained_variance(sae, X_train, device="cpu")
val_ev = compute_explained_variance(sae, X_val, device="cpu")
gap = train_ev - val_ev
print(f"  train_ev = {train_ev:.4f}")
print(f"  val_ev   = {val_ev:.4f}")
print(f"  gap      = {gap:+.4f}")
assert -0.5 < train_ev <= 1.0, f"train_ev out of range: {train_ev}"
assert -0.5 < val_ev <= 1.0, f"val_ev out of range: {val_ev}"
assert abs(gap) < 0.5, f"gap suspicious: {gap}"
print(f"  {OK} both EV values are in a sensible range and gap is small")


# ---------------------------------------------------------------
# 4. Feature extraction
# ---------------------------------------------------------------
section("[4] extract_sae_features round-trip")

Z, D = extract_sae_features(sae, X_full, device="cpu")
assert Z.shape == (N, DIM * 4), f"Z shape wrong: {Z.shape}"
assert D.shape == (DIM * 4, DIM), f"D shape wrong: {D.shape}"
mean_l0 = float((Z > 0).sum(axis=1).mean())
print(f"  Z shape: {Z.shape}  D shape: {D.shape}")
print(f"  mean L0: {mean_l0:.2f}  (expect ≤8 — TopK guarantees)")
assert mean_l0 <= 8.0 + 1e-6, f"L0 exceeds k_sparse: {mean_l0}"
assert mean_l0 > 0, "all zeros"
print(f"  {OK} Z + D shapes correct, L0 within TopK budget")


# ---------------------------------------------------------------
# 5. analyze_hypotheses.run_cross_model on synthetic all_data
# ---------------------------------------------------------------
section("[5] analyze_hypotheses end-to-end (synthetic, planted ESM-2 trend)")

from analyze_hypotheses import run_cross_model


def make_fake_layer(model: str, layer: int, n_feat: int, struct_trend: float):
    """Synthesize one layer's worth of CSV-equivalent DataFrames."""
    rng = np.random.RandomState(layer * 7919 + (hash(model) & 0xFFFF))
    base = struct_trend * (layer / 32.0)  # planted positive trend
    ss = pd.DataFrame({
        "feature_idx": np.arange(n_feat),
        "struct_delta": rng.randn(n_feat).astype(np.float32) * 0.5 + base,
        "seq_delta":    rng.randn(n_feat).astype(np.float32) * 0.5,
    })
    ip = pd.DataFrame({
        "feature_idx": np.arange(n_feat),
        "q_helix":  rng.uniform(0, 1, n_feat).astype(np.float32),
        "q_strand": rng.uniform(0, 1, n_feat).astype(np.float32),
        "q_burial": rng.uniform(0, 1, n_feat).astype(np.float32),
        "corr_helix":  rng.randn(n_feat).astype(np.float32),
        "corr_strand": rng.randn(n_feat).astype(np.float32),
        "corr_burial": rng.randn(n_feat).astype(np.float32),
    })
    fe = pd.DataFrame({"feature_idx": [0], "fold": ["a.1"], "enrichment": [6.0]})
    return f"{model}_layer_{layer}", {"ss": ss, "ip": ip, "fe": fe}


all_data = {}
plan = {
    "esm2":     ([0, 8, 16, 24, 32], 0.30),     # planted positive trend
    "protgpt2": ([0, 9, 18, 27, 35], 0.00),     # no trend
}
for model, (layers, trend) in plan.items():
    for layer in layers:
        lab, d = make_fake_layer(model, layer, n_feat=200, struct_trend=trend)
        all_data[lab] = d

# Mimic the per-layer summary rows that run_cross_model consumes
all_summaries = []
for lab, d in all_data.items():
    ss, ip = d["ss"], d["ip"]
    sig = ((ip["q_helix"] < 0.05) | (ip["q_strand"] < 0.05) | (ip["q_burial"] < 0.05)).sum()
    all_summaries.append({
        "label": lab,
        "n_features": len(ss),
        "struct_mean":   float(ss["struct_delta"].mean()),
        "struct_std":    float(ss["struct_delta"].std()),
        "struct_median": float(np.median(ss["struct_delta"])),
        "struct_pct_gt0": float((ss["struct_delta"] > 0).mean() * 100),
        "seq_mean":      float(ss["seq_delta"].mean()),
        "seq_std":       float(ss["seq_delta"].std()),
        "seq_median":    float(np.median(ss["seq_delta"])),
        "seq_pct_gt0":   float((ss["seq_delta"] > 0).mean() * 100),
        "pct_sig_helix":  5.0,
        "pct_sig_strand": 5.0,
        "pct_sig_burial": 5.0,
        "pct_any_interp": float(100 * sig / len(ss)),
        "pct_dead": 1.0,
        "pct_fold_enriched": 0.5,
        "n_folds_enriched": 1,
    })

model_layers = {"esm2": [0, 8, 16, 24, 32], "protgpt2": [0, 9, 18, 27, 35]}

with tempfile.TemporaryDirectory() as td:
    run_cross_model(all_summaries, all_data, model_layers, td)
    h5_csv = Path(td) / "H5_depth_trends.csv"
    assert h5_csv.exists(), "H5_depth_trends.csv not written"
    h5_df = pd.read_csv(h5_csv)
    print(f"  H5 columns: {list(h5_df.columns)}")
    for col in ("macro_struct_spearman_rho", "feature_struct_spearman_rho", "feature_n"):
        assert col in h5_df.columns, f"missing column: {col}"
    print(h5_df[["model", "macro_struct_spearman_rho",
                 "feature_struct_spearman_rho", "feature_n"]].to_string(index=False))

    esm = h5_df[h5_df["model"] == "esm2"].iloc[0]
    assert esm["feature_struct_spearman_rho"] > 0, "Planted ESM-2 trend NOT recovered"
    assert esm["feature_n"] == 1000, f"feature_n mismatch: {esm['feature_n']} (expect 5×200)"
    print(f"  {OK} per-feature path recovers planted trend "
          f"(esm2 ρ={esm['feature_struct_spearman_rho']:+.3f}, N={int(esm['feature_n'])})")


# ---------------------------------------------------------------
# 6. CLI surfaces
# ---------------------------------------------------------------
section("[6] --help on CLI scripts")

for script in ("cpu_stage.py", "analyze_hypotheses.py"):
    r = subprocess.run([sys.executable, script, "--help"],
                       capture_output=True, text=True, timeout=60)
    combined = (r.stdout + r.stderr).lower()
    if r.returncode == 0 and "usage" in combined:
        print(f"  {OK} {script}: argparse OK")
    else:
        print(f"  {FAIL} {script}: exit={r.returncode}")
        print(r.stderr[:500])
        sys.exit(1)


print(f"\n{'='*60}\n  {OK}  ALL SMOKE TESTS PASSED\n{'='*60}")
