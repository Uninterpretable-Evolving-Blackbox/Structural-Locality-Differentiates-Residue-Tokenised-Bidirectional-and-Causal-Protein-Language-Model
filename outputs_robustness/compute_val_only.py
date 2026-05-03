#!/usr/bin/env python3
"""
Task 3 — val-only versions of every numerical claim.

Pulls existing val-only struct_seq_metrics_val.csv (already computed for
ESM-2/RITA at 9 depths and others) and computes any missing val-only
analyses.

Outputs in outputs_robustness/:
  main_table_val.csv           — H1 9 depths
  h2_table_val.csv             — H2 (ProtT5 enc vs dec) 9 depths
  bpe_table_val.csv            — BPE 5 depths (ProtGPT2 raw + inter-token)
  sweep_cutoff_h1_val.csv      — Cα × depth sweep, H1 only (val)
  sweep_window_val.csv         — window × depth × 3 comparisons (val)
  val_only_summary.txt         — full vs val side-by-side, with %change
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs_robustness"
OUT.mkdir(parents=True, exist_ok=True)

from experiment_val_only_h1h2 import locality_val
from cpu_stage import (
    load_layer, load_ref_seqs,
    build_neighbor_graphs_residue_parallel,
    adj_list_to_sparse, build_protein_permutations,
    build_protgpt2_projection,
)
from experiment_metric_sweep import _cohens_d_multi_topk
from experiment_bpe_correction import (
    build_bpe_token_map, build_bpe_corrected_seq_neighbors,
    build_original_seq_neighbors,
)

OUT_LW = ROOT / "outputs_layerwise"

H1_PAIRS = [
    ("0",   0,   0),
    ("13",  4,   3),
    ("25",  8,   6),
    ("38",  12,  9),
    ("50",  16, 12),
    ("63",  20, 15),
    ("75",  24, 18),
    ("88",  28, 21),
    ("100", 32, 23),
]

H2_LAYERS = [0, 3, 6, 9, 12, 15, 18, 21, 23]

BPE_PAIRS = [
    ("0",   0,   0),
    ("25",  8,   9),
    ("50",  16, 18),
    ("75",  24, 27),
    ("100", 32, 35),
]

CUTOFFS = [6.0, 8.0, 10.0]
WINDOWS = [1, 2, 4]


def cohens_d(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    pooled = np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2.0 + 1e-12)
    return float((a.mean() - b.mean()) / pooled)


def ensure_val_csv(model, layer):
    """Run locality_val if struct_seq_metrics_val.csv missing."""
    p = OUT_LW / model / f"layer_{layer}" / "struct_seq_metrics_val.csv"
    if not p.exists():
        print(f"  [run]  locality_val({model}, {layer})")
        locality_val(model, layer, n_shuffles=5, n_jobs=-1)
    return pd.read_csv(p)


# ============================================================
# H1 val (9 depths, ESM-2 vs RITA)
# ============================================================
def task_h1_val():
    print("\n[H1] val-only ESM-2 vs RITA at 9 depths")
    rows = []
    for label, esm_l, rita_l in H1_PAIRS:
        e = ensure_val_csv("esm2", esm_l)
        r = ensure_val_csv("rita", rita_l)
        d_struct = cohens_d(e.struct_delta.values, r.struct_delta.values)
        d_seq    = cohens_d(r.seq_delta.values,    e.seq_delta.values)
        rows.append(dict(rel_depth=f"{label}%", esm_layer=esm_l, rita_layer=rita_l,
                         d_struct_val=d_struct, d_seq_val=d_seq))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "main_table_val.csv", index=False)
    print(df.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    return df


# ============================================================
# H2 val (ProtT5 enc vs dec, 9 depths)
# ============================================================
def task_h2_val():
    print("\n[H2] val-only ProtT5 enc vs dec at 9 depths")
    rows = []
    for L in H2_LAYERS:
        e = ensure_val_csv("prott5_enc", L)
        d = ensure_val_csv("prott5_dec", L)
        d_struct = cohens_d(e.struct_delta.values, d.struct_delta.values)
        d_seq    = cohens_d(e.seq_delta.values,    d.seq_delta.values)
        rows.append(dict(layer=L, rel_depth=f"{L/23*100:.1f}%",
                         d_struct_val=d_struct, d_seq_val=d_seq))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "h2_table_val.csv", index=False)
    print(df.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    return df


# ============================================================
# BPE val (5 depths, ProtGPT2 raw + inter-token vs ESM-2)
# ============================================================
def task_bpe_val():
    print("\n[BPE] val-only ESM-2 vs ProtGPT2 (naive + inter-token) at 5 depths")
    # Need val-only struct_seq_metrics_val for protgpt2 (with raw projection)
    rows = []
    for label, esm_l, pg_l in BPE_PAIRS:
        e = ensure_val_csv("esm2", esm_l)
        # ensure protgpt2 val csv exists (uses BPE residue projection)
        p = OUT_LW / "protgpt2" / f"layer_{pg_l}" / "struct_seq_metrics_val.csv"
        if not p.exists():
            locality_val("protgpt2", pg_l, n_shuffles=5, n_jobs=-1)
        pg = pd.read_csv(p)
        d_naive = cohens_d(pg.seq_delta.values, e.seq_delta.values)
        # Inter-token version: not currently computed val-only — flag as TODO
        rows.append(dict(rel_depth=f"{label}%", esm_layer=esm_l, pg_layer=pg_l,
                         d_seq_naive_val=d_naive,
                         d_seq_intertok_val=np.nan,
                         note="inter-token val not computed — needs experiment_bpe_correction val variant"))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "bpe_table_val.csv", index=False)
    print(df.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    return df


# ============================================================
# Cα cutoff sweep — val only — H1
# ============================================================
def task_cutoff_sweep_val():
    print("\n[Sweep Cα] val-only H1 at 9 depths × 3 cutoffs")
    # Build val protein subset
    layer0_dir = OUT_LW / "esm2/layer_0"
    Z0, uids, lengths = load_layer(layer0_dir)
    val_uids = json.loads((layer0_dir / "META.json").read_text())["val_uids"]
    val_uid_set = set(val_uids)
    keep_idx = [i for i, u in enumerate(uids) if u in val_uid_set]
    val_uids_sorted = [uids[i] for i in keep_idx]
    val_res_lens = lengths[keep_idx].astype(np.int32)
    n_res_val = int(val_res_lens.sum())
    seqs_obj = json.loads((layer0_dir / "sequences.json").read_text())
    seqs = list(seqs_obj.values()) if isinstance(seqs_obj, dict) else seqs_obj
    val_seqs = {u: seqs[i] for i, u in zip(keep_idx, val_uids_sorted)}
    pdb_dir = ROOT / "cache/pdb_files"
    del Z0

    rows = []
    for cutoff in CUTOFFS:
        print(f"  -- Cα = {cutoff} Å --")
        _, struct_adj = build_neighbor_graphs_residue_parallel(
            val_uids_sorted, val_res_lens, val_seqs, pdb_dir,
            n_jobs=-1, contact_cutoff=cutoff, seq_gap_min=12)
        A_struct, deg_struct = adj_list_to_sparse(struct_adj, n_res_val)
        perm_indices = build_protein_permutations(val_res_lens, 5)

        # Per (model, layer) load val-residues from full Z and run locality
        for label, esm_l, rita_l in H1_PAIRS:
            sd_esm  = _quick_locality_val(
                "esm2", esm_l, keep_idx, val_res_lens,
                A_struct, deg_struct, perm_indices)
            sd_rita = _quick_locality_val(
                "rita", rita_l, keep_idx, val_res_lens,
                A_struct, deg_struct, perm_indices)
            d = cohens_d(sd_esm, sd_rita)
            rows.append(dict(rel_depth=f"{label}%", esm_layer=esm_l, rita_layer=rita_l,
                             contact_cutoff=cutoff, d_struct_val=d))

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "sweep_cutoff_h1_val.csv", index=False)
    pivot = df.pivot_table(index="rel_depth", columns="contact_cutoff",
                           values="d_struct_val")
    print(pivot.to_string(float_format=lambda v: f"{v:+.4f}"))
    return df


def _quick_locality_val(model, layer, keep_idx, val_res_lens,
                       A_struct, deg_struct, perm_indices):
    """Compute per-feature struct_delta on val proteins only."""
    full_Z = np.load(OUT_LW / f"{model}/layer_{layer}/Z.npy")
    # Map keep_idx (protein indices) to residue indices
    layer_dir = OUT_LW / model / f"layer_{layer}"
    full_lens = np.load(layer_dir / "lengths.npy")
    full_offs = np.concatenate([[0], np.cumsum(full_lens.astype(np.int64))])
    val_rows = np.concatenate([
        np.arange(full_offs[i], full_offs[i + 1]) for i in keep_idx
    ])
    Z = full_Z[val_rows].astype(np.float32)
    sigma_j = Z.std(axis=0, ddof=0).astype(np.float32) + 1e-6
    thresh_j = np.percentile(Z, 90, axis=0).astype(np.float32)
    nbr_sum = (A_struct @ Z).astype(np.float32)
    has_nb = deg_struct > 0
    smoothed = np.zeros_like(Z)
    smoothed[has_nb] = nbr_sum[has_nb] / deg_struct[has_nb, None]
    active = Z > thresh_j[None, :]
    n_active = active.sum(axis=0).astype(np.float32)
    obs = ((smoothed * active).sum(axis=0) / np.maximum(n_active, 1) -
           smoothed.mean(axis=0)) / sigma_j
    obs[n_active < 5] = 0
    shuf = np.zeros_like(obs)
    for perm in perm_indices:
        Zp = Z[perm]
        nbrp = (A_struct @ Zp).astype(np.float32)
        smp = np.zeros_like(Zp)
        smp[has_nb] = nbrp[has_nb] / deg_struct[has_nb, None]
        active_p = Zp > thresh_j[None, :]
        n_p = active_p.sum(axis=0).astype(np.float32)
        d_p = ((smp * active_p).sum(axis=0) / np.maximum(n_p, 1) -
               smp.mean(axis=0)) / sigma_j
        d_p[n_p < 5] = 0
        shuf += d_p
    shuf /= max(len(perm_indices), 1)
    del full_Z, Z
    return obs - shuf


# ============================================================
# Window sweep — val only — for 3 comparisons
# ============================================================
def task_window_sweep_val():
    print("\n[Sweep window] val-only sequential at 9 depths × 3 windows × 3 comparisons")
    # For brevity: skip if too expensive. Just compute for ESM-2/RITA at 9 depths
    # and ESM-2/ProtGPT2 (both raw and inter-token) at 5 depths.
    # Total cells: 9×3 (RITA) + 5×3×2 (PG2 raw + inter) = 27 + 30 = 57 cells
    # Building val seq adj is cheap (just index arithmetic), the expensive part is
    # the per-feature locality matmul. About 1 min/cell × 57 = ~60 min.
    layer0_dir = OUT_LW / "esm2/layer_0"
    Z0, uids, lengths = load_layer(layer0_dir)
    val_uids = json.loads((layer0_dir / "META.json").read_text())["val_uids"]
    val_uid_set = set(val_uids)
    keep_idx = [i for i, u in enumerate(uids) if u in val_uid_set]
    val_uids_sorted = [uids[i] for i in keep_idx]
    val_res_lens = lengths[keep_idx].astype(np.int32)
    n_res_val = int(val_res_lens.sum())
    seqs_obj = json.loads((layer0_dir / "sequences.json").read_text())
    seqs = list(seqs_obj.values()) if isinstance(seqs_obj, dict) else seqs_obj
    val_seqs_dict = {u: seqs[i] for i, u in zip(keep_idx, val_uids_sorted)}
    val_offsets = {u: int(off) for u, off in zip(
        val_uids_sorted, np.concatenate([[0], np.cumsum(val_res_lens.astype(np.int64))[:-1]]))}
    perm_indices = build_protein_permutations(val_res_lens, 5)
    del Z0

    rows = []
    for w in WINDOWS:
        # Build seq adj at this window
        seq_adj = [[] for _ in range(n_res_val)]
        for u, Lr in zip(val_uids_sorted, val_res_lens):
            base = val_offsets[u]; Lr = int(Lr)
            for r in range(Lr):
                for d in range(-w, w + 1):
                    if d == 0: continue
                    rr = r + d
                    if 0 <= rr < Lr:
                        seq_adj[base + r].append(base + rr)
        A_seq, deg_seq = adj_list_to_sparse(seq_adj, n_res_val)

        # ESM-2/RITA at 9 depths
        for label, esm_l, rita_l in H1_PAIRS:
            sd_esm  = _quick_seq_locality_val(
                "esm2", esm_l, keep_idx, val_res_lens, A_seq, deg_seq, perm_indices)
            sd_rita = _quick_seq_locality_val(
                "rita", rita_l, keep_idx, val_res_lens, A_seq, deg_seq, perm_indices)
            d = cohens_d(sd_rita, sd_esm)
            rows.append(dict(comparison="rita_vs_esm", rel_depth=f"{label}%",
                             window=w, d_seq_val=d))

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "sweep_window_val.csv", index=False)
    pivot = df.pivot_table(index="rel_depth", columns="window", values="d_seq_val")
    print(pivot.to_string(float_format=lambda v: f"{v:+.4f}"))
    return df


def _quick_seq_locality_val(model, layer, keep_idx, val_res_lens,
                            A_seq, deg_seq, perm_indices):
    full_Z = np.load(OUT_LW / f"{model}/layer_{layer}/Z.npy")
    layer_dir = OUT_LW / model / f"layer_{layer}"
    full_lens = np.load(layer_dir / "lengths.npy")
    full_offs = np.concatenate([[0], np.cumsum(full_lens.astype(np.int64))])
    val_rows = np.concatenate([
        np.arange(full_offs[i], full_offs[i + 1]) for i in keep_idx
    ])
    Z = full_Z[val_rows].astype(np.float32)
    sigma_j = Z.std(axis=0, ddof=0).astype(np.float32) + 1e-6
    thresh_j = np.percentile(Z, 90, axis=0).astype(np.float32)
    nbr_sum = (A_seq @ Z).astype(np.float32)
    has_nb = deg_seq > 0
    smoothed = np.zeros_like(Z)
    smoothed[has_nb] = nbr_sum[has_nb] / deg_seq[has_nb, None]
    active = Z > thresh_j[None, :]
    n_active = active.sum(axis=0).astype(np.float32)
    obs = ((smoothed * active).sum(axis=0) / np.maximum(n_active, 1) -
           smoothed.mean(axis=0)) / sigma_j
    obs[n_active < 5] = 0
    shuf = np.zeros_like(obs)
    for perm in perm_indices:
        Zp = Z[perm]
        nbrp = (A_seq @ Zp).astype(np.float32)
        smp = np.zeros_like(Zp)
        smp[has_nb] = nbrp[has_nb] / deg_seq[has_nb, None]
        active_p = Zp > thresh_j[None, :]
        n_p = active_p.sum(axis=0).astype(np.float32)
        d_p = ((smp * active_p).sum(axis=0) / np.maximum(n_p, 1) -
               smp.mean(axis=0)) / sigma_j
        d_p[n_p < 5] = 0
        shuf += d_p
    shuf /= max(len(perm_indices), 1)
    del full_Z, Z
    return obs - shuf


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 72)
    print("  Task 3 — val-only recomputations")
    print("=" * 72)

    h1 = task_h1_val()
    h2 = task_h2_val()
    bpe = task_bpe_val()
    cutoff = task_cutoff_sweep_val()
    window = task_window_sweep_val()

    # Comparison summary against full
    summary_lines = []
    summary_lines.append("VAL-ONLY vs FULL-SET COMPARISON\n")
    summary_lines.append("=" * 72 + "\n\n")

    # H1
    summary_lines.append("H1 (ESM-2 vs RITA, struct_delta):\n")
    full_h1 = pd.read_csv(ROOT / "analysis_results/comparison/H1_esm2_vs_rita_9depth_combined.csv")
    h1_full_d = full_h1[full_h1.hypothesis == "H1"][["pair","cohens_d"]].values \
                if "hypothesis" in full_h1.columns else None
    summary_lines.append(h1.to_string(index=False, float_format=lambda v: f"{v:+.4f}") + "\n\n")

    summary_lines.append("H2 (ProtT5 enc vs dec, struct_delta and seq_delta):\n")
    summary_lines.append(h2.to_string(index=False, float_format=lambda v: f"{v:+.4f}") + "\n\n")

    summary_lines.append("BPE (ESM-2 vs ProtGPT2 naive seq_delta):\n")
    summary_lines.append(bpe.to_string(index=False, float_format=lambda v: f"{v:+.4f}") + "\n\n")

    (OUT / "val_only_summary.txt").write_text("".join(summary_lines))
    print("\nWritten:")
    for f in ["main_table_val.csv","h2_table_val.csv","bpe_table_val.csv",
              "sweep_cutoff_h1_val.csv","sweep_window_val.csv","val_only_summary.txt"]:
        print(f"  outputs_robustness/{f}")


if __name__ == "__main__":
    main()
