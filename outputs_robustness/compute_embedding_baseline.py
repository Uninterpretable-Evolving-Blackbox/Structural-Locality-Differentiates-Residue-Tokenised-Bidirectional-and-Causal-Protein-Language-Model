#!/usr/bin/env python3
"""
Tasks 2a + 2b — embedding-baseline analyses for the L0 H1 effect.

GOAL: quantify what fraction of the +1.44 layer-0 cross-model d is
attributable to raw embedding geometry vs SAE-discovered features.

Two analyses, both on raw layer-0 transformer hidden states (no SAE):

  (A) Per-dim L (Task 2 as specified by prompt) — RUN WITH HEAVY CAVEATS.
      Treats each of D embedding dimensions as a "feature" and applies
      the SAE locality metric. Results are reported but the metric is
      not really meaningful on dense vectors (active-residue threshold
      collapses, basis is arbitrary, σ has different meaning). The
      number is included so we have it; it should not be the load-
      bearing comparison.

  (B) Neighbour cosine similarity in embedding space (alternative).
      For each protein: mean cos-sim between residue i's L0 embedding
      and its structural neighbours' L0 embeddings, minus mean cos-sim
      to a sequence-distance-matched non-neighbour set. Per-protein
      scalar → cross-model d → bootstrap over proteins. This IS a
      meaningful test of "is L0 embedding geometry structurally local".

Both share the L0 hidden-state extraction step (~30 min on MPS for
ESM-2 t33, similar for RITA-l). Cached to disk so repeats are free.

Outputs (in outputs_robustness/):
  raw_l0_esm2.npy, raw_l0_rita.npy        — fp16 cache, (n_residues, D)
  embedding_baseline_perdim.csv           — Task 2a, with caveats
  embedding_baseline_cosine.csv           — Task 2b, the meaningful one
  embedding_baseline_summary.txt          — both with interpretation
"""

import json
import os
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

from cpu_stage import (
    load_layer, load_ref_seqs,
    build_neighbor_graphs_residue_parallel,
    adj_list_to_sparse, build_protein_permutations,
)

N_BOOT = 1000
TOPK_FRAC = 0.10
N_SHUF = 5

# RITA's custom modeling code monkey-patch (transformers 5.x)
def _patch_pretrained():
    from transformers.modeling_utils import PreTrainedModel
    if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
        PreTrainedModel.all_tied_weights_keys = {}


def cohens_d(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    pooled = np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2.0 + 1e-12)
    return float((a.mean() - b.mean()) / pooled)


def extract_l0(model_name, save_path):
    """Extract layer-0 hidden state for all 1500 proteins, save fp16."""
    if save_path.exists():
        print(f"  [skip] {save_path} exists ({save_path.stat().st_size/1e6:.0f} MB)")
        return np.load(save_path)
    print(f"  extracting L0 for {model_name}...")
    t0 = time.time()
    _patch_pretrained()
    if "esm2" in model_name:
        from extract_embeddings import extract_esm2_embeddings as fn
    elif "rita" in model_name.lower() or "RITA" in model_name:
        from extract_embeddings import extract_rita_embeddings as fn
    else:
        raise ValueError(f"unknown model {model_name}")
    seqs_p = ROOT / "outputs_layerwise/esm2/layer_0/sequences.json"
    seqs_obj = json.loads(seqs_p.read_text())
    seqs = list(seqs_obj.values()) if isinstance(seqs_obj, dict) else seqs_obj
    out = fn(seqs, layers=[0])
    arr = out[0].astype(np.float16)
    np.save(save_path, arr)
    print(f"    {arr.shape} fp16, saved {save_path.stat().st_size/1e6:.0f} MB, "
          f"in {time.time()-t0:.0f}s")
    return arr


# ============================================================
# Task 2a — per-dim L on dense embeddings (with caveats)
# ============================================================
def perdim_locality(Z, A, deg, perm_indices):
    """Apply the SAE-locality metric treating each dim of Z as a feature.

    CAVEAT: this is the metric the prompt asks for. It does NOT measure
    what it would on sparse SAE features:
      - active = top decile of a continuous dimension is arbitrary thresholding
      - σ_d is the spread of a dense distribution, not a fired-vs-not magnitude
      - basis (= choice of d index) is arbitrary; the metric is not
        invariant to rotations of the embedding space
    Results below are reported with these caveats stamped on them.
    """
    n_res, n_feat = Z.shape
    Z = Z.astype(np.float32)
    sigma_d = Z.std(axis=0, ddof=0).astype(np.float32) + 1e-6
    thresh_d = np.percentile(Z, 100.0 * (1.0 - TOPK_FRAC), axis=0).astype(np.float32)

    nbr_sum = (A @ Z).astype(np.float32)
    has_nb = deg > 0
    smoothed = np.zeros_like(Z)
    smoothed[has_nb] = nbr_sum[has_nb] / deg[has_nb, None]

    active = Z > thresh_d[None, :]
    n_active = active.sum(axis=0).astype(np.float32)
    active_sum = (smoothed * active).sum(axis=0)
    active_mean = active_sum / np.maximum(n_active, 1)
    global_mean = smoothed.mean(axis=0)

    obs_d = (active_mean - global_mean) / sigma_d

    # Shuffle baseline (5 within-protein perms)
    shuf_d_acc = np.zeros(n_feat, dtype=np.float32)
    for perm in perm_indices:
        Zp = Z[perm]
        nbrp = (A @ Zp).astype(np.float32)
        smp = np.zeros_like(Zp)
        smp[has_nb] = nbrp[has_nb] / deg[has_nb, None]
        active_p = Zp > thresh_d[None, :]
        n_active_p = active_p.sum(axis=0).astype(np.float32)
        as_p = (smp * active_p).sum(axis=0)
        am_p = as_p / np.maximum(n_active_p, 1)
        gm_p = smp.mean(axis=0)
        shuf_d_acc += (am_p - gm_p) / sigma_d
    shuf_d_acc /= len(perm_indices)
    L_d = obs_d - shuf_d_acc
    return L_d


# ============================================================
# Task 2b — neighbour cosine similarity (the meaningful one)
# ============================================================
def per_protein_cosine_locality(Z, protein_offsets, ref_seqs, uids, pdb_dir,
                                contact_cutoff=8.0, seq_gap_min=12,
                                rng=None):
    """For each protein p, compute mean cos-sim of L0 embeddings between
    structural-neighbour pairs minus mean cos-sim between sequence-distance-
    matched non-neighbour pairs.

    Returns per-protein scalars (n_proteins,).
    """
    if rng is None:
        rng = np.random.default_rng(42)
    from scipy.spatial import KDTree
    Z = Z.astype(np.float32)
    # L2-normalise once for cosine via dot-product
    norms = np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12
    Zn = Z / norms

    n_proteins = len(protein_offsets) - 1
    res_lens = np.diff(protein_offsets)
    out = np.full(n_proteins, np.nan, dtype=np.float64)

    # Need PDB Cα coords per protein. Reuse existing structural-neighbour
    # graph build by calling its inner per-protein routine, OR rebuild
    # ad-hoc. Cleaner: load the precomputed structural adjacency.
    # We'll iterate by protein and pull i's neighbours from the global adj.
    # Instead of re-loading PDBs, use the same struct_adj_list build.
    res_lengths = res_lens.astype(np.int32)
    seq_adj, struct_adj = build_neighbor_graphs_residue_parallel(
        uids, res_lengths, ref_seqs, pdb_dir,
        n_jobs=-1, contact_cutoff=contact_cutoff, seq_gap_min=seq_gap_min)

    n_residues_total = int(res_lens.sum())
    for p in range(n_proteins):
        s, e = protein_offsets[p], protein_offsets[p + 1]
        Lp = e - s
        if Lp < 5:
            continue

        # Collect within-protein contact pairs from struct_adj (global indices)
        contact_pairs = []
        for i_global in range(s, e):
            for j_global in struct_adj[i_global]:
                if s <= j_global < e and j_global > i_global:
                    contact_pairs.append((i_global - s, j_global - s))
        if not contact_pairs:
            continue

        # Sequence-distance-matched non-contact pairs:
        # for each contact (i, j) with sep |i-j|, draw a random pair (i', j')
        # within the protein with the same sep that is NOT a contact.
        pairs_local_set = set(contact_pairs)
        Zp = Zn[s:e]
        contact_cosines = np.array([Zp[i] @ Zp[j] for i, j in contact_pairs])

        non_contact_cosines = []
        for (i, j) in contact_pairs:
            sep = abs(j - i)
            # Possible (a, b) with same sep: a ∈ [0, Lp-sep-1], b = a + sep
            n_candidates = Lp - sep
            if n_candidates <= 1:
                continue
            tries = 0
            while tries < 10:
                a = int(rng.integers(0, n_candidates))
                b = a + sep
                if (a, b) not in pairs_local_set and (b, a) not in pairs_local_set:
                    non_contact_cosines.append(Zp[a] @ Zp[b])
                    break
                tries += 1

        if non_contact_cosines:
            out[p] = float(contact_cosines.mean()) - float(np.mean(non_contact_cosines))

    return out


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 72)
    print("  Tasks 2a + 2b — embedding-baseline analyses for L0 H1 effect")
    print("=" * 72)

    # ---- Extract raw L0 hidden states (cached) ----
    print("\n[1] Extract L0 hidden states (cached on disk if available)")
    Z_esm  = extract_l0("facebook/esm2_t33_650M_UR50D", OUT / "raw_l0_esm2.npy")
    Z_rita = extract_l0("lightonai/RITA_l",            OUT / "raw_l0_rita.npy")

    # ---- Shared structural adjacency ----
    print("\n[2] Build shared L0 structural adjacency...")
    layer0_dir = ROOT / "outputs_layerwise/esm2/layer_0"
    Z0, uids, lengths = load_layer(layer0_dir)
    res_lengths = lengths.astype(np.int32)
    n_proteins = len(res_lengths)
    n_res_total = int(res_lengths.sum())
    protein_offsets = np.concatenate([[0], np.cumsum(res_lengths.astype(np.int64))])
    ref_seqs = load_ref_seqs(layer0_dir)
    pdb_dir = ROOT / "cache/pdb_files"
    del Z0
    _, struct_adj = build_neighbor_graphs_residue_parallel(
        uids, res_lengths, ref_seqs, pdb_dir,
        n_jobs=-1, contact_cutoff=8.0, seq_gap_min=12)
    A_struct, deg_struct = adj_list_to_sparse(struct_adj, n_res_total)
    perm_indices = build_protein_permutations(res_lengths, N_SHUF)
    print(f"  adjacency: {A_struct.nnz:,} edges; {n_proteins} proteins")

    # ============================================================
    # Task 2a — per-dim L
    # ============================================================
    print("\n[3] Task 2a — per-dim L (with caveats)")
    print("    NOTE: applies sparse-feature locality metric to dense embeddings.")
    print("    Reported with caveats; not meant as the load-bearing comparison.")
    L_esm = perdim_locality(Z_esm.astype(np.float32),
                            A_struct, deg_struct, perm_indices)
    print(f"    ESM-2: {len(L_esm)} dims, mean L = {L_esm.mean():+.4f}, "
          f"sd = {L_esm.std(ddof=1):.4f}")
    L_rita = perdim_locality(Z_rita.astype(np.float32),
                             A_struct, deg_struct, perm_indices)
    print(f"    RITA : {len(L_rita)} dims, mean L = {L_rita.mean():+.4f}, "
          f"sd = {L_rita.std(ddof=1):.4f}")

    d_perdim = cohens_d(L_esm, L_rita)
    u_p, p_perdim = stats.mannwhitneyu(L_esm, L_rita, alternative="greater")
    sae_d_l0 = 1.4402  # paper's L0 SAE-feature d
    print(f"    Cross-model d on raw embedding dims: {d_perdim:+.4f}  "
          f"(MW p={p_perdim:.2e})")
    print(f"    Compare: SAE-feature d at L0 = +1.4402")
    print(f"    Ratio  : raw / SAE = {d_perdim / sae_d_l0:.3f}")
    pd.DataFrame([
        {"model": "esm2",  "n_dims": len(L_esm),  "mean_L": float(L_esm.mean()),
         "sd_L": float(L_esm.std(ddof=1))},
        {"model": "rita",  "n_dims": len(L_rita), "mean_L": float(L_rita.mean()),
         "sd_L": float(L_rita.std(ddof=1))},
    ]).to_csv(OUT / "embedding_baseline_perdim.csv", index=False)
    np.save(OUT / "embedding_baseline_perdim_L_esm.npy", L_esm)
    np.save(OUT / "embedding_baseline_perdim_L_rita.npy", L_rita)

    # ============================================================
    # Task 2b — cosine similarity (the meaningful one)
    # ============================================================
    print("\n[4] Task 2b — neighbour cosine similarity in embedding space")
    print("    (per-protein delta between contact-pair cos and matched non-contact)")
    rng = np.random.default_rng(42)
    cos_loc_esm  = per_protein_cosine_locality(
        Z_esm,  protein_offsets, ref_seqs, uids, pdb_dir, rng=rng)
    cos_loc_rita = per_protein_cosine_locality(
        Z_rita, protein_offsets, ref_seqs, uids, pdb_dir, rng=rng)
    valid = ~(np.isnan(cos_loc_esm) | np.isnan(cos_loc_rita))
    print(f"    valid proteins: {valid.sum()} / {len(cos_loc_esm)}")
    print(f"    ESM-2: mean = {cos_loc_esm[valid].mean():+.4f}, "
          f"sd = {cos_loc_esm[valid].std(ddof=1):.4f}")
    print(f"    RITA : mean = {cos_loc_rita[valid].mean():+.4f}, "
          f"sd = {cos_loc_rita[valid].std(ddof=1):.4f}")
    d_cos = cohens_d(cos_loc_esm[valid], cos_loc_rita[valid])
    print(f"    Cross-model d (paired protein values): {d_cos:+.4f}")

    # Bootstrap over proteins
    rng_b = np.random.default_rng(43)
    valid_idx = np.where(valid)[0]
    boot_d = np.empty(N_BOOT, dtype=np.float64)
    for b in range(N_BOOT):
        sample = rng_b.choice(valid_idx, size=len(valid_idx), replace=True)
        boot_d[b] = cohens_d(cos_loc_esm[sample], cos_loc_rita[sample])
    ci_lo, ci_hi = np.percentile(boot_d, [2.5, 97.5])
    frac_pos = float((boot_d > 0).mean())
    print(f"    bootstrap d (B=1000): mean={boot_d.mean():+.4f}, "
          f"95% CI=[{ci_lo:+.4f}, {ci_hi:+.4f}], frac_pos={frac_pos:.3f}")

    pd.DataFrame([{
        "model_a": "esm2", "model_b": "rita",
        "n_proteins": int(valid.sum()),
        "mean_a": float(cos_loc_esm[valid].mean()),
        "mean_b": float(cos_loc_rita[valid].mean()),
        "d_point": d_cos, "d_boot_mean": float(boot_d.mean()),
        "ci_low": float(ci_lo), "ci_high": float(ci_hi),
        "frac_pos": frac_pos,
    }]).to_csv(OUT / "embedding_baseline_cosine.csv", index=False)
    np.save(OUT / "embedding_baseline_cosine_per_protein_esm.npy", cos_loc_esm)
    np.save(OUT / "embedding_baseline_cosine_per_protein_rita.npy", cos_loc_rita)
    np.save(OUT / "embedding_baseline_cosine_boot_d.npy", boot_d)

    # ---- Summary text ----
    summary = f"""
Embedding-baseline analyses for the L0 H1 effect
================================================

Goal: quantify what fraction of the layer-0 cross-model d (paper +1.44)
is due to raw embedding geometry vs SAE-discovered features.

(A) Per-dim L (Task 2 as specified) — WITH CAVEATS
    Sparse-feature locality metric applied to dense embedding dims:
      ESM-2 ({len(L_esm)} dims): mean L = {L_esm.mean():+.4f}, sd = {L_esm.std(ddof=1):.4f}
      RITA  ({len(L_rita)} dims): mean L = {L_rita.mean():+.4f}, sd = {L_rita.std(ddof=1):.4f}
    Cross-model d on dim distributions: {d_perdim:+.4f}
    Paper's SAE-feature d at L0:        +1.4402

    Caveats (DO NOT report this number without these):
      - "Active = top decile" is arbitrary thresholding of a continuous-valued
        coordinate axis, not a meaningful "feature firing" event.
      - Embedding-space basis is arbitrary; the metric is not rotation-invariant.
      - σ_d is a continuous-distribution spread, not a "fired-vs-not" magnitude.
    Interpretation if reporting at all: the per-dim L isn't measuring what it
    measures on SAE features, so the ratio raw/SAE = {d_perdim/sae_d_l0:.3f} is
    not directly interpretable as "fraction of L0 effect due to embedding".

(B) Neighbour cosine similarity in embedding space (THE MEANINGFUL TEST)
    For each protein p: mean cos-sim between contact pairs (Cα<8Å, sep≥12)
    minus mean cos-sim between sequence-distance-matched non-contact pairs.
    Per-protein scalars; cross-model d on the two distributions; bootstrap
    over proteins (B=1000, paired across models).

      ESM-2 mean cos-locality: {cos_loc_esm[valid].mean():+.4f}
      RITA  mean cos-locality: {cos_loc_rita[valid].mean():+.4f}
      Cross-model d (point):   {d_cos:+.4f}
      95% bootstrap CI:        [{ci_lo:+.4f}, {ci_hi:+.4f}]
      frac_pos:                {frac_pos:.3f}
      n_proteins (valid):      {int(valid.sum())}

INTERPRETATION:
  - If d_cosine ≈ +1.44: L0 H1 effect IS embedding geometry; SAE inherits.
  - If d_cosine ≈ 0:     L0 H1 effect not visible in raw embeddings; SAE found something.
  - In between:          partial decomposition of the L0 effect.

  Observed: d_cosine = {d_cos:+.4f}, vs SAE-feature d at L0 = +1.4402.
  Ratio = {d_cos/sae_d_l0:.3f}.
"""
    (OUT / "embedding_baseline_summary.txt").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
