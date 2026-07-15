#!/usr/bin/env python3
"""
cluster_bootstrap.py — fold/superfamily cluster bootstrap for SCOPe-domain data
================================================================================

Why this module exists
----------------------
Every CI in the project (headline H1/H2 and the new extension experiments) is
currently a PROTEIN-level cluster bootstrap: the 1,500 SCOPe domains are
resampled as if independent. They are not — domains in the same SCOPe *fold*
(or superfamily) share ancestry and structural statistics, so protein-level
resampling commits pseudo-replication and yields CIs that are too narrow
(p-values too optimistic). The effective sample size is closer to the number of
folds (432) than the number of proteins (1,500).

This module provides ONE shared implementation of:
  1. `load_uid_clusters` / `clusters_for_uids` — the SCOPe clustering key
     (supersedes the three ad-hoc fold parsers in build_dataset.py,
     subsample_dataset.py, experiment_concept_f1.py — all of which compute the
     same `".".join(sccs[:2])` fold key).
  2. `make_cluster_weights` — a (B, n_proteins) integer weight matrix that the
     existing weight-driven bootstrap machinery (`w @ contribs`) consumes
     unchanged. Supports protein / fold / superfamily clustering and one- vs
     two-stage (hierarchical) resampling. `level="protein"` reproduces the
     original protein bootstrap exactly (each protein is its own cluster).
  3. `design_effect` — ICC, design effect Deff, and effective sample size n_eff,
     so the dependency is reported alongside n=1,500.

Determinism: all randomness flows through a caller-supplied numpy Generator.

Run `python cluster_bootstrap.py` to execute the built-in correctness test.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional

import numpy as np

# SCOPe sccs (e.g. "a.1.1.1") = class.fold.superfamily.family
_LEVEL_DEPTH = {"class": 1, "fold": 2, "superfamily": 3, "family": 4}


# ===========================================================================
#                       SCOPe CLUSTERING KEY
# ===========================================================================

def load_uid_clusters(fasta_path, level: str = "fold") -> Dict[str, str]:
    """uid -> SCOPe cluster string at `level` from a SCOPe FASTA.

    Header format: ">d1dlwa_ a.1.1.1 ...". The cluster key for level L is the
    first `_LEVEL_DEPTH[L]` dot-components of the sccs code (fold -> "a.1").

    uids whose header is missing/short are simply absent from the returned map;
    `clusters_for_uids` assigns them unique singleton clusters (conservative:
    an unknown domain becomes its own independent unit, never merged).
    """
    if level not in _LEVEL_DEPTH:
        raise ValueError(f"level must be one of {list(_LEVEL_DEPTH)}, got {level!r}")
    depth = _LEVEL_DEPTH[level]
    out: Dict[str, str] = {}
    with open(fasta_path) as f:
        for line in f:
            if not line.startswith(">"):
                continue
            parts = line[1:].split()
            if len(parts) < 2:
                continue
            uid = parts[0]
            sccs = parts[1].split(".")
            if len(sccs) >= depth:
                out[uid] = ".".join(sccs[:depth])
    return out


def clusters_for_uids(uids, fasta_path, level: str = "fold") -> np.ndarray:
    """Integer cluster id per uid (aligned to `uids` order).

    level="protein" -> every uid is its own cluster (reduces the bootstrap to
    the standard protein-level resample). Otherwise uses the SCOPe key, with a
    unique singleton id for any uid missing from the FASTA.
    """
    uids = [str(u) for u in uids]
    if level == "protein":
        return np.arange(len(uids), dtype=np.int64)

    uid_to_key = load_uid_clusters(fasta_path, level=level)
    key_to_int: Dict[str, int] = {}
    cluster_id = np.empty(len(uids), dtype=np.int64)
    next_id = 0
    for i, u in enumerate(uids):
        key = uid_to_key.get(u)
        if key is None:
            # singleton: unique key so it can never collide with a real cluster
            key = f"__singleton__{u}"
        if key not in key_to_int:
            key_to_int[key] = next_id
            next_id += 1
        cluster_id[i] = key_to_int[key]
    return cluster_id


def cluster_stats(cluster_id: np.ndarray, indices: Optional[np.ndarray] = None) -> dict:
    """n_clusters, n_units, mean/median cluster size over the in-split subset."""
    cid = cluster_id if indices is None else cluster_id[indices]
    _, counts = np.unique(cid, return_counts=True)
    return {
        "n_units": int(cid.size),
        "n_clusters": int(counts.size),
        "mean_cluster_size": float(counts.mean()),
        "median_cluster_size": float(np.median(counts)),
        "max_cluster_size": int(counts.max()),
    }


# ===========================================================================
#                    CLUSTER BOOTSTRAP WEIGHT MATRIX
# ===========================================================================

def make_cluster_weights(n_proteins: int, cluster_id: np.ndarray, indices: np.ndarray,
                         n_boot: int, rng: np.random.Generator,
                         two_stage: bool = True) -> np.ndarray:
    """(n_boot, n_proteins) int weight matrix for a cluster bootstrap.

    Resample whole clusters (folds) with replacement from the in-split subset.
    - two_stage=False (one-stage): a drawn cluster contributes ALL its in-split
      members once (keeps the cluster intact). Captures between-cluster variance.
    - two_stage=True (hierarchical): also resample proteins *within* each drawn
      cluster with replacement. Captures both between- and within-cluster
      variance (preferred; standard two-stage cluster bootstrap).

    Weights are integer multiplicities; the downstream statistic computes
    `w @ contribs` and divides by `w @ n_residues`, so a protein drawn twice
    counts double in both numerator and denominator (correct).

    With cluster_id = arange(n_proteins) (every protein its own cluster) this
    reduces to the standard protein-level bootstrap.
    """
    indices = np.asarray(indices, dtype=np.int64)
    members: Dict[int, list] = defaultdict(list)
    for p in indices:
        members[int(cluster_id[p])].append(int(p))
    member_arrays = {c: np.asarray(m, dtype=np.int64) for c, m in members.items()}
    clusters = np.fromiter(member_arrays.keys(), dtype=np.int64, count=len(member_arrays))

    W = np.zeros((n_boot, n_proteins), dtype=np.int32)
    n_clusters = clusters.size
    for b in range(n_boot):
        drawn = rng.choice(clusters, size=n_clusters, replace=True)
        if not two_stage:
            # one-stage: count how many times each cluster was drawn, add to all members
            uc, cc = np.unique(drawn, return_counts=True)
            for c, mult in zip(uc, cc):
                W[b, member_arrays[int(c)]] += int(mult)
        else:
            for c in drawn:
                m = member_arrays[int(c)]
                pick = rng.choice(m, size=m.size, replace=True)
                np.add.at(W[b], pick, 1)
    return W


def resample_indices_by_cluster(item_clusters, rng: np.random.Generator,
                                two_stage: bool = True) -> np.ndarray:
    """One cluster-bootstrap resample of positional indices 0..n-1.

    For simple mean-style bootstraps (e.g. steering specificity, faithfulness
    ΔCE) that resample a 1-D per-item array directly rather than via a weight
    matrix. `item_clusters[i]` is item i's cluster id. Returns an array of
    resampled positional indices (length varies under unequal cluster sizes).
    `item_clusters = arange(n)` reduces to the ordinary i.i.d. bootstrap.
    """
    item_clusters = np.asarray(item_clusters)
    n = item_clusters.size
    members: Dict[int, list] = defaultdict(list)
    for i in range(n):
        members[int(item_clusters[i])].append(i)
    member_arrays = {c: np.asarray(m, dtype=np.int64) for c, m in members.items()}
    clusters = np.fromiter(member_arrays.keys(), dtype=np.int64, count=len(member_arrays))
    drawn = rng.choice(clusters, size=clusters.size, replace=True)
    out = []
    for c in drawn:
        m = member_arrays[int(c)]
        out.append(rng.choice(m, size=m.size, replace=True) if two_stage else m)
    return np.concatenate(out) if out else np.empty(0, dtype=np.int64)


# ===========================================================================
#                    DESIGN EFFECT / EFFECTIVE SAMPLE SIZE
# ===========================================================================

def design_effect(values: np.ndarray, cluster_id: np.ndarray,
                  indices: Optional[np.ndarray] = None) -> dict:
    """One-way-random-effects ICC, design effect, and effective sample size.

    `values` is a per-protein scalar (e.g. per-protein mean L_struct contribution
    or a per-protein summary). ICC is estimated by the standard unbalanced
    one-way ANOVA moment estimator:
        ICC = (MSB - MSW) / (MSB + (m0 - 1) * MSW)
    Deff = 1 + (mean_cluster_size - 1) * ICC ;  n_eff = N / Deff.
    Negative ICC (anti-clustering) is reported as-is but clipped to 0 for n_eff.
    """
    values = np.asarray(values, dtype=np.float64)
    cid = np.asarray(cluster_id)
    if indices is not None:
        values = values[indices]
        cid = cid[indices]
    N = values.size
    groups, inv, counts = np.unique(cid, return_inverse=True, return_counts=True)
    k = groups.size
    if k < 2 or N <= k:
        return {"icc": float("nan"), "design_effect": float("nan"),
                "n_eff": float(N), "n_clusters": int(k), "n_units": int(N),
                "mean_cluster_size": float(N / max(k, 1))}

    grand = values.mean()
    group_means = np.zeros(k)
    np.add.at(group_means, inv, values)
    group_means /= counts
    ssb = float((counts * (group_means - grand) ** 2).sum())
    ssw = float(((values - group_means[inv]) ** 2).sum())
    msb = ssb / (k - 1)
    msw = ssw / (N - k)
    m0 = (N - (counts ** 2).sum() / N) / (k - 1)
    denom = msb + (m0 - 1) * msw
    icc = (msb - msw) / denom if denom > 0 else 0.0
    m_bar = N / k
    icc_eff = max(icc, 0.0)
    deff = 1.0 + (m_bar - 1.0) * icc_eff
    n_eff = N / deff if deff > 0 else float(N)
    return {"icc": float(icc), "design_effect": float(deff), "n_eff": float(n_eff),
            "n_clusters": int(k), "n_units": int(N), "mean_cluster_size": float(m_bar)}


# ===========================================================================
#                         CORRECTNESS TEST
# ===========================================================================

def correctness_test():
    """Sanity checks for the cluster bootstrap (run via `python cluster_bootstrap.py`)."""
    print("=== cluster_bootstrap correctness test ===")
    rng = np.random.default_rng(0)

    # (1) protein-level reduction: each protein its own cluster, two_stage.
    n = 40
    cid_protein = np.arange(n)
    idx = np.arange(n)
    W = make_cluster_weights(n, cid_protein, idx, n_boot=500, rng=rng, two_stage=True)
    row_sums = W.sum(axis=1)
    assert (row_sums == n).all(), "protein-level rows must each sum to N"
    # marginal: expected count per protein ~ 1, variance ~ Poisson(1)-like
    mean_count = W.mean()
    assert abs(mean_count - 1.0) < 0.05, f"protein-level mean weight {mean_count} != ~1"
    # ~1/e fraction of proteins unselected per resample (classic bootstrap)
    frac_zero = (W == 0).mean()
    assert 0.30 < frac_zero < 0.42, f"frac unselected {frac_zero} not ~1/e"
    print(f"  [1] protein-level reduction OK (row_sum==N, mean~1, frac0={frac_zero:.3f})")

    # (2) one-stage fold: whole clusters kept together — within a resample, all
    #     members of a drawn cluster share the same weight.
    cid_fold = np.repeat(np.arange(8), 5)  # 8 folds x 5 proteins = 40
    rng2 = np.random.default_rng(1)
    W1 = make_cluster_weights(n, cid_fold, idx, n_boot=200, rng=rng2, two_stage=False)
    for b in range(W1.shape[0]):
        for f in range(8):
            members = np.where(cid_fold == f)[0]
            assert len(set(W1[b, members].tolist())) == 1, "one-stage: cluster members must share weight"
    # each row sums to N (8 clusters drawn, each size 5)
    assert (W1.sum(axis=1) == n).all(), "one-stage fold rows must sum to N"
    print("  [2] one-stage fold OK (members share weight; row_sum==N)")

    # (3) two-stage fold with UNEQUAL cluster sizes: total resample size varies
    #     (sum of drawn cluster sizes) AND members within a drawn cluster are
    #     resampled, so weights differ within a cluster (unlike one-stage).
    #     (Equal-size clusters would give a constant row sum, so unequal sizes
    #     are required to exercise this property.)
    cid_uneq = np.repeat(np.arange(7), [2, 3, 5, 6, 7, 8, 9])  # 7 clusters, 40 proteins
    rng3 = np.random.default_rng(2)
    W2 = make_cluster_weights(n, cid_uneq, idx, n_boot=200, rng=rng3, two_stage=True)
    assert W2.sum(axis=1).std() > 0, "two-stage row sums should vary with unequal cluster sizes"
    big = np.where(cid_uneq == 6)[0]  # largest cluster (size 9)
    varied = any(np.unique(W2[b, big]).size > 1 for b in range(W2.shape[0]))
    assert varied, "two-stage should resample within a cluster (member weights differ)"
    print(f"  [3] two-stage fold OK (row_sum std={W2.sum(axis=1).std():.2f}, within-cluster resampling)")

    # (4) design effect: clustered values must give Deff > 1, n_eff < N.
    rng4 = np.random.default_rng(3)
    fold_means = rng4.normal(0, 1, 8)              # strong between-fold signal
    vals = fold_means[cid_fold] + rng4.normal(0, 0.1, n)  # tiny within-fold noise
    de = design_effect(vals, cid_fold)
    assert de["icc"] > 0.5, f"clustered values should have high ICC, got {de['icc']}"
    assert de["design_effect"] > 1.0 and de["n_eff"] < n
    # independent values -> ICC ~ 0, Deff ~ 1
    vals_indep = rng4.normal(0, 1, n)
    de0 = design_effect(vals_indep, cid_fold)
    assert de0["design_effect"] < 1.5, f"independent values Deff should be ~1, got {de0['design_effect']}"
    print(f"  [4] design effect OK (clustered ICC={de['icc']:.2f} Deff={de['design_effect']:.2f} "
          f"n_eff={de['n_eff']:.1f}; indep Deff={de0['design_effect']:.2f})")

    # (5) resample_indices_by_cluster: i.i.d. reduction + cluster size variation
    rng5 = np.random.default_rng(4)
    idx_iid = resample_indices_by_cluster(np.arange(n), rng5)
    assert idx_iid.size == n, "i.i.d. cluster resample must return n indices"
    sizes = [resample_indices_by_cluster(cid_uneq, rng5).size for _ in range(50)]
    assert len(set(sizes)) > 1, "unequal-cluster resample sizes should vary"
    print(f"  [5] resample_indices_by_cluster OK (iid n={idx_iid.size}, fold-resample sizes vary)")

    print("  ALL CLUSTER-BOOTSTRAP TESTS PASSED")


if __name__ == "__main__":
    correctness_test()
