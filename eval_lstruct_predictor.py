#!/usr/bin/env python3
"""
eval_lstruct_predictor.py — convergent-validity test: does L_struct SELECT features that
predict long-range contacts on HELD-OUT proteins?

Design (see PLAN_LSTRUCT_AS_PREDICTOR.md):
  score(i,j) over a feature subset S  ==  Z[:, S] @ Z[:, S].T
  -> so we reuse predict_contacts_from_embeddings() UNCHANGED, passing Z[:, S]
     instead of the raw embeddings. Same readout, same metric, same contact
     definition as §4.4 -> the ONLY thing that varies is the matrix fed in.

Anchor: Rao et al. 2021 (unsupervised contact readout, top-L/5 long-range precision).
Deviations logged in PLAN_LSTRUCT_AS_PREDICTOR.md.

CIRCULARITY GUARD (the whole point):
  * L_struct for feature selection is recomputed on the 1,350 TRAIN proteins ONLY
    (cached as struct_seq_metrics_train.csv). The existing struct_seq_metrics.csv is
    computed on all 1,500 and would leak the evaluation proteins into selection.
  * Contact precision is evaluated ONLY on the 150 held-out val proteins.

CONFOUND GUARD:
  TopK SAEs fire k=256 of 10,240 features per residue. A uniform random K-subset leaves
  most residues all-zero, whereas high-L_struct features may fire more often -> "top-K
  beats random-K" could be a firing-rate artifact. So we include a FIRING-RATE-MATCHED
  random control (stratified by firing-rate quantile), which is the baseline the claim
  must beat. Uniform random is reported too, but it is NOT the control.

The claim rests on CONTRASTS, never the absolute number:
  top-K vs matched-random, top-K vs bottom-K, SAE vs raw embeddings.

Usage:
  python eval_lstruct_predictor.py --layer-dir outputs_layerwise/esm2/layer_16 \
      --out results_lstruct_predictor/esm2_l16_seed42
"""
import argparse, json, os, time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, cpu_count

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from cpu_stage import (
    load_ref_seqs,
    adj_list_to_sparse,
    build_protein_permutations,
    build_neighbor_graphs_residue_parallel,
    _process_struct_seq_chunk_v3,
    DEFAULT_CONTACT_CUTOFF,
    DEFAULT_SEQ_GAP_MIN,
    DEFAULT_TOPK_FRAC,
)
from experiment_activation_clamping import (
    extract_ca_coords,
    compute_true_contacts,
    predict_contacts_from_embeddings,
    top_l_precision,
)

ROOT = Path(__file__).resolve().parent
PDB_DIR = ROOT / "cache" / "pdb_files"


# ---------------------------------------------------------------- train-only L_struct

def compute_train_struct_delta(layer_dir: Path, n_shuffles: int, n_jobs: int) -> pd.DataFrame:
    """Recompute per-feature struct/seq delta on the TRAIN proteins only. Cached."""
    cache = layer_dir / "struct_seq_metrics_train.csv"
    if cache.exists():
        print(f"    [cache hit] {cache}", flush=True)
        return pd.read_csv(cache)

    meta = json.loads((layer_dir / "META.json").read_text())
    uids = json.loads((layer_dir / "uids.json").read_text())
    lengths_tok = np.load(layer_dir / "lengths.npy")
    seqs = load_ref_seqs(layer_dir)
    val_set = set(meta["val_uids"])

    train_uids = [u for u in uids if u not in val_set]
    print(f"    train proteins: {len(train_uids)} (of {len(uids)}; {len(val_set)} held out)",
          flush=True)

    uid_to_idx = {u: i for i, u in enumerate(uids)}
    tr_idx = [uid_to_idx[u] for u in train_uids]
    tok_off = np.concatenate([[0], np.cumsum(lengths_tok)]).astype(np.int64)
    rows = np.concatenate([np.arange(tok_off[i], tok_off[i + 1], dtype=np.int64) for i in tr_idx])

    Z = np.load(layer_dir / "Z.npy", mmap_mode="r")
    Z_tr = np.asarray(Z[rows], dtype=np.float16)
    tr_res_len = np.array([len(seqs[u]) for u in train_uids], dtype=np.int64)
    n_res = int(tr_res_len.sum())
    if Z_tr.shape[0] != n_res:
        raise RuntimeError(f"train tokens {Z_tr.shape[0]} != residues {n_res} (not 1:1)")
    print(f"    train residues: {n_res:,}", flush=True)

    seq_adj, struct_adj = build_neighbor_graphs_residue_parallel(
        train_uids, tr_res_len, {u: seqs[u] for u in train_uids}, PDB_DIR,
        n_jobs=n_jobs, contact_cutoff=DEFAULT_CONTACT_CUTOFF, seq_gap_min=DEFAULT_SEQ_GAP_MIN)
    A_seq, deg_seq = adj_list_to_sparse(seq_adj, n_res)
    A_struct, deg_struct = adj_list_to_sparse(struct_adj, n_res)
    del seq_adj, struct_adj
    print(f"    seq edges {A_seq.nnz:,}  struct edges {A_struct.nnz:,}", flush=True)

    perm = build_protein_permutations(tr_res_len, n_shuffles)

    n_features = int(Z_tr.shape[1])
    chunk = 256
    n_chunks = (n_features + chunk - 1) // chunk
    mem_per_worker = n_res * chunk * 4 * 5 / 1e9
    budget = float(os.environ.get("CPU_STAGE_MEM_GB", 100.0))
    eff = min(n_jobs if n_jobs > 0 else cpu_count(), max(1, int(budget / max(mem_per_worker, 0.1))))
    print(f"    {n_features} features / {n_chunks} chunks / {eff} workers", flush=True)

    t0 = time.time()
    res = Parallel(n_jobs=eff, verbose=1)(
        delayed(_process_struct_seq_chunk_v3)(
            ci, chunk, Z_tr, None, A_seq, deg_seq, A_struct, deg_struct,
            perm, n_features, DEFAULT_TOPK_FRAC)
        for ci in range(n_chunks))

    idx = np.concatenate([r[0] for r in res])
    so, to_, ss, ts = (np.concatenate([r[i] for r in res]) for i in (1, 2, 3, 4))
    o = np.argsort(idx)
    df = pd.DataFrame({
        "feature_idx": idx[o].astype(np.int32),
        "seq_delta": (so - ss)[o],
        "struct_delta": (to_ - ts)[o],
    })
    df.to_csv(cache, index=False)
    print(f"    saved -> {cache}  ({time.time()-t0:.0f}s)", flush=True)
    return df


# ---------------------------------------------------------------- feature subsets

def firing_rates(Z_tr_rows: np.ndarray) -> np.ndarray:
    """Fraction of residues where each feature is active (>0), on TRAIN rows."""
    n = Z_tr_rows.shape[0]
    out = np.zeros(Z_tr_rows.shape[1], dtype=np.float64)
    step = 20000
    for s in range(0, n, step):
        out += (np.asarray(Z_tr_rows[s:s + step], dtype=np.float32) > 0).sum(axis=0)
    return out / n


def matched_random(target: np.ndarray, fr: np.ndarray, rng, n_bins=20) -> np.ndarray:
    """Random features stratified to match `target`'s firing-rate distribution."""
    edges = np.quantile(fr[fr > 0], np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    bin_of = np.digitize(fr, edges) - 1
    banned = set(target.tolist())
    picked = []
    for b in np.unique(bin_of[target]):
        need = int((bin_of[target] == b).sum())
        pool = np.array([j for j in np.where(bin_of == b)[0] if j not in banned and j not in picked])
        if len(pool) == 0:
            continue
        picked.extend(rng.choice(pool, size=min(need, len(pool)), replace=False).tolist())
    return np.array(sorted(picked), dtype=np.int64)


# ---------------------------------------------------------------- contact eval

def eval_subsets(layer_dir: Path, ranked: np.ndarray, fr: np.ndarray,
                 Ks, n_random: int, seed: int):
    meta = json.loads((layer_dir / "META.json").read_text())
    uids = json.loads((layer_dir / "uids.json").read_text())
    lengths_tok = np.load(layer_dir / "lengths.npy")
    seqs = load_ref_seqs(layer_dir)
    val_set = set(meta["val_uids"])
    val_uids = [u for u in uids if u in val_set]

    uid_to_idx = {u: i for i, u in enumerate(uids)}
    tok_off = np.concatenate([[0], np.cumsum(lengths_tok)]).astype(np.int64)

    Z = np.load(layer_dir / "Z.npy", mmap_mode="r")
    RAW = np.load(layer_dir / "raw_embeddings.npy", mmap_mode="r")

    # ---- true contacts on val proteins (once)
    keep, true, slices = [], {}, {}
    for u in val_uids:
        gi = uid_to_idx[u]
        pdb = PDB_DIR / f"{str(u)[1:5].lower()}.pdb"
        if not pdb.exists():
            continue
        coords = extract_ca_coords(str(pdb), seqs[u])
        L = int(tok_off[gi + 1] - tok_off[gi])
        if coords is None or coords.shape[0] != L:
            continue
        tc = compute_true_contacts(coords, cutoff=DEFAULT_CONTACT_CUTOFF,
                                   seq_gap=DEFAULT_SEQ_GAP_MIN)
        if tc.sum() == 0:
            continue
        keep.append(u); true[u] = tc; slices[u] = (int(tok_off[gi]), int(tok_off[gi + 1]))
    print(f"    val proteins with valid coords+contacts: {len(keep)}", flush=True)

    rng = np.random.default_rng(seed)
    conds = {}
    for K in Ks:
        conds[f"top{K}"] = ranked[:K]
        conds[f"bottom{K}"] = ranked[-K:]
        for r in range(n_random):
            conds[f"rand{K}_u{r}"] = rng.choice(len(ranked), size=K, replace=False)
            conds[f"rand{K}_m{r}"] = matched_random(ranked[:K], fr, rng)
    conds["all"] = np.arange(len(ranked))

    per_prot = {c: [] for c in conds}
    per_prot["raw_emb"] = []
    for n, u in enumerate(keep):
        a, b = slices[u]
        Zp = np.asarray(Z[a:b], dtype=np.float32)
        tc = true[u]
        for c, S in conds.items():
            per_prot[c].append(top_l_precision(
                predict_contacts_from_embeddings(Zp[:, S]), tc, seq_gap=12, fraction=0.2))
        per_prot["raw_emb"].append(top_l_precision(
            predict_contacts_from_embeddings(np.asarray(RAW[a:b], dtype=np.float32)),
            tc, seq_gap=12, fraction=0.2))
        if (n + 1) % 25 == 0:
            print(f"      {n+1}/{len(keep)} proteins", flush=True)
    return {c: np.array(v) for c, v in per_prot.items()}, keep


def collapse_random(per_prot, Ks, n_random):
    """Average the random draws into a single per-protein vector per (K, kind)."""
    out = {}
    for c, v in per_prot.items():
        if not c.startswith("rand"):
            out[c] = v
    for K in Ks:
        for kind, tag in (("u", "rand_uniform"), ("m", "rand_matched")):
            arrs = [per_prot[f"rand{K}_{kind}{r}"] for r in range(n_random)]
            out[f"{tag}{K}"] = np.mean(np.stack(arrs), axis=0)
    return out


def bootstrap(per_prot, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(next(iter(per_prot.values())))
    idx = rng.integers(0, n, size=(B, n))
    res = {}
    for c, v in per_prot.items():
        bs = v[idx].mean(axis=1)
        res[c] = dict(mean=float(v.mean()),
                      boot_mean=float(bs.mean()),
                      ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
    return res, idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ks", default="64,256,1024")
    ap.add_argument("--n-random", type=int, default=10)
    ap.add_argument("--n-shuffles", type=int, default=5)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    ld = Path(a.layer_dir)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    Ks = [int(x) for x in a.ks.split(",")]
    print(f"\n=== {ld} ===", flush=True)

    print("  [1/3] train-only L_struct", flush=True)
    df = compute_train_struct_delta(ld, a.n_shuffles, a.n_jobs)
    sd = df.sort_values("feature_idx")["struct_delta"].to_numpy()
    ranked = np.argsort(sd)[::-1]          # high L_struct first

    print("  [2/3] firing rates (train rows)", flush=True)
    meta = json.loads((ld / "META.json").read_text())
    uids = json.loads((ld / "uids.json").read_text())
    lengths_tok = np.load(ld / "lengths.npy")
    val_set = set(meta["val_uids"])
    uid_to_idx = {u: i for i, u in enumerate(uids)}
    tok_off = np.concatenate([[0], np.cumsum(lengths_tok)]).astype(np.int64)
    tr_rows = np.concatenate([np.arange(tok_off[uid_to_idx[u]], tok_off[uid_to_idx[u] + 1])
                              for u in uids if u not in val_set])
    Z = np.load(ld / "Z.npy", mmap_mode="r")
    fr = firing_rates(Z[tr_rows])

    print("  [3/3] contact eval on held-out val", flush=True)
    per_prot, keep = eval_subsets(ld, ranked, fr, Ks, a.n_random, a.seed)
    per_prot = collapse_random(per_prot, Ks, a.n_random)
    stats, _ = bootstrap(per_prot, B=a.boot, seed=a.seed)

    summary = dict(
        layer_dir=str(ld), n_val_proteins=len(keep), ks=Ks,
        n_random_draws=a.n_random, boot=a.boot,
        mean_firing_rate_top=dict((f"top{K}", float(fr[ranked[:K]].mean())) for K in Ks),
        mean_firing_rate_all=float(fr.mean()),
        results=stats,
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    pd.DataFrame(per_prot, index=keep).to_csv(out / "per_protein.csv")

    print("\n  --- top-L/5 on held-out val ---", flush=True)
    for c in sorted(stats, key=lambda c: -stats[c]["mean"]):
        s = stats[c]
        print(f"    {c:20s} {s['mean']:.4f}  [{s['ci'][0]:.4f},{s['ci'][1]:.4f}]", flush=True)
    print(f"\n  saved -> {out}/summary.json", flush=True)


if __name__ == "__main__":
    main()
