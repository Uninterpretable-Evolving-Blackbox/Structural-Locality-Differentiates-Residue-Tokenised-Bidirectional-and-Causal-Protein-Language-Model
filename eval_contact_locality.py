#!/usr/bin/env python3
"""
eval_contact_locality.py — connect L_struct to downstream contact prediction (Reviewer 6P8E).

For each layer of ESM-2 and RITA:
  - unsupervised APC contact prediction (Rao et al. 2020) from raw per-residue embeddings
  - top-L/5 long-range precision, with contacts = Cα < 8 Å AND |i-j| >= 12
    (IDENTICAL to L_struct's definition — reuses compute_true_contacts / top_l_precision
     so the contact metric matches L_struct exactly, not the generic |i-j|>=6 convention)
  - L_struct = mean struct_delta (from the layer's struct_seq_metrics.csv)
Then correlate contact-P@L/5 vs L_struct across layers (per model + pooled) and check the
matched-depth pattern (does the higher-L_struct model also predict contacts better?).

Grounding (all reused from experiment_activation_clamping, not reinvented):
  cutoff 8 Å + |i-j|>=12 (Marks 2011 / Morcos 2011; == L_struct); top-L/5 (Rao 2021);
  APC-corrected embedding contacts (Rao 2020).

Pitfall guard: sequences pre-filtered to <= MAX_RES so the extractor never truncates
(truncation would silently misalign per-protein offsets).
"""
import argparse, json, glob, os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from cpu_stage import load_ref_seqs
from experiment_activation_clamping import (
    extract_ca_coords, compute_true_contacts,
    predict_contacts_from_embeddings, top_l_precision,
)
import extract_embeddings as ee

MAX_RES = 1000   # < extractor max_length(1024) - specials, so NO truncation -> offsets align


def _extract(model_key, seqs, layers):
    """Per-model extraction (interfaces differ: ESM-2/RITA take model_name, ProtT5 doesn't)."""
    s = list(seqs)
    if model_key == "esm2":
        return ee.extract_esm2_embeddings(s, layers=layers, model_name="facebook/esm2_t33_650M_UR50D")
    if model_key == "rita":
        return ee.extract_rita_embeddings(s, layers=layers, model_name="lightonai/RITA_l")
    if model_key == "prott5_enc":
        return ee.extract_prott5_encoder_embeddings(s, layers=layers)
    if model_key == "prott5_dec":
        return ee.extract_prott5_decoder_embeddings(s, layers=layers)
    raise ValueError(model_key)


def layers_of(model):
    ds = glob.glob(f"outputs_layerwise/{model}/layer_*")
    return sorted(int(p.split("layer_")[1]) for p in ds)


def lstruct_3seed(model, layer):
    """Mean struct_delta averaged over SAE seeds 42/43/44 (+ cross-seed SD)."""
    vals = []
    for suf in ("", "_seed43", "_seed44"):
        p = f"outputs_layerwise{suf}/{model}/layer_{layer}/struct_seq_metrics.csv"
        if os.path.exists(p):
            vals.append(float(pd.read_csv(p)["struct_delta"].mean()))
    sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return float(np.mean(vals)), sd, len(vals)


def contact_pl5(model_key, layers, uids, seqs, pdb_dir):
    print(f"  [{model_key}] extracting {len(seqs)} seqs at layers {layers} ...", flush=True)
    emb = _extract(model_key, seqs, layers)   # {layer: (sum_res, D)} in input order
    lengths = np.array([len(s) for s in seqs])
    offs = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    # sanity: total rows must equal sum(len) (no truncation) for at least one layer
    tot = int(emb[layers[0]].shape[0])
    if tot != int(lengths.sum()):
        raise RuntimeError(f"row mismatch {tot} != {lengths.sum()} — truncation/alignment bug")

    # per-protein true contacts (once), skip missing/mismatched PDBs
    true, keep = {}, []
    for gi, uid in enumerate(uids):
        pdb = Path(pdb_dir) / f"{str(uid)[1:5].lower()}.pdb"
        if not pdb.exists():
            continue
        coords = extract_ca_coords(str(pdb), seqs[gi])
        if coords is None or coords.shape[0] != lengths[gi]:
            continue
        tc = compute_true_contacts(coords)   # 8 Å, |i-j|>=12
        if tc.sum() == 0:
            continue
        true[gi] = tc; keep.append(gi)
    print(f"  [{model_key}] {len(keep)} proteins with valid coords+contacts", flush=True)

    out = {}   # layer -> per-protein precision array (aligned to `keep`)
    for L in layers:
        arr = np.asarray(emb[L])
        precs = [top_l_precision(predict_contacts_from_embeddings(arr[offs[gi]:offs[gi] + lengths[gi]]),
                                 true[gi], seq_gap=12, fraction=0.2) for gi in keep]
        out[L] = np.array(precs, dtype=np.float64)
        print(f"    layer {L}: contact P@L/5 = {out[L].mean():.4f} (n={len(precs)})", flush=True)
    return out, len(keep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-layer-dir", default="outputs_layerwise/esm2/layer_16")
    ap.add_argument("--pdb-dir", default="cache/pdb_files")
    ap.add_argument("--max-proteins", type=int, default=500)
    ap.add_argument("--models", default="esm2,rita")
    ap.add_argument("--out", default="results_contact_locality/summary.json")
    args = ap.parse_args()

    ref = Path(args.ref_layer_dir)
    uids = [str(u) for u in json.loads((ref / "uids.json").read_text())]
    smap = load_ref_seqs(ref)
    # pre-filter: has sequence, 24 <= len <= MAX_RES  (>=24 so L/5 top-k >= a few pairs)
    uids = [u for u in uids if u in smap and 24 <= len(smap[u]) <= MAX_RES]
    uids = uids[:args.max_proteins]
    seqs = [smap[u] for u in uids]
    print(f"eval proteins: {len(uids)} (len<= {MAX_RES}, no truncation)")

    B = 1000
    rows, res, ppdir = [], {}, Path("results_contact_locality")
    ppdir.mkdir(parents=True, exist_ok=True)
    for m in args.models.split(","):
        Ls = layers_of(m)
        cpl, nkeep = contact_pl5(m, Ls, uids, seqs, args.pdb_dir)
        np.savez(ppdir / f"{m}_perprotein.npz", **{str(L): cpl[L] for L in Ls})
        # ONE set of protein resamples, shared across layers (so the correlation bootstrap is coherent)
        rng = np.random.default_rng(0)
        boot = [rng.integers(0, nkeep, nkeep) for _ in range(B)]
        ls = np.array([lstruct_3seed(m, L)[0] for L in Ls])
        for i, L in enumerate(Ls):
            pp = cpl[L]
            bmeans = np.array([pp[bi].mean() for bi in boot])
            lsm, lssd, lsn = lstruct_3seed(m, L)
            rows.append(dict(model=m, layer=L, n_proteins=nkeep,
                             lstruct=lsm, lstruct_sd=lssd, lstruct_nseed=lsn,
                             contact_pl5=float(pp.mean()),
                             contact_lo=float(np.percentile(bmeans, 2.5)),
                             contact_hi=float(np.percentile(bmeans, 97.5))))
        # correlation + protein-bootstrap CI (resample proteins -> per-layer means -> Spearman)
        rhos = np.array([spearmanr(ls, [cpl[L][bi].mean() for L in Ls])[0] for bi in boot])
        rho0, p0 = spearmanr(ls, [cpl[L].mean() for L in Ls])
        res[m] = dict(spearman=round(float(rho0), 3), p=round(float(p0), 4), n_layers=len(Ls),
                      boot_ci=[round(float(np.nanpercentile(rhos, 2.5)), 3),
                               round(float(np.nanpercentile(rhos, 97.5)), 3)])

    df = pd.DataFrame(rows)
    outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(outp.with_suffix(".csv"), index=False)
    print("\n" + df.to_string(index=False))
    print("\ncorrelations (3-seed L_struct vs contact P@L/5; boot_ci = protein bootstrap):")
    print(json.dumps(res, indent=2))
    outp.write_text(json.dumps({
        "grounding": "8A/|i-j|>=12/top-L5/APC (== L_struct); L_struct=mean over seeds 42/43/44; "
                     "protein bootstrap B=1000 for contact CIs and correlation CI",
        "correlations": res, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
