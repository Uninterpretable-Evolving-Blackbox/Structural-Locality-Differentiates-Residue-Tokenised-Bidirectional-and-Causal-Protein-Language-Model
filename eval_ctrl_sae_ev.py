#!/usr/bin/env python3
"""
eval_ctrl_sae_ev.py — the SAE-quality check that was never run on the controlled pilot.

WHY: ProGen2 was DROPPED from this project because its SAE basis was degenerate
(val_EV 0.99 -> the SAE is ~an identity map, so "features" are not a meaningful
decomposition). eval_ctrl_plm.py never logged val_EV, so outputs_ctrl/ has no EV
recorded anywhere. If ctrl_clm_A's basis is degenerate, Appendix O's late-depth CLM
deficit could be an SAE artefact rather than a property of the objective.

WHAT: re-extract the PLM activations (deterministic; no training), apply the SAVED
norm_scale from META.json, load the SAVED sae_model.pt, and compute train/val
explained variance with the project's canonical compute_explained_variance()
(train_sae.py) so the numbers are comparable to Appendix D.

Reuses eval_ctrl_plm.extract_layer so the activations are byte-identical to the ones
the SAE was trained on.

Usage:
  python eval_ctrl_sae_ev.py --out results_ctrl_sae_ev/summary.json
"""
import argparse, json, os
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from model_ctrl_plm import PLM, PLMConfig
from cpu_stage import load_ref_seqs
from sae import SparseAutoencoder
from train_sae import compute_explained_variance
from eval_ctrl_plm import extract_layer

ROOT = Path(__file__).resolve().parent


def ev_for(layer_dir: Path, device: str, sae_device: str):
    meta = json.loads((layer_dir / "META.json").read_text())
    uids = json.loads((layer_dir / "uids.json").read_text())
    seqs_by_uid = load_ref_seqs(layer_dir)
    seqs = [seqs_by_uid[u] for u in uids]
    val_set = set(meta["val_uids"])

    # load exactly as eval_ctrl_plm.py does — `causal` comes from the ckpt's own cfg,
    # never inferred from the dir name
    ck = torch.load(meta["ckpt"], map_location="cpu")
    cfg = PLMConfig(**ck["cfg"])
    model = PLM(cfg)
    model.load_state_dict(ck["model"])
    model = model.to(device).eval()
    assert cfg.causal == ("clm" in meta["model"]), \
        f"ckpt causal={cfg.causal} disagrees with dir {meta['model']}"

    X, lengths = extract_layer(model, uids, seqs, meta["layer"], device)
    offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
    is_val = np.array([u in val_set for u in uids], dtype=bool)
    val_rows = np.concatenate([np.arange(offsets[i], offsets[i] + lengths[i])
                               for i in range(len(uids)) if is_val[i]])
    tr_rows = np.concatenate([np.arange(offsets[i], offsets[i] + lengths[i])
                              for i in range(len(uids)) if not is_val[i]])

    ns = float(meta["norm_scale"])
    D = X.shape[1]
    # constructed exactly as regen_z_from_sae.py does (expansion, not hidden_dim)
    sae = SparseAutoencoder(
        input_dim=D,
        expansion=int(meta["sae_hidden_dim"]) // D,
        k_sparse=int(meta["k_sparse"]),
        k_aux=int(meta.get("k_aux", 64)),
        dead_threshold=int(meta.get("dead_threshold", 1_000_000)),
    )
    sae.load_state_dict(torch.load(layer_dir / "sae_model.pt", map_location="cpu"))
    sae = sae.to(sae_device).float().eval()

    ev_val = compute_explained_variance(sae, (X[val_rows] * ns).astype(np.float32), device=sae_device)
    ev_tr = compute_explained_variance(sae, (X[tr_rows] * ns).astype(np.float32), device=sae_device)
    return dict(model=meta["model"], layer=int(meta["layer"]),
                train_EV=round(float(ev_tr), 4), val_EV=round(float(ev_val), 4),
                gap=round(float(ev_tr - ev_val), 4),
                n_val_rows=int(len(val_rows)), n_train_rows=int(len(tr_rows)),
                embed_dim=D, sae_hidden_dim=int(meta["sae_hidden_dim"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", default="outputs_ctrl/ctrl_mlm_A,outputs_ctrl/ctrl_clm_A")
    ap.add_argument("--layers", default="0,3,6,9,11")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--sae-device", default="cpu")
    ap.add_argument("--out", default="results_ctrl_sae_ev/summary.json")
    a = ap.parse_args()

    rows = []
    for root in a.roots.split(","):
        for L in [int(x) for x in a.layers.split(",")]:
            ld = ROOT / root / f"layer_{L}"
            if not (ld / "sae_model.pt").exists():
                print(f"[skip] {ld} (no sae_model.pt)"); continue
            print(f"\n=== {ld} ===", flush=True)
            r = ev_for(ld, a.device, a.sae_device)
            rows.append(r)
            print(f"  train_EV {r['train_EV']:.4f}  val_EV {r['val_EV']:.4f}  gap {r['gap']:.4f}",
                  flush=True)

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(
        note="ProGen2 was dropped at val_EV 0.99 (degenerate basis). Reference: "
             "ESM-2 L16 val_EV 0.717, RITA L18 val_EV 0.981 (results_sae_diagnostics).",
        rows=rows), indent=1))
    print(f"\nsaved -> {out}")
    print(f"\n{'model':14} {'layer':>5} {'train_EV':>9} {'val_EV':>8} {'gap':>7}")
    for r in rows:
        print(f"{r['model']:14} {r['layer']:>5} {r['train_EV']:>9.4f} {r['val_EV']:>8.4f} {r['gap']:>7.4f}")


if __name__ == "__main__":
    main()
