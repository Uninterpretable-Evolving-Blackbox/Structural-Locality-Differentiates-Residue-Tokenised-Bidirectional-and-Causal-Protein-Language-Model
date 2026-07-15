#!/usr/bin/env python3
"""
experiment_lm_faithfulness.py — Delta-LM-loss / CE-increase / KL (behavioral fidelity)
======================================================================================

The standard SAE faithfulness metric (Gao 2024; Lieberum 2024; Villegas &
Ansuini 2025): how much does substituting the SAE reconstruction into the
forward pass degrade the model's *predictions*? Explained variance measures
reconstruction in activation space; this measures it in BEHAVIOR space.

ESM-2 (masked LM): mask 15% of residues, then for the masked positions compute
  - CE_clean : cross-entropy of the true residue under the unmodified model
  - CE_sae   : same, but layer-L output replaced by the SAE reconstruction
  - CE_mean  : same, but layer-L output replaced by the per-protein mean (a
               destroyed-information baseline, for the "loss recovered" denom)
and report  ΔCE = CE_sae − CE_clean,  KL(clean‖sae),  and
  loss_recovered = 1 − (CE_sae − CE_clean) / (CE_mean − CE_clean).

A faithful SAE has ΔCE≈0, KL≈0, loss_recovered≈1. Protein-level bootstrap CIs.

The PLM is loaded fp32 (avoids the MPS fp16-hook bug); norm_scale is applied
around encode/decode.

Usage:
  python experiment_lm_faithfulness.py --layer-dir outputs_layerwise/esm2/layer_16 \
    --layer 16 --max-proteins 60 --save-dir results_faithfulness/esm2_l16
"""

import argparse
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from sae import SparseAutoencoder
from cpu_stage import load_ref_seqs, load_layer
from cluster_bootstrap import load_uid_clusters, resample_indices_by_cluster

warnings.filterwarnings("ignore")


class SAEReconHook:
    def __init__(self, sae, ns):
        self.sae, self.ns = sae, float(ns)

    def __call__(self, module, inp, output):
        hidden = output[0] if isinstance(output, tuple) else output
        rest = output[1:] if isinstance(output, tuple) else None
        B, L, D = hidden.shape
        with torch.no_grad():
            xn = hidden.reshape(-1, D) * self.ns
            xhat, _, _ = self.sae(xn)
            recon = (xhat / self.ns).reshape(B, L, D).to(hidden.dtype)
        return (recon,) + rest if rest is not None else recon


class MeanAblationHook:
    """Replace each token's hidden with the per-protein mean (destroyed-info baseline)."""

    def __call__(self, module, inp, output):
        hidden = output[0] if isinstance(output, tuple) else output
        rest = output[1:] if isinstance(output, tuple) else None
        m = hidden.mean(dim=1, keepdim=True).expand_as(hidden)
        return (m,) + rest if rest is not None else m


def main():
    ap = argparse.ArgumentParser(description="Delta-LM-loss / CE / KL faithfulness")
    ap.add_argument("--layer-dir", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--max-proteins", type=int, default=60)
    ap.add_argument("--mask-frac", type=float, default=0.15)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--cluster-level", choices=["protein", "fold", "superfamily", "family"],
                    default="fold", help="bootstrap unit: protein (orig) or SCOPe cluster (honest CIs)")
    ap.add_argument("--fasta", default="cache/scope_40.fa")
    ap.add_argument("--two-stage", dest="two_stage", action="store_true", default=True)
    ap.add_argument("--one-stage", dest="two_stage", action="store_false")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-dir", required=True)
    args = ap.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    layer_dir = Path(args.layer_dir)
    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)

    meta = json.loads((layer_dir / "META.json").read_text())
    embed_dim, hidden_dim = meta["embed_dim"], meta["sae_hidden_dim"]
    ns = float(meta.get("norm_scale", 1.0))
    sae = SparseAutoencoder(input_dim=embed_dim, expansion=hidden_dim // embed_dim,
                            k_sparse=meta.get("k_sparse", 256), k_aux=meta.get("k_aux", 64))
    sae.load_state_dict(torch.load(layer_dir / "sae_model.pt", map_location="cpu"))
    sae = sae.to(device).float().eval()

    _, uids, lengths = load_layer(layer_dir)
    uids = [str(u) for u in uids]
    ref = load_ref_seqs(layer_dir)
    pairs = list(zip(uids, [ref[u] for u in uids]))[:args.max_proteins]

    from transformers import AutoTokenizer, AutoModelForMaskedLM
    tok = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    model = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t33_650M_UR50D").to(device).float().eval()
    mask_id = tok.mask_token_id
    special = set(getattr(tok, "all_special_ids", []) or [])
    for a in ("pad_token_id", "eos_token_id", "bos_token_id", "cls_token_id", "sep_token_id"):
        t = getattr(tok, a, None)
        if t is not None:
            special.add(int(t))

    layer_mod = model.esm.encoder.layer[args.layer]
    print("=" * 70)
    print(f"  LM FAITHFULNESS — ESM-2 L{args.layer} | {device} | {len(pairs)} proteins")
    print("=" * 70)

    rows = []
    for uid, seq in tqdm(pairs, desc="  proteins"):
        t = tok([seq], return_tensors="pt", add_special_tokens=True).to(device)
        ids = t["input_ids"][0]
        keep = t["attention_mask"][0].bool().clone()
        for sid in special:
            keep &= (ids != sid)
        res_pos = torch.where(keep)[0].cpu().numpy()
        if len(res_pos) < 8:
            continue
        n_mask = max(1, int(args.mask_frac * len(res_pos)))
        masked = np.sort(rng.choice(res_pos, n_mask, replace=False))
        true_ids = ids[masked].clone()
        t["input_ids"][0, masked] = mask_id

        def ce_and_probs():
            with torch.no_grad():
                logits = model(**t).logits[0]
            lp = F.log_softmax(logits[masked], dim=-1)
            ce = -lp[torch.arange(len(masked)), true_ids].mean().item()
            return ce, lp

        ce_clean, lp_clean = ce_and_probs()
        h = layer_mod.register_forward_hook(SAEReconHook(sae, ns))
        ce_sae, lp_sae = ce_and_probs(); h.remove()
        h = layer_mod.register_forward_hook(MeanAblationHook())
        ce_mean, _ = ce_and_probs(); h.remove()

        kl = (lp_clean.exp() * (lp_clean - lp_sae)).sum(-1).mean().item()  # KL(clean||sae)
        rows.append({"uid": uid, "ce_clean": ce_clean, "ce_sae": ce_sae, "ce_mean": ce_mean,
                     "delta_ce": ce_sae - ce_clean, "kl_clean_sae": kl,
                     "n_masked": int(len(masked))})

    df = pd.DataFrame(rows)
    df.to_csv(save_dir / "faithfulness_per_protein.csv", index=False)
    if not len(df):
        print("  No proteins scored."); return

    # Cluster ids per scored protein (fold-level by default) for honest CIs.
    cluster_of = (load_uid_clusters(args.fasta, level=args.cluster_level)
                  if args.cluster_level != "protein" and Path(args.fasta).exists() else None)
    if cluster_of is not None:
        _keys = np.array([cluster_of.get(str(u), f"__singleton__{u}") for u in df["uid"]])
        _, item_clusters = np.unique(_keys, return_inverse=True)
    else:
        item_clusters = np.arange(len(df))

    def boot_ci(x, B, seed=0, clusters=None):
        r = np.random.default_rng(seed)
        ic = clusters if clusters is not None else np.arange(len(x))
        m = [x[resample_indices_by_cluster(ic, r, two_stage=args.two_stage)].mean()
             for _ in range(B)]
        return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]

    dce = df["delta_ce"].to_numpy()
    # loss_recovered = 1 - (CE_sae-CE_clean)/(CE_mean-CE_clean) is only well-defined
    # where mean-ablation is a meaningfully "destroyed" baseline. At deep layers
    # mean-ablation barely raises CE (denominator ~0), making the ratio explode to
    # garbage. Restrict to proteins with a real denominator; ALWAYS report delta_ce
    # (the robust primary fidelity metric) regardless.
    num = (df["ce_sae"] - df["ce_clean"]).to_numpy()
    den = (df["ce_mean"] - df["ce_clean"]).to_numpy()
    DENOM_MIN = 0.1  # nats; below this mean-ablation is not a valid destroyed baseline
    valid = den > DENOM_MIN
    lr_valid = 1.0 - num[valid] / den[valid]
    summary = {
        "layer_dir": str(layer_dir), "layer": args.layer, "n_proteins": int(len(df)),
        "mean_ce_clean": float(df["ce_clean"].mean()),
        "mean_ce_sae": float(df["ce_sae"].mean()),
        "mean_ce_mean_ablation": float(df["ce_mean"].mean()),
        "mean_delta_ce": float(dce.mean()), "ci_delta_ce": boot_ci(dce, args.n_boot, clusters=item_clusters),
        "cluster_level": args.cluster_level,
        "mean_KL_clean_sae": float(df["kl_clean_sae"].mean()),
        "n_loss_recovered_valid": int(valid.sum()),
        "mean_loss_recovered": (float(lr_valid.mean()) if valid.sum() > 0 else None),
        "ci_loss_recovered": (boot_ci(lr_valid, args.n_boot, clusters=item_clusters[valid]) if valid.sum() > 5 else None),
        "note": "delta_ce is the robust primary metric; loss_recovered only over proteins "
                f"with mean-ablation CE increase > {DENOM_MIN} nats (unstable at deep layers).",
    }
    (save_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n  Summary:"); print(json.dumps(summary, indent=2))
    print(f"  Saved to {save_dir}/")


if __name__ == "__main__":
    main()
