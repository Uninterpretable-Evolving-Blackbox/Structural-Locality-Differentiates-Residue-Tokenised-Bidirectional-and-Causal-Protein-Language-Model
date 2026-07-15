#!/usr/bin/env python3
"""
eval_ctrl_causal.py — computational-axis lens for the controlled PLMs.

Gradient attribution (Anthropic-style grad x act) of each model's OWN LM loss to its
SAE features at a chosen block. One backward pass -> per-feature causal attribution.
Then: are STRUCTURAL features (high struct_delta) more causally load-bearing, and does
the MLM>early / CLM>late crossover seen in L_struct also appear on the causal axis?

Insert SAE reconstruction at block L (features z tracked), run to output, backprop to z.
attribution_j = mean over active tokens of |z_j * dL/dz_j|.

MLM loss = masked-residue CE (15% mask); CLM loss = next-residue CE. Each model attributed
to its NATIVE objective; comparison is within-model (struct vs random), so cross-model loss
scale doesn't need to match.
"""
import argparse, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from model_ctrl_plm import PLM, PLMConfig, build_rope
from sae import SparseAutoencoder
from cpu_stage import load_ref_seqs
import pandas as pd

AA2ID = {a: 5 + i for i, a in enumerate("ACDEFGHIKLMNPQRSTVWY")}
BOS, EOS, MASK, PAD = 1, 2, 3, 0


def tok(seq, maxlen=512):
    return ([BOS] + [AA2ID.get(a, 4) for a in seq.upper()] + [EOS])[:maxlen]


def load_sae(layer_dir, device):
    meta = json.loads((Path(layer_dir) / "META.json").read_text())
    D, H = meta["embed_dim"], meta["sae_hidden_dim"]
    sae = SparseAutoencoder(input_dim=D, expansion=H // D, k_sparse=meta["k_sparse"], k_aux=64)
    sae.load_state_dict(torch.load(Path(layer_dir) / "sae_model.pt", map_location="cpu"))
    return sae.to(device).float().eval(), float(meta["norm_scale"])


def run_block_range(model, x, lo, hi, cos, sin, mask):
    for b in model.blocks[lo:hi]:
        x = b(x, cos, sin, mask)
    return x


def attribute(model, sae, ns, ids, am, layer, causal, mask_id, rng):
    dev = ids.device
    B, T = ids.shape
    # build MLM/CLM labels + input
    if not causal:  # MLM: mask 15%
        prob = torch.full(ids.shape, 0.15, device=dev)
        prob[(am == 0) | (ids == BOS) | (ids == EOS)] = 0
        sel = torch.bernoulli(prob).bool()
        if sel.sum() == 0:
            sel[am.bool()][0] = True
        labels = torch.where(sel, ids, torch.full_like(ids, -100))
        inp = ids.clone(); inp[sel] = mask_id
    else:            # CLM: next-token
        inp = ids
        labels = ids.clone(); labels[:, :-1] = ids[:, 1:]; labels[:, -1] = -100
        labels[am == 0] = -100
        np_ = torch.zeros_like(am); np_[:, :-1] = am[:, 1:]; labels[np_ == 0] = -100
    # forward to block `layer` (no grad), then insert SAE with z tracked
    x = model.tok(inp)
    cos, sin = build_rope(T, model.cfg.d_model // model.cfg.n_heads, model.cfg.rope_theta, dev, x.dtype)
    m = model.build_mask(am).to(x.dtype)
    with torch.no_grad():
        h = run_block_range(model, x, 0, layer + 1, cos, sin, m)   # residual after block `layer`
        z0, _ = sae.encode(h.reshape(-1, h.shape[-1]) * ns)
    z = z0.detach().requires_grad_(True)
    recon = (sae.decode(z) / ns).reshape(B, T, -1).to(x.dtype)
    out = run_block_range(model, recon, layer + 1, len(model.blocks), cos, sin, m)
    logits = model.head(model.norm(out))
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), labels.reshape(-1), ignore_index=-100)
    loss.backward()
    attr = (z * z.grad).abs()                     # (B*T, hidden)
    active = (z > 0).float()
    # per-feature: mean |grad*act| over ACTIVE tokens
    num = (attr * active).sum(0)
    den = active.sum(0).clamp_min(1)
    return (num / den).detach().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--layer-dir", required=True, help="outputs_ctrl/<name>/layer_<L> (has SAE + struct csv)")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--causal", action="store_true", help="CLM (else MLM)")
    ap.add_argument("--ref-layer-dir", default="outputs_layerwise/esm2/layer_16")
    ap.add_argument("--max-proteins", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    dev = args.device
    ck = torch.load(args.ckpt, map_location="cpu")
    model = PLM(PLMConfig(**ck["cfg"])); model.load_state_dict(ck["model"]); model = model.to(dev).float().eval()
    sae, ns = load_sae(args.layer_dir, dev)
    ref = Path(args.ref_layer_dir)
    uids = [str(u) for u in json.loads((ref / "uids.json").read_text())]
    seqs = load_ref_seqs(ref)
    uids = [u for u in uids if u in seqs][:args.max_proteins]
    rng = np.random.default_rng(0)

    attrs = []
    for i in range(0, len(uids), args.batch_size):
        chunk = uids[i:i + args.batch_size]
        toks = [tok(seqs[u]) for u in chunk]
        T = max(len(t) for t in toks)
        ids = np.full((len(toks), T), PAD, np.int64); am = np.zeros((len(toks), T), np.int64)
        for r, t in enumerate(toks):
            ids[r, :len(t)] = t; am[r, :len(t)] = 1
        a = attribute(model, sae, ns, torch.from_numpy(ids).to(dev), torch.from_numpy(am).to(dev),
                      args.layer, args.causal, MASK, rng)
        attrs.append(a)
    attr = np.mean(attrs, axis=0)                 # per-feature mean attribution

    ss = pd.read_csv(Path(args.layer_dir) / "struct_seq_metrics.csv").sort_values("feature_idx")
    sd = ss["struct_delta"].to_numpy()
    n = min(len(sd), len(attr)); sd, attr = sd[:n], attr[:n]
    rho, _ = spearmanr(sd, attr)
    top = np.argsort(sd)[-max(1, n // 20):]       # top-5% structural features
    randf = rng.choice(n, len(top), replace=False)
    print(json.dumps({
        "layer": args.layer, "causal": args.causal,
        "spearman_struct_vs_attribution": round(float(rho), 4),
        "mean_attr_top5pct_struct": float(attr[top].mean()),
        "mean_attr_random": float(attr[randf].mean()),
        "ratio_topstruct_to_random": round(float(attr[top].mean() / (attr[randf].mean() + 1e-12)), 3),
    }, indent=2))


if __name__ == "__main__":
    main()
