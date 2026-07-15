#!/usr/bin/env python3
"""
experiment_steering.py — E3: causal steering of masked-LM predictions
=====================================================================

InterPLM Fig 7 / Villegas Garcia & Ansuini steering test, adapted to structural
features: clamp a single SAE feature at the residue where it fires most strongly
(the anchor) and measure how the ESM-2 masked-LM output distribution shifts at
OTHER positions, bucketed by whether they are 3D structural contacts of the
anchor (Cα < 8 Å, |i-j| >= 12) versus non-contacts.

Causal claim: clamping a high-structural-locality feature perturbs the model's
predictions preferentially at the anchor's spatial contacts, more than a random
feature does. This shows the feature carries structural information that the
model *uses*, not just correlates with.

PLM is loaded in fp32 (avoids the MPS fp16-hook dtype bug); the SAE's norm_scale
is applied around encode/decode.

Usage:
  python experiment_steering.py --layer-dir outputs_layerwise/esm2/layer_16 \
    --pdb-dir cache/pdb_files --top-k 8 --n-control 8 --max-proteins 60 \
    --scales 1,2,4 --save-dir results_steering/esm2_l16

  # smoke
  python experiment_steering.py --layer-dir outputs_layerwise/esm2/layer_16 \
    --save-dir /tmp/steer_smoke --top-k 2 --n-control 2 --max-proteins 10 --scales 2
"""

import argparse
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from tqdm import tqdm
from scipy.spatial import KDTree

from sae import SparseAutoencoder
from cpu_stage import load_ref_seqs, load_layer
from experiment_activation_clamping import extract_ca_coords

warnings.filterwarnings("ignore")


class SteeringHook:
    """Clamp one feature at one residue to target_value; passthrough elsewhere."""

    def __init__(self, sae, norm_scale, feature, anchor_pos, target_value):
        self.sae, self.norm_scale = sae, float(norm_scale)
        self.feature, self.anchor, self.target = feature, anchor_pos, target_value
        self.enabled = target_value is not None

    def __call__(self, module, inp, output):
        hidden = output[0] if isinstance(output, tuple) else output
        rest = output[1:] if isinstance(output, tuple) else None
        B, L, D = hidden.shape
        flat = hidden.reshape(-1, D) * self.norm_scale
        with torch.no_grad():
            z, _ = self.sae.encode(flat)
            if self.enabled:
                z = z.reshape(B, L, -1)
                z[:, self.anchor, self.feature] = self.target
                z = z.reshape(-1, z.shape[-1])
            recon = self.sae.decode(z) / self.norm_scale
        recon = recon.reshape(B, L, D).to(hidden.dtype)
        return (recon,) + rest if rest is not None else recon


def contact_partners(coords, anchor, cutoff=8.0, sep=12):
    valid = ~np.isnan(coords).any(axis=1)
    idx = np.where(valid)[0]
    if anchor not in idx or len(idx) < 2:
        return set()
    tree = KDTree(coords[valid])
    near = tree.query_ball_point(coords[anchor], r=cutoff)
    real = {int(idx[n]) for n in near}
    return {j for j in real if abs(j - anchor) >= sep}


def main():
    ap = argparse.ArgumentParser(description="E3 steering")
    ap.add_argument("--layer-dir", required=True)
    ap.add_argument("--pdb-dir", default="cache/pdb_files")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--n-control", type=int, default=8)
    ap.add_argument("--max-proteins", type=int, default=60)
    ap.add_argument("--scales", default="1,2,4")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-dir", required=True)
    args = ap.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    layer_dir = Path(args.layer_dir)
    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    scales = [float(x) for x in args.scales.split(",")]

    meta = json.loads((layer_dir / "META.json").read_text())
    embed_dim, hidden_dim = meta["embed_dim"], meta["sae_hidden_dim"]
    layer, norm_scale = meta["layer"], meta.get("norm_scale", 1.0)
    k_sparse = meta.get("k_sparse", 256)

    sae = SparseAutoencoder(input_dim=embed_dim, expansion=hidden_dim // embed_dim,
                            k_sparse=k_sparse, k_aux=meta.get("k_aux", 64))
    sae.load_state_dict(torch.load(layer_dir / "sae_model.pt", map_location="cpu"))
    sae = sae.to(device).float().eval()

    Z, uids, lengths = load_layer(layer_dir)
    uids = [str(u) for u in uids]
    ref = load_ref_seqs(layer_dir)
    offsets, off = [], 0
    for L in lengths:
        offsets.append(off); off += int(L)

    ss = pd.read_csv(layer_dir / "struct_seq_metrics.csv").sort_values("struct_delta", ascending=False)
    top_features = ss.head(args.top_k)["feature_idx"].astype(int).tolist()
    rng = np.random.default_rng(args.seed)
    mid = ss.iloc[len(ss) // 4:3 * len(ss) // 4]["feature_idx"].astype(int).to_numpy()
    control_features = rng.choice(mid, size=min(args.n_control, len(mid)), replace=False).tolist()

    from transformers import AutoTokenizer, AutoModelForMaskedLM
    tok = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    model = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t33_650M_UR50D").to(device).float().eval()

    # choose proteins that have a PDB
    chosen = []
    for gi, uid in enumerate(uids):
        if (Path(args.pdb_dir) / f"{uid[1:5].lower()}.pdb").exists():
            chosen.append(gi)
        if len(chosen) >= args.max_proteins:
            break

    print("=" * 70)
    print(f"  E3 STEERING — ESM-2 L{layer} | device {device} | {len(chosen)} proteins")
    print("=" * 70)

    special = set(getattr(tok, "all_special_ids", []) or [])
    for a in ("pad_token_id", "eos_token_id", "bos_token_id", "cls_token_id", "sep_token_id"):
        t = getattr(tok, a, None)
        if t is not None:
            special.add(int(t))

    def base_probs(seq):
        t = tok([seq], return_tensors="pt", add_special_tokens=True).to(device)
        with torch.no_grad():
            logits = model(**t).logits[0]
        return t, F.softmax(logits, dim=-1)

    def keep_index(t):
        ids = t["input_ids"][0]
        keep = t["attention_mask"][0].bool().clone()
        for sid in special:
            keep &= (ids != sid)
        return torch.where(keep)[0]  # token positions of residues, in order

    rows = []
    for kind, feats in [("struct", top_features), ("control", control_features)]:
        for f in tqdm(feats, desc=f"  steering {kind}"):
            for gi in chosen:
                seq = ref[uids[gi]]
                base = offsets[gi]
                acts = np.asarray(Z[base:base + len(seq), f], dtype=np.float32)
                if acts.max() <= 0:
                    continue
                anchor = int(acts.argmax())  # residue-space anchor
                fmax = float(acts.max())
                pdb = Path(args.pdb_dir) / f"{uids[gi][1:5].lower()}.pdb"
                coords = extract_ca_coords(str(pdb), seq)
                if coords is None:
                    continue
                partners = contact_partners(coords, anchor)
                if not partners:
                    continue
                t, p_base = base_probs(seq)
                ridx = keep_index(t)  # maps residue i -> token position ridx[i]
                anchor_tok = int(ridx[anchor])
                for scale in scales:
                    hook = SteeringHook(sae, norm_scale, f, anchor_tok, scale * fmax)
                    h = model.esm.encoder.layer[layer].register_forward_hook(hook)
                    with torch.no_grad():
                        logits = model(**t).logits[0]
                    h.remove()
                    p_steer = F.softmax(logits, dim=-1)
                    dl1 = (p_steer - p_base).abs().sum(-1).detach().cpu().numpy()  # per token
                    # map to residues
                    res_d = dl1[ridx.cpu().numpy()]
                    contact_mask = np.zeros(len(seq), bool)
                    for jp in partners:
                        if jp < len(seq):
                            contact_mask[jp] = True
                    contact_mask[anchor] = False
                    noncontact = ~contact_mask
                    noncontact[anchor] = False
                    rows.append({
                        "kind": kind, "feature_idx": int(f), "uid": uids[gi], "scale": scale,
                        "delta_contact": float(res_d[contact_mask].mean()) if contact_mask.any() else np.nan,
                        "delta_noncontact": float(res_d[noncontact].mean()) if noncontact.any() else np.nan,
                        "n_contacts": int(contact_mask.sum()),
                    })

    df = pd.DataFrame(rows)
    df.to_csv(save_dir / "steering_per_protein.csv", index=False)
    if len(df):
        df["contact_enrichment"] = df["delta_contact"] / df["delta_noncontact"].replace(0, np.nan)
        agg = df.groupby(["kind", "scale"]).agg(
            delta_contact=("delta_contact", "mean"),
            delta_noncontact=("delta_noncontact", "mean"),
            contact_enrichment=("contact_enrichment", "median"),
            n=("uid", "count")).reset_index()
        agg.to_csv(save_dir / "steering_summary.csv", index=False)
        print("\n  Steering summary (Δprob at contacts vs non-contacts):")
        print(agg.to_string(index=False))

        fig, ax = plt.subplots(figsize=(6, 4))
        for kind in ["struct", "control"]:
            sub = agg[agg.kind == kind].sort_values("scale")
            if len(sub):
                ax.plot(sub["scale"], sub["contact_enrichment"], marker="o", label=kind)
        ax.axhline(1.0, color="k", ls="--", lw=1)
        ax.set_xlabel("clamp scale (x feature max)")
        ax.set_ylabel("contact / non-contact Δprob (median)")
        ax.set_title(f"Steering contact-enrichment (ESM-2 L{layer})")
        ax.legend(); fig.tight_layout()
        fig.savefig(save_dir / "steering_enrichment.png", dpi=200); plt.close(fig)
    print(f"\n  Saved to {save_dir}/")


if __name__ == "__main__":
    main()
