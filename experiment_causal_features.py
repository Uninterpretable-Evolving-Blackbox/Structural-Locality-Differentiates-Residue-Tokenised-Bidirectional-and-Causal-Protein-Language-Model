#!/usr/bin/env python3
"""
experiment_causal_features.py — E2: per-feature causal effect + causal H1
=========================================================================

Reviewer 6P8E's top request (causal, not just correlational) and the user's
priority (analyse the causally load-bearing features).

For each candidate SAE feature we ABLATE it singly during an ESM-2 forward pass
at the SAE's layer and measure the change in long-range (|i-j|>=12) top-L/5
contact precision (APC-corrected embedding contacts; same readout as the
existing clamping experiment). A feature is "causal-structural" if ablating it
significantly *reduces* contact precision relative to a matched set of random
control features.

We then test causal-H1: do ESM-2's top structural-locality features carry a
larger causal effect than random features? (and, run for RITA too if a hook is
available, whether ESM-2's causal-structural feature population is larger.)

Key correctness points vs the original clamping script:
  - the SAE was trained on Bricken-normalised inputs, so the hook scales hidden
    states by norm_scale before encode and unscales after decode;
  - the PLM is loaded in fp32 so the fp16-model / fp32-SAE MPS dtype mismatch
    does not occur (lets us use MPS instead of CPU).

Usage:
  python experiment_causal_features.py --layer-dir outputs_layerwise/esm2/layer_16 \
    --pdb-dir cache/pdb_files --top-k 15 --n-control 15 --max-proteins 120 \
    --save-dir results_causal/esm2_l16

  # smoke
  python experiment_causal_features.py --layer-dir outputs_layerwise/esm2/layer_16 \
    --save-dir /tmp/causal_smoke --top-k 3 --n-control 3 --max-proteins 24
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
from tqdm import tqdm
from scipy.stats import mannwhitneyu

from sae import SparseAutoencoder
from cpu_stage import load_ref_seqs, load_layer
from experiment_activation_clamping import (
    extract_ca_coords, compute_true_contacts,
    predict_contacts_from_embeddings, top_l_precision,
)

warnings.filterwarnings("ignore")


class NormScaleSAEHook:
    """Forward hook: scale -> SAE encode -> (optionally ablate) -> decode -> unscale."""

    def __init__(self, sae, norm_scale, ablate_features=None):
        self.sae = sae
        self.norm_scale = float(norm_scale)
        self.ablate = ablate_features

    def __call__(self, module, inp, output):
        hidden = output[0] if isinstance(output, tuple) else output
        rest = output[1:] if isinstance(output, tuple) else None
        B, L, D = hidden.shape
        flat = hidden.reshape(-1, D) * self.norm_scale
        with torch.no_grad():
            z, _ = self.sae.encode(flat)
            if self.ablate is not None:
                z[:, self.ablate] = 0.0
            recon = self.sae.decode(z) / self.norm_scale
        recon = recon.reshape(B, L, D).to(hidden.dtype)
        return (recon,) + rest if rest is not None else recon


def esm2_contacts(sequences, hook, layer, device, batch_size=8):
    """Run ESM-2 (fp32) with optional hook, return final-layer embeddings per protein."""
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    model = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D").to(device).float().eval()
    handle = model.encoder.layer[layer].register_forward_hook(hook) if hook is not None else None
    special = set(getattr(tok, "all_special_ids", []) or [])
    for a in ("pad_token_id", "eos_token_id", "bos_token_id", "cls_token_id", "sep_token_id"):
        t = getattr(tok, a, None)
        if t is not None:
            special.add(int(t))
    embs = {}
    batches = [sequences[i:i + batch_size] for i in range(0, len(sequences), batch_size)]
    with torch.no_grad():
        for bi, bs in enumerate(batches):
            t = tok(bs, return_tensors="pt", add_special_tokens=True, padding=True,
                    truncation=True, max_length=1024).to(device)
            out = model(**t, output_hidden_states=True, return_dict=True)
            fin = out.hidden_states[-1]
            for i in range(len(bs)):
                attn = t["attention_mask"][i].bool()
                ids = t["input_ids"][i]
                keep = attn.clone()
                for sid in special:
                    keep &= (ids != sid)
                if keep.sum() == 0:
                    keep = attn
                embs[bi * batch_size + i] = fin[i, keep, :].detach().cpu().float().numpy()
            del out
    if handle is not None:
        handle.remove()
    del model
    return embs


def mean_precision(embs, sequences, uids, pdb_dir):
    """Mean top-L/5 long-range contact precision over proteins (skips missing PDBs)."""
    precs = []
    for gi, seq in enumerate(sequences):
        emb = embs.get(gi)
        if emb is None or emb.shape[0] != len(seq):
            continue
        pdb = Path(pdb_dir) / f"{str(uids[gi])[1:5].lower()}.pdb"
        if not pdb.exists():
            continue
        coords = extract_ca_coords(str(pdb), seq)
        if coords is None:
            continue
        contacts = compute_true_contacts(coords)
        if contacts.sum() == 0:
            continue
        scores = predict_contacts_from_embeddings(emb)
        precs.append(top_l_precision(scores, contacts))
    return np.array(precs, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser(description="E2 causal feature scoring")
    ap.add_argument("--layer-dir", required=True)
    ap.add_argument("--pdb-dir", default="cache/pdb_files")
    ap.add_argument("--top-k", type=int, default=15)
    ap.add_argument("--n-control", type=int, default=15)
    ap.add_argument("--max-proteins", type=int, default=120)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-dir", required=True)
    args = ap.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    layer_dir = Path(args.layer_dir)
    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")

    meta = json.loads((layer_dir / "META.json").read_text())
    embed_dim, hidden_dim = meta["embed_dim"], meta["sae_hidden_dim"]
    layer, norm_scale = meta["layer"], meta.get("norm_scale", 1.0)
    k_sparse = meta.get("k_sparse", 256)

    sae = SparseAutoencoder(input_dim=embed_dim, expansion=hidden_dim // embed_dim,
                            k_sparse=k_sparse, k_aux=meta.get("k_aux", 64))
    sae.load_state_dict(torch.load(layer_dir / "sae_model.pt", map_location="cpu"))
    sae = sae.to(device).float().eval()

    ss = pd.read_csv(layer_dir / "struct_seq_metrics.csv").sort_values("struct_delta", ascending=False)
    top_features = ss.head(args.top_k)["feature_idx"].astype(int).tolist()
    rng = np.random.default_rng(args.seed)
    mid = ss.iloc[len(ss) // 4: 3 * len(ss) // 4]["feature_idx"].astype(int).to_numpy()
    control_features = rng.choice(mid, size=min(args.n_control, len(mid)), replace=False).tolist()

    _, uids, lengths = load_layer(layer_dir)
    uids = [str(u) for u in uids]
    ref = load_ref_seqs(layer_dir)
    sequences = [ref[u] for u in uids]
    if args.max_proteins and args.max_proteins < len(uids):
        uids, sequences = uids[:args.max_proteins], sequences[:args.max_proteins]

    print("=" * 70)
    print(f"  E2 CAUSAL FEATURES — ESM-2 layer {layer} | device {device}")
    print(f"  top-K struct features: {top_features}")
    print("=" * 70)

    print("  Baseline (SAE passthrough, no ablation)...")
    base_emb = esm2_contacts(sequences, NormScaleSAEHook(sae, norm_scale), layer, device)
    base_prec = mean_precision(base_emb, sequences, uids, args.pdb_dir)
    base_mean = float(base_prec.mean()) if len(base_prec) else float("nan")
    print(f"    baseline mean top-L/5 precision = {base_mean:.4f}  ({len(base_prec)} proteins)")

    rows = []
    for kind, feats in [("struct", top_features), ("control", control_features)]:
        for f in tqdm(feats, desc=f"  ablating {kind}"):
            emb = esm2_contacts(sequences, NormScaleSAEHook(sae, norm_scale, ablate_features=[f]),
                                layer, device)
            prec = mean_precision(emb, sequences, uids, args.pdb_dir)
            n = min(len(prec), len(base_prec))
            delta = float((prec[:n] - base_prec[:n]).mean()) if n else float("nan")
            rows.append({"feature_idx": int(f), "kind": kind,
                         "ablated_precision": float(prec.mean()) if len(prec) else float("nan"),
                         "delta_vs_baseline": delta,
                         "struct_delta": float(ss.set_index("feature_idx").loc[f, "struct_delta"])})
    df = pd.DataFrame(rows).sort_values("delta_vs_baseline")
    df.to_csv(save_dir / "causal_feature_effects.csv", index=False)

    s_delta = df[df.kind == "struct"]["delta_vs_baseline"].dropna().to_numpy()
    c_delta = df[df.kind == "control"]["delta_vs_baseline"].dropna().to_numpy()
    test = {}
    if len(s_delta) and len(c_delta):
        u, p = mannwhitneyu(s_delta, c_delta, alternative="less")  # struct more negative?
        test = {"mannwhitney_u": float(u), "p_struct_more_harmful": float(p)}
    summary = {
        "layer_dir": str(layer_dir), "baseline_precision": base_mean,
        "mean_delta_struct": float(np.nanmean(s_delta)) if len(s_delta) else None,
        "mean_delta_control": float(np.nanmean(c_delta)) if len(c_delta) else None,
        "n_proteins_scored": int(len(base_prec)), **test,
    }
    (save_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot([s_delta, c_delta], labels=["top-struct", "random control"])
    ax.axhline(0, color="k", ls="--", lw=1)
    ax.set_ylabel("Δ top-L/5 precision (ablate - baseline)")
    ax.set_title(f"Causal effect of ablating features (ESM-2 L{layer})")
    fig.tight_layout(); fig.savefig(save_dir / "causal_effects.png", dpi=200); plt.close(fig)

    print("\n  Summary:"); print(json.dumps(summary, indent=2))
    print(f"\n  Saved to {save_dir}/")


if __name__ == "__main__":
    main()
