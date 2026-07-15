#!/usr/bin/env python3
"""
experiment_steering_sweep.py — E3-v2: InterPLM-style steering dose-response
===========================================================================

Steers SAE features and measures a concept-specific, masked-recovery readout
(InterPLM Fig 7) across a clamp dose-response [0, 0.5, 1, 2, 4] x feature-max:
  - mask the anchor's 3D contact partners AND sequence-separation-matched
    non-contact controls,
  - measure the change in p(true residue) at those masked positions as we clamp
    one feature at the (unmasked) anchor.

De-circularised feature selection (review feedback): three INDEPENDENTLY-chosen
feature groups are steered with the SAME readout —
  - "annotation": top features by concept-F1 (annotation-aligned; independent of
    L_struct, with which it rank-correlates only ~0.07) — the InterPLM analogue;
  - "structural": top features by L_struct (struct_delta);
  - "random": random control features.
Selecting "annotation" features by concept-F1 and testing them on a STRUCTURAL
(contact) readout removes the circularity of selecting and testing on the same
metric.

Significance uses a PROTEIN-LEVEL cluster bootstrap (B=1000) on the contact
specificity (Δp_contact − Δp_noncontact), matching the paper's main statistics —
a rigour improvement over InterPLM's (illustrative, untested) steering figures.

Usage:
  python experiment_steering_sweep.py --layer-dir outputs_layerwise/esm2/layer_16 \
    --concept-csv results_concept_f1/esm2_l16/feature_concept_best.csv \
    --pdb-dir cache/pdb_files --top-k 8 --max-proteins 50 --scales 0,0.5,1,2,4 \
    --save-dir results_steering_sweep/esm2_l16

  # smoke
  python experiment_steering_sweep.py --layer-dir outputs_layerwise/esm2/layer_16 \
    --concept-csv results_concept_f1/esm2_l16/feature_concept_best.csv \
    --save-dir /tmp/sweep_smoke --top-k 2 --max-proteins 8 --scales 0,1,4 --n-boot 200
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
from cluster_bootstrap import load_uid_clusters, resample_indices_by_cluster

warnings.filterwarnings("ignore")

GROUP_COLORS = {"annotation": "#E91E63", "structural": "#2196F3", "random": "#999999"}


class ClampHook:
    """scale -> SAE encode -> (optionally clamp one feature at one pos) -> decode -> unscale."""

    def __init__(self, sae, norm_scale, feature=None, anchor=None, target=None):
        self.sae, self.ns = sae, float(norm_scale)
        self.feature, self.anchor, self.target = feature, anchor, target

    def __call__(self, module, inp, output):
        hidden = output[0] if isinstance(output, tuple) else output
        rest = output[1:] if isinstance(output, tuple) else None
        B, L, D = hidden.shape
        flat = hidden.reshape(-1, D) * self.ns
        with torch.no_grad():
            z, _ = self.sae.encode(flat)
            if self.feature is not None and self.target is not None:
                z = z.reshape(B, L, -1)
                z[:, self.anchor, self.feature] = self.target
                z = z.reshape(-1, z.shape[-1])
            recon = self.sae.decode(z) / self.ns
        recon = recon.reshape(B, L, D).to(hidden.dtype)
        return (recon,) + rest if rest is not None else recon


def matched_controls(coords, anchor, partners, sep=12, n=None):
    """Pick non-contact residues sequence-separation-matched to the partners."""
    valid = ~np.isnan(coords).any(axis=1)
    L = coords.shape[0]
    eligible = [j for j in range(L) if valid[j] and abs(j - anchor) >= sep and j not in partners]
    if not eligible:
        return []
    elig = np.array(eligible)
    chosen, used = [], set()
    for p in partners:
        target_sep = abs(p - anchor)
        order = np.argsort(np.abs(np.abs(elig - anchor) - target_sep))
        for idx in order:
            j = int(elig[idx])
            if j not in used:
                used.add(j); chosen.append(j); break
        if n and len(chosen) >= n:
            break
    return chosen


def select_groups(layer_dir, concept_csv, top_k, n_random, seed):
    """Three independent feature groups: annotation (concept-F1), structural (L_struct), random."""
    ss = pd.read_csv(layer_dir / "struct_seq_metrics.csv")
    n_feat = len(ss)
    rng = np.random.default_rng(seed)
    groups = {}
    # structural: top by struct_delta
    groups["structural"] = (ss.sort_values("struct_delta", ascending=False)
                            .head(top_k)["feature_idx"].astype(int).tolist())
    # annotation: top by concept test-F1 (independent selection signal)
    if concept_csv and Path(concept_csv).exists():
        cdf = pd.read_csv(concept_csv)
        score = "test_f1" if "test_f1" in cdf.columns else "val_f1"
        groups["annotation"] = (cdf.sort_values(score, ascending=False)
                                .head(top_k)["feature_idx"].astype(int).tolist())
    # random: avoid overlap with the selected sets
    selected = set(sum(groups.values(), []))
    pool = [f for f in ss["feature_idx"].astype(int).tolist() if f not in selected]
    groups["random"] = rng.choice(pool, size=min(n_random, len(pool)), replace=False).tolist()
    return groups


def protein_bootstrap(df, scale, group_a, group_b, B=1000, seed=0,
                      cluster_of=None, two_stage=True):
    """Cluster bootstrap of (group_a - group_b) contact specificity at a scale.

    PAIRED across groups: each resample draws proteins once and takes BOTH
    groups' values for the same resampled proteins, then applies a paired
    non-NaN mask. (Fixes the earlier per-group independent dropna, audit 9.3.)
    If cluster_of (uid->SCOPe cluster key) is given, resamples whole folds
    instead of individual proteins; cluster_of=None reduces to protein bootstrap.
    """
    rng = np.random.default_rng(seed)
    sub = df[df.scale == scale].copy()
    sub["spec"] = sub["dp_contact"] - sub["dp_noncontact"]
    piv = sub.groupby(["uid", "kind"])["spec"].mean().unstack("kind")
    if group_a not in piv.columns or group_b not in piv.columns:
        return None
    uids = [str(u) for u in piv.index.to_numpy()]
    a = piv[group_a].to_numpy(dtype=float)
    b = piv[group_b].to_numpy(dtype=float)
    if cluster_of is not None:
        keys = np.array([cluster_of.get(u, f"__singleton__{u}") for u in uids])
        _, item_clusters = np.unique(keys, return_inverse=True)
    else:
        item_clusters = np.arange(len(uids))
    diffs, as_, bs_ = [], [], []
    for _ in range(B):
        samp = resample_indices_by_cluster(item_clusters, rng, two_stage=two_stage)
        aa, bb = a[samp], b[samp]
        m = ~(np.isnan(aa) | np.isnan(bb))     # paired: same proteins in both groups
        if m.sum() == 0:
            continue
        as_.append(float(aa[m].mean())); bs_.append(float(bb[m].mean()))
        diffs.append(float(aa[m].mean() - bb[m].mean()))
    if not diffs:
        return None
    diffs = np.array(diffs); as_ = np.array(as_); bs_ = np.array(bs_)
    mfull = ~(np.isnan(a) | np.isnan(b))
    return {
        f"mean_{group_a}": float(a[mfull].mean()) if mfull.any() else float("nan"),
        f"ci_{group_a}": [float(np.percentile(as_, 2.5)), float(np.percentile(as_, 97.5))],
        f"mean_{group_b}": float(b[mfull].mean()) if mfull.any() else float("nan"),
        f"ci_{group_b}": [float(np.percentile(bs_, 2.5)), float(np.percentile(bs_, 97.5))],
        "mean_diff": float(np.mean(diffs)),
        "ci_diff": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
        "p_diff_le_0": float(np.mean(diffs <= 0)),
        "n_paired_proteins": int(mfull.sum()),
    }


def main():
    ap = argparse.ArgumentParser(description="E3-v2 steering dose-response sweep")
    ap.add_argument("--layer-dir", required=True)
    ap.add_argument("--concept-csv", default=None, help="feature_concept_best.csv from E0 (annotation group)")
    ap.add_argument("--pdb-dir", default="cache/pdb_files")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--n-random", type=int, default=8)
    ap.add_argument("--max-proteins", type=int, default=50)
    ap.add_argument("--scales", default="0,0.5,1,2,4")
    ap.add_argument("--contact-cutoff", type=float, default=8.0)
    ap.add_argument("--seq-gap-min", type=int, default=12)
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

    groups = select_groups(layer_dir, args.concept_csv, args.top_k, args.n_random, args.seed)
    print("  Feature groups:", {k: v[:5] for k, v in groups.items()})

    from transformers import AutoTokenizer, AutoModelForMaskedLM
    tok = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    model = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t33_650M_UR50D").to(device).float().eval()
    mask_id = tok.mask_token_id
    special = set(getattr(tok, "all_special_ids", []) or [])
    for a in ("pad_token_id", "eos_token_id", "bos_token_id", "cls_token_id", "sep_token_id"):
        t = getattr(tok, a, None)
        if t is not None:
            special.add(int(t))

    chosen = []
    for gi, uid in enumerate(uids):
        if (Path(args.pdb_dir) / f"{uid[1:5].lower()}.pdb").exists():
            chosen.append(gi)
        if len(chosen) >= args.max_proteins:
            break

    print("=" * 70)
    print(f"  E3-v2 STEERING SWEEP — ESM-2 L{layer} | {device} | {len(chosen)} proteins | scales {scales}")
    print("=" * 70)

    def residue_token_index(t):
        ids = t["input_ids"][0]
        keep = t["attention_mask"][0].bool().clone()
        for sid in special:
            keep &= (ids != sid)
        return torch.where(keep)[0]

    rows = []
    for kind, feats in groups.items():
        for f in tqdm(feats, desc=f"  {kind}"):
            for gi in chosen:
                seq = ref[uids[gi]]
                base = offsets[gi]
                acts = np.asarray(Z[base:base + len(seq), f], dtype=np.float32)
                if acts.max() <= 0:
                    continue
                anchor = int(acts.argmax()); fmax = float(acts.max())
                pdb = Path(args.pdb_dir) / f"{uids[gi][1:5].lower()}.pdb"
                coords = extract_ca_coords(str(pdb), seq)
                if coords is None:
                    continue
                valid = ~np.isnan(coords).any(axis=1)
                if not valid[anchor]:
                    continue
                idx = np.where(valid)[0]
                near = KDTree(coords[valid]).query_ball_point(coords[anchor], r=args.contact_cutoff)
                partners = [int(idx[n]) for n in near if abs(int(idx[n]) - anchor) >= args.seq_gap_min]
                if not partners:
                    continue
                controls = matched_controls(coords, anchor, set(partners),
                                            sep=args.seq_gap_min, n=len(partners))
                if not controls:
                    continue
                t = tok([seq], return_tensors="pt", add_special_tokens=True).to(device)
                ridx = residue_token_index(t)
                anchor_tok = int(ridx[anchor])
                masked_res = partners + controls
                masked_tok = [int(ridx[r]) for r in masked_res]
                true_ids = t["input_ids"][0, masked_tok].clone()
                t["input_ids"][0, masked_tok] = mask_id
                is_contact = np.array([True] * len(partners) + [False] * len(controls))

                def probs_true():
                    with torch.no_grad():
                        logits = model(**t).logits[0]
                    p = F.softmax(logits[masked_tok], dim=-1)
                    return p[torch.arange(len(masked_tok)), true_ids].detach().cpu().numpy()

                h = model.esm.encoder.layer[layer].register_forward_hook(ClampHook(sae, norm_scale))
                p_base = probs_true(); h.remove()
                for scale in scales:
                    h = model.esm.encoder.layer[layer].register_forward_hook(
                        ClampHook(sae, norm_scale, feature=f, anchor=anchor_tok, target=scale * fmax))
                    p_s = probs_true(); h.remove()
                    dp = p_s - p_base
                    rows.append({
                        "kind": kind, "feature_idx": int(f), "uid": uids[gi], "scale": scale,
                        "dp_contact": float(dp[is_contact].mean()),
                        "dp_noncontact": float(dp[~is_contact].mean()),
                        "n_contacts": int(is_contact.sum()),
                    })

    df = pd.DataFrame(rows)
    df.to_csv(save_dir / "steering_sweep_per_event.csv", index=False)
    if not len(df):
        print("  No events collected."); return

    agg = df.groupby(["kind", "scale"]).agg(
        dp_contact=("dp_contact", "mean"), dp_noncontact=("dp_noncontact", "mean"),
        n=("uid", "count")).reset_index()
    agg["contact_minus_noncontact"] = agg["dp_contact"] - agg["dp_noncontact"]
    agg.to_csv(save_dir / "steering_sweep_summary.csv", index=False)

    present = [g for g in ["annotation", "structural", "random"] if g in groups and (df.kind == g).any()]
    slopes = {}
    for kind in present:
        sub = agg[agg.kind == kind].sort_values("scale")
        if len(sub) >= 2:
            slopes[kind] = float(np.polyfit(sub["scale"], sub["contact_minus_noncontact"], 1)[0])

    # cluster bootstrap at the largest clamp scale: each group vs random
    max_scale = max(scales)
    cluster_of = (load_uid_clusters(args.fasta, level=args.cluster_level)
                  if args.cluster_level != "protein" and Path(args.fasta).exists() else None)
    boot = {}
    if "random" in present:
        for g in [x for x in present if x != "random"]:
            r = protein_bootstrap(df, max_scale, g, "random", B=args.n_boot, seed=args.seed,
                                  cluster_of=cluster_of, two_stage=args.two_stage)
            if r:
                boot[f"{g}_vs_random@{max_scale}x"] = r

    (save_dir / "summary.json").write_text(json.dumps({
        "layer": layer, "scales": scales, "slopes_contact_specificity": slopes,
        "cluster_level": args.cluster_level, "two_stage": bool(args.two_stage),
        "bootstrap": boot, "n_events": int(len(df)),
        "groups": {k: len(v) for k, v in groups.items()}}, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for kind in present:
        c = GROUP_COLORS.get(kind, "#000")
        sub = agg[agg.kind == kind].sort_values("scale")
        axes[0].plot(sub["scale"], sub["dp_contact"], marker="o", color=c, label=f"{kind} contact")
        axes[0].plot(sub["scale"], sub["dp_noncontact"], marker="s", ls="--", color=c, alpha=0.5)
        axes[1].plot(sub["scale"], sub["contact_minus_noncontact"], marker="o", color=c, label=kind)
    axes[0].axhline(0, color="k", lw=0.8); axes[0].set_xlabel("clamp scale (x feature max)")
    axes[0].set_ylabel("Δ p(true residue) at masked position"); axes[0].legend(fontsize=8)
    axes[0].set_title("Masked-recovery dose-response (solid=contact, dashed=non-contact)")
    axes[1].axhline(0, color="k", lw=0.8); axes[1].set_xlabel("clamp scale (x feature max)")
    axes[1].set_ylabel("contact − non-contact Δp"); axes[1].legend()
    axes[1].set_title("Contact specificity of steering")
    fig.tight_layout(); fig.savefig(save_dir / "steering_sweep.png", dpi=200); plt.close(fig)

    print("\n  Summary (Δp at contacts vs non-contacts by scale):")
    print(agg.to_string(index=False))
    print(f"\n  Contact-specificity slopes: {slopes}")
    print(f"  Bootstrap (vs random @ {max_scale}x): {json.dumps(boot, indent=2)}")
    print(f"  Saved to {save_dir}/")


if __name__ == "__main__":
    main()
