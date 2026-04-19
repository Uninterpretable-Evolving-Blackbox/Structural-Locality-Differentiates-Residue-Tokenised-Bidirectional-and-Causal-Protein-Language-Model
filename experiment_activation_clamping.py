#!/usr/bin/env python3
"""
experiment_activation_clamping.py — Causal evidence via SAE feature intervention
=================================================================================

Addresses Limitation 2 (correlational-only analysis) by performing activation
clamping on ESM-2 at the 50% depth layer (layer 16).

Experiment:
  1. Identify top-K SAE features with highest structural locality (Δ).
  2. Run zero-shot contact prediction from ESM-2 embeddings (baseline).
  3. Intervene during the forward pass via a PyTorch hook at layer 16:
     - Ablation:      clamp those K feature activations to exactly 0
     - Amplification:  clamp to 3× their per-protein maximum activation
  4. Measure top-L/5 long-range (|i−j| ≥ 12) contact precision.

The hook works as follows:
  - Intercept hidden states at layer 16 output
  - Encode through SAE → z
  - Modify z (ablation or amplification of target features)
  - Decode back → modified hidden state
  - Replace hidden state → propagates through layers 17–32

Usage:
  python experiment_activation_clamping.py \
    --layer-dir outputs_layerwise/esm2/layer_16 \
    --pdb-dir cache/pdb_files \
    --features-csv cache/residue_features.csv \
    --device cuda \
    --top-k 10 \
    --save-dir results_clamping

Prerequisites:
  - Trained SAE for ESM-2 layer 16 (sae_model.pt in layer-dir)
  - struct_seq_metrics.csv in layer-dir (from cpu_stage.py)
  - PDB files in pdb-dir
  - sequences.json + uids.json in layer-dir
"""

import argparse
import json
import gc
import warnings
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial import KDTree
from Bio.PDB import PDBParser
from Bio import pairwise2

from sae import SparseAutoencoder
from extract_embeddings import _get_hw_config, _get_autocast_context, _get_device

warnings.filterwarnings("ignore")


# =====================================================================
#                    CONTACT MAP GROUND TRUTH
# =====================================================================

def extract_ca_coords(pdb_path: str, ref_seq: str,
                      ) -> Optional[np.ndarray]:
    """Extract Cα coordinates aligned to the reference sequence.

    Returns (L, 3) array with NaN for unresolved residues, or None on failure.
    """
    aa3 = {
        "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
        "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
        "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
        "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
        "SEC": "U", "PYL": "O",
    }
    try:
        parser = PDBParser(QUIET=True)
        struct = parser.get_structure("s", pdb_path)
    except Exception:
        return None

    best_score, best_coords = None, None
    for model in struct:
        for chain in model:
            seq_c, coords_c = [], []
            for res in chain:
                if res.id[0] != " " or "CA" not in res:
                    continue
                seq_c.append(aa3.get(res.get_resname().upper(), "X"))
                coords_c.append(res["CA"].get_coord())
            if len(seq_c) < 10:
                continue
            chain_seq = "".join(seq_c)
            chain_coords = np.array(coords_c, dtype=np.float32)
            aln = pairwise2.align.globalms(
                ref_seq, chain_seq, 2, -1, -5, -0.5,
                one_alignment_only=True)
            if not aln:
                continue
            aln = aln[0]
            coords_ref = np.full((len(ref_seq), 3), np.nan, dtype=np.float32)
            ri, ci = -1, -1
            for a, b in zip(aln.seqA, aln.seqB):
                if a != "-":
                    ri += 1
                if b != "-":
                    ci += 1
                if a != "-" and b != "-" and ri < len(ref_seq) and ci < len(chain_coords):
                    coords_ref[ri] = chain_coords[ci]
            if best_score is None or aln.score > best_score:
                best_score, best_coords = aln.score, coords_ref
        break  # first model only
    return best_coords


def compute_true_contacts(coords: np.ndarray, cutoff: float = 8.0,
                          seq_gap: int = 12) -> np.ndarray:
    """Binary contact map (L, L) for long-range contacts."""
    L = coords.shape[0]
    valid = ~np.isnan(coords).any(axis=1)
    cmap = np.zeros((L, L), dtype=bool)
    idx = np.where(valid)[0]
    if len(idx) < 2:
        return cmap
    dists = KDTree(coords[valid]).sparse_distance_matrix(
        KDTree(coords[valid]), cutoff).toarray()
    for a_i, a_real in enumerate(idx):
        for b_i, b_real in enumerate(idx):
            if abs(int(a_real) - int(b_real)) >= seq_gap and dists[a_i, b_i] > 0:
                cmap[a_real, b_real] = True
    return cmap


# =====================================================================
#         EMBEDDING-BASED CONTACT PREDICTION (APC-corrected)
# =====================================================================

def predict_contacts_from_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Predict contact scores from residue embeddings via APC-corrected
    outer product.

    Args:
        embeddings: (L, D) residue embeddings

    Returns:
        (L, L) symmetric contact score matrix (higher = more likely contact)
    """
    # L2-normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb_norm = embeddings / norms

    # Pairwise cosine similarity
    S = emb_norm @ emb_norm.T  # (L, L)

    # Average Product Correction (APC) — removes phylogenetic background
    row_mean = S.mean(axis=1, keepdims=True)
    col_mean = S.mean(axis=0, keepdims=True)
    global_mean = S.mean()
    S_apc = S - (row_mean * col_mean) / (global_mean + 1e-8)

    return S_apc


def top_l_precision(scores: np.ndarray, contacts: np.ndarray,
                    seq_gap: int = 12, fraction: float = 0.2) -> float:
    """Top-L/5 long-range precision.

    Args:
        scores: (L, L) predicted contact scores
        contacts: (L, L) true binary contacts
        seq_gap: minimum sequence separation for long-range
        fraction: L/fraction predictions to evaluate (default L/5)

    Returns:
        Precision (float)
    """
    L = scores.shape[0]
    k = max(1, int(L * fraction))

    # Mask: only upper triangle, long-range
    mask = np.zeros_like(scores, dtype=bool)
    for i in range(L):
        for j in range(i + seq_gap, L):
            mask[i, j] = True

    # Get top-k predictions among valid pairs
    valid_scores = scores[mask]
    valid_contacts = contacts[mask]

    if len(valid_scores) == 0:
        return 0.0

    topk_idx = np.argsort(valid_scores)[::-1][:k]
    precision = valid_contacts[topk_idx].mean()
    return float(precision)


# =====================================================================
#              ESM-2 FORWARD PASS WITH SAE HOOK
# =====================================================================

class SAEInterventionHook:
    """PyTorch forward hook that intercepts hidden states at a target layer,
    passes them through an SAE, optionally modifies feature activations,
    and replaces the hidden states with the SAE reconstruction.
    """

    def __init__(self, sae_model: SparseAutoencoder, device: str,
                 mode: str = "none", target_features: Optional[np.ndarray] = None,
                 amplification_factor: float = 3.0):
        """
        Args:
            sae_model: trained SAE
            device: torch device
            mode: "none" (passthrough with SAE reconstruction),
                  "ablate" (clamp target features to 0),
                  "amplify" (clamp target features to amplification_factor × max)
            target_features: array of feature indices to intervene on
            amplification_factor: multiplier for amplification mode
        """
        self.sae = sae_model
        self.device = device
        self.mode = mode
        self.target_features = target_features
        self.amp_factor = amplification_factor
        # Store per-protein max activations (computed per forward pass)
        self._feature_maxes = None

    def __call__(self, module, input, output):
        """Hook function — modifies hidden states in-place."""
        # output is a tuple: (hidden_states, ...) or just hidden_states
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
        else:
            hidden = output
            rest = None

        B, L, D = hidden.shape
        flat = hidden.reshape(-1, D)  # (B*L, D)

        with torch.no_grad():
            z, z_pre = self.sae.encode(flat)

            if self.mode == "ablate" and self.target_features is not None:
                z[:, self.target_features] = 0.0

            elif self.mode == "amplify" and self.target_features is not None:
                # Per-sequence max for each target feature
                z_reshaped = z.reshape(B, L, -1)
                for feat_idx in self.target_features:
                    for b in range(B):
                        seq_max = z_reshaped[b, :, feat_idx].max().item()
                        if seq_max > 0:
                            z_reshaped[b, :, feat_idx] = seq_max * self.amp_factor
                z = z_reshaped.reshape(-1, z.shape[-1])

            reconstructed = self.sae.decode(z)

        reconstructed = reconstructed.reshape(B, L, D)

        if rest is not None:
            return (reconstructed,) + rest
        return reconstructed


def run_esm2_with_hook(
    sequences: List[str],
    hook: Optional[SAEInterventionHook],
    target_layer: int,
    device: str,
    batch_size: int = 8,
) -> Dict[str, np.ndarray]:
    """Run ESM-2 forward pass with optional hook at target_layer.

    Returns dict mapping uid-index to (L, D) final-layer embeddings.
    """
    from transformers import AutoTokenizer, AutoModel

    config = _get_hw_config(device)

    model_kwargs = {}
    if config["use_amp"]:
        model_kwargs["torch_dtype"] = config["dtype"]

    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    model = AutoModel.from_pretrained(
        "facebook/esm2_t33_650M_UR50D", **model_kwargs
    ).to(device).eval()

    # Register hook at the target transformer layer
    handle = None
    if hook is not None:
        # ESM-2 layers are at model.encoder.layer[i]
        # target_layer N corresponds to encoder.layer[N]
        target_module = model.encoder.layer[target_layer]
        handle = target_module.register_forward_hook(hook)

    embeddings = {}
    batches = [sequences[i:i + batch_size]
               for i in range(0, len(sequences), batch_size)]

    with torch.no_grad():
        for batch_idx, batch_seqs in enumerate(tqdm(batches, desc="ESM-2 fwd")):
            tokens = tokenizer(
                batch_seqs, return_tensors="pt", add_special_tokens=True,
                padding=True, truncation=True, max_length=1024,
            ).to(device)

            with _get_autocast_context(device, config["dtype"], config["use_amp"]):
                outputs = model(**tokens, output_hidden_states=True, return_dict=True)

            # Final layer hidden states (index -1 or index 33)
            final_hidden = outputs.hidden_states[-1]  # (B, seq_len, D)

            for i, seq in enumerate(batch_seqs):
                # Exclude special tokens
                attn_mask = tokens["attention_mask"][i].bool()
                input_ids = tokens["input_ids"][i]
                special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
                for attr in ("pad_token_id", "eos_token_id", "bos_token_id",
                             "cls_token_id", "sep_token_id"):
                    tid = getattr(tokenizer, attr, None)
                    if tid is not None:
                        special_ids.add(int(tid))
                keep = attn_mask.clone()
                for sid in special_ids:
                    keep &= (input_ids != sid)
                if keep.sum() == 0:
                    keep = attn_mask

                global_idx = batch_idx * batch_size + i
                emb = final_hidden[i, keep, :].detach().cpu().float().numpy()
                embeddings[global_idx] = emb

            del outputs

    if handle is not None:
        handle.remove()

    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return embeddings


# =====================================================================
#                        MAIN EXPERIMENT
# =====================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Activation clamping experiment for causal SAE evidence")
    ap.add_argument("--layer-dir", required=True,
                    help="Path to outputs_layerwise/esm2/layer_16")
    ap.add_argument("--pdb-dir", default="cache/pdb_files")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--top-k", type=int, default=10,
                    help="Number of top structural features to intervene on")
    ap.add_argument("--amp-factor", type=float, default=3.0,
                    help="Amplification multiplier")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--save-dir", default="results_clamping")
    ap.add_argument("--max-proteins", type=int, default=200,
                    help="Subsample proteins for faster iteration (0=all)")
    args = ap.parse_args()

    layer_dir = Path(args.layer_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    pdb_dir = Path(args.pdb_dir)
    device = _get_device(args.device)

    # ------------------------------------------------------------------
    # 1. Load metadata and identify target features
    # ------------------------------------------------------------------
    print("=" * 70)
    print("  ACTIVATION CLAMPING EXPERIMENT")
    print("=" * 70)

    meta = json.loads((layer_dir / "META.json").read_text())
    target_layer = meta["layer"]
    embed_dim = meta["embed_dim"]
    hidden_dim = meta["sae_hidden_dim"]
    expansion = hidden_dim // embed_dim

    print(f"  Model: ESM-2, Layer: {target_layer}")
    print(f"  SAE: {embed_dim}D -> {hidden_dim}D (expansion={expansion})")

    # Load struct_seq_metrics to find top-K structural features
    ss_df = pd.read_csv(layer_dir / "struct_seq_metrics.csv")
    ss_df_sorted = ss_df.sort_values("struct_delta", ascending=False)
    top_features = ss_df_sorted.head(args.top_k)["feature_idx"].values
    top_deltas = ss_df_sorted.head(args.top_k)["struct_delta"].values

    print(f"\n  Top-{args.top_k} structural features:")
    for feat, delta in zip(top_features, top_deltas):
        print(f"    Feature {feat:5d}:  struct_delta = {delta:+.4f}")

    # ------------------------------------------------------------------
    # 2. Load SAE model
    # ------------------------------------------------------------------
    print("\n  Loading SAE model...")
    sae = SparseAutoencoder(
        input_dim=embed_dim,
        expansion=expansion,
        k_sparse=meta.get("k_sparse", 1024),
    )
    state_dict = torch.load(layer_dir / "sae_model.pt", map_location="cpu",
                            weights_only=True)
    sae.load_state_dict(state_dict)
    sae = sae.to(device).eval()

    # ------------------------------------------------------------------
    # 3. Load sequences and PDB structures
    # ------------------------------------------------------------------
    uids = json.loads((layer_dir / "uids.json").read_text())
    seqs_raw = json.loads((layer_dir / "sequences.json").read_text())
    if isinstance(seqs_raw, dict):
        sequences = [seqs_raw[uid] for uid in uids]
    elif isinstance(seqs_raw, list):
        sequences = seqs_raw
    else:
        raise ValueError("Unexpected sequences.json format")

    # Filter to proteins with PDB structures
    valid_indices = []
    valid_uids = []
    valid_seqs = []
    valid_coords = []

    print(f"\n  Loading PDB structures from {pdb_dir}...")
    for i, (uid, seq) in enumerate(tqdm(zip(uids, sequences),
                                         total=len(uids), desc="  PDBs")):
        pdb_path = pdb_dir / f"{str(uid)[1:5].lower()}.pdb"
        if not pdb_path.exists():
            continue
        coords = extract_ca_coords(str(pdb_path), seq)
        if coords is None:
            continue
        valid_mask = ~np.isnan(coords).any(axis=1)
        if valid_mask.sum() < 30:
            continue
        valid_indices.append(i)
        valid_uids.append(uid)
        valid_seqs.append(seq)
        valid_coords.append(coords)

    print(f"  {len(valid_seqs)} proteins with valid structures")

    # Subsample if requested
    if args.max_proteins > 0 and len(valid_seqs) > args.max_proteins:
        rng = np.random.RandomState(42)
        keep = sorted(rng.choice(len(valid_seqs), args.max_proteins, replace=False))
        valid_uids = [valid_uids[i] for i in keep]
        valid_seqs = [valid_seqs[i] for i in keep]
        valid_coords = [valid_coords[i] for i in keep]
        print(f"  Subsampled to {len(valid_seqs)} proteins")

    # ------------------------------------------------------------------
    # 4. Run three conditions: baseline, ablation, amplification
    # ------------------------------------------------------------------
    target_feat_tensor = torch.tensor(top_features, dtype=torch.long).to(device)

    conditions = {
        "baseline": SAEInterventionHook(sae, device, mode="none"),
        "ablation": SAEInterventionHook(sae, device, mode="ablate",
                                        target_features=target_feat_tensor),
        "amplification": SAEInterventionHook(sae, device, mode="amplify",
                                             target_features=target_feat_tensor,
                                             amplification_factor=args.amp_factor),
    }

    all_results = {}

    for cond_name, hook in conditions.items():
        print(f"\n{'─' * 60}")
        print(f"  Condition: {cond_name.upper()}")
        print(f"{'─' * 60}")

        # Run ESM-2 with hook
        embeddings = run_esm2_with_hook(
            valid_seqs, hook, target_layer, device,
            batch_size=args.batch_size,
        )

        # Evaluate contact prediction per protein
        precisions = []
        for idx in range(len(valid_seqs)):
            emb = embeddings[idx]  # (L, D)
            coords = valid_coords[idx]  # (L, 3)
            L = len(valid_seqs[idx])

            # Ensure shapes match
            if emb.shape[0] != L:
                print(f"    Warning: embedding length {emb.shape[0]} != seq "
                      f"length {L} for protein {idx}, skipping")
                continue

            # Predict contacts from embeddings
            scores = predict_contacts_from_embeddings(emb)
            true_contacts = compute_true_contacts(coords)

            # Top-L/5 precision
            prec = top_l_precision(scores, true_contacts, seq_gap=12,
                                   fraction=0.2)
            precisions.append(prec)

        mean_prec = np.mean(precisions) if precisions else 0.0
        std_prec = np.std(precisions) if precisions else 0.0
        median_prec = np.median(precisions) if precisions else 0.0

        all_results[cond_name] = {
            "precisions": precisions,
            "mean": mean_prec,
            "std": std_prec,
            "median": median_prec,
            "n_proteins": len(precisions),
        }

        print(f"\n  Top-L/5 long-range precision:")
        print(f"    Mean:   {mean_prec:.4f} +/- {std_prec:.4f}")
        print(f"    Median: {median_prec:.4f}")
        print(f"    N:      {len(precisions)} proteins")

    # ------------------------------------------------------------------
    # 5. Statistical comparison
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("  COMPARISON")
    print(f"{'=' * 70}")

    from scipy.stats import wilcoxon

    baseline_prec = np.array(all_results["baseline"]["precisions"])

    for cond in ["ablation", "amplification"]:
        cond_prec = np.array(all_results[cond]["precisions"])
        n = min(len(baseline_prec), len(cond_prec))
        if n < 10:
            print(f"\n  {cond}: too few proteins ({n}) for statistical test")
            continue

        diff = cond_prec[:n] - baseline_prec[:n]
        stat, p = wilcoxon(diff)
        mean_diff = diff.mean()

        direction = "DROP" if mean_diff < 0 else "INCREASE"
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

        print(f"\n  {cond.upper()} vs BASELINE:")
        print(f"    Mean Δ precision: {mean_diff:+.4f}")
        print(f"    Wilcoxon p:       {p:.2e} {sig}")
        print(f"    Direction:        {direction}")

    # ------------------------------------------------------------------
    # 6. Save results
    # ------------------------------------------------------------------
    # CSV summary
    rows = []
    for cond, res in all_results.items():
        rows.append({
            "condition": cond,
            "mean_precision": res["mean"],
            "std_precision": res["std"],
            "median_precision": res["median"],
            "n_proteins": res["n_proteins"],
        })
    pd.DataFrame(rows).to_csv(save_dir / "clamping_summary.csv", index=False)

    # Per-protein details
    detail_rows = []
    for cond, res in all_results.items():
        for i, prec in enumerate(res["precisions"]):
            detail_rows.append({
                "condition": cond,
                "protein_idx": i,
                "precision": prec,
            })
    pd.DataFrame(detail_rows).to_csv(save_dir / "clamping_per_protein.csv",
                                      index=False)

    # Target features
    pd.DataFrame({
        "feature_idx": top_features,
        "struct_delta": top_deltas,
    }).to_csv(save_dir / "target_features.csv", index=False)

    # ------------------------------------------------------------------
    # 7. Plot
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Box plot of per-protein precision
    ax = axes[0]
    data_for_box = []
    labels_for_box = []
    for cond in ["baseline", "ablation", "amplification"]:
        data_for_box.append(all_results[cond]["precisions"])
        labels_for_box.append(cond.capitalize())

    bp = ax.boxplot(data_for_box, labels=labels_for_box, patch_artist=True,
                    showfliers=False)
    colors = ["#4CAF50", "#F44336", "#2196F3"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel("Top-L/5 Long-range Precision")
    ax.set_title("Contact Prediction Under SAE Feature Intervention")
    ax.grid(alpha=0.3, axis="y")

    # Panel B: Paired difference (baseline - ablation, amplification - baseline)
    ax = axes[1]
    if len(baseline_prec) >= 10:
        n = min(len(baseline_prec), len(all_results["ablation"]["precisions"]))
        ablation_diff = np.array(all_results["ablation"]["precisions"][:n]) - baseline_prec[:n]
        n2 = min(len(baseline_prec), len(all_results["amplification"]["precisions"]))
        amp_diff = np.array(all_results["amplification"]["precisions"][:n2]) - baseline_prec[:n2]

        parts = ax.violinplot([ablation_diff, amp_diff], positions=[0, 1],
                              showmeans=True, showmedians=True)
        ax.axhline(0, color="grey", ls="--", lw=1)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Ablation\n- Baseline", "Amplification\n- Baseline"])
        ax.set_ylabel("Δ Precision")
        ax.set_title("Per-Protein Precision Change")
        ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(save_dir / "clamping_results.png", dpi=300)
    fig.savefig(save_dir / "clamping_results.pdf")
    plt.close(fig)

    print(f"\n  Results saved to {save_dir}/")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
