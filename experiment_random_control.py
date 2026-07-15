#!/usr/bin/env python3
"""
experiment_random_control.py — C1: randomized-weights negative control
======================================================================

InterPLM's key control (Simon & Zou, Fig 3): train an SAE on embeddings from a
RANDOMLY-INITIALISED PLM. If concept alignment (E0) and structural locality (E4)
require a *trained* model, both should collapse to floor on the random model.

This proves the bidirectional-vs-causal dissociation is a property of LEARNED
representations, not of tokenisation / random projection geometry.

Pipeline (mirrors run_unsupervised.py for one layer, with random weights):
  1. Instantiate the PLM from config with random weights (no pretrained load).
  2. Extract hidden states at the target layer (special tokens trimmed).
  3. Protein-level split (reuse the trained layer's val_uids).
  4. Bricken normalise (train-only scale), train TopK SAE, extract Z.
  5. Write a control layer dir that experiment_concept_f1.py / experiment_null_
     calibration.py can consume unchanged.

Usage:
  python experiment_random_control.py --ref-layer-dir outputs_layerwise/esm2/layer_16 \
    --model esm2 --layer 16 --out-layer-dir outputs_random/esm2/layer_16

  # smoke: few epochs
  python experiment_random_control.py --ref-layer-dir outputs_layerwise/esm2/layer_16 \
    --model esm2 --layer 16 --out-layer-dir /tmp/rand_esm2_l16 --epochs 4 --max-proteins 120
"""

import argparse
import json
import os
import warnings
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from cpu_stage import load_layer, load_ref_seqs
from train_sae import (
    train_sae, extract_sae_features, compute_explained_variance, compute_norm_scale,
)
from extract_embeddings import _get_device, _get_hw_config, _get_autocast_context

warnings.filterwarnings("ignore")

MODEL_NAMES = {
    "esm2": "facebook/esm2_t33_650M_UR50D",
    "esm2_small": "facebook/esm2_t12_35M_UR50D",
}


def extract_random_esm2(sequences, layer, device, batch_size=16, max_length=1024,
                        model_name="facebook/esm2_t33_650M_UR50D", seed=0):
    """Extract hidden states at `layer` from a RANDOM-INIT ESM-2.

    Residue alignment matches extract_esm2_embeddings: layer N -> hidden_states[N+1],
    special tokens removed.
    """
    from transformers import AutoTokenizer, AutoModel, AutoConfig
    torch.manual_seed(seed)
    cfg = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_config(cfg).to(device).eval()  # RANDOM weights
    print(f"  Random-init {model_name} instantiated ({sum(p.numel() for p in model.parameters())/1e6:.0f}M params)")

    config = _get_hw_config(device)
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    for attr in ("pad_token_id", "eos_token_id", "bos_token_id", "cls_token_id", "sep_token_id"):
        tid = getattr(tokenizer, attr, None)
        if tid is not None:
            special_ids.add(int(tid))

    buf = []
    batches = [sequences[i:i + batch_size] for i in range(0, len(sequences), batch_size)]
    with torch.no_grad():
        for batch_seqs in tqdm(batches, desc="  random ESM-2 fwd"):
            tok = tokenizer(batch_seqs, return_tensors="pt", add_special_tokens=True,
                            padding=True, truncation=True, max_length=max_length).to(device)
            with _get_autocast_context(device, config["dtype"], config["use_amp"]):
                out = model(**tok, output_hidden_states=True, return_dict=True)
            hs = out.hidden_states[layer + 1]  # layer N -> block N+1
            for i in range(len(batch_seqs)):
                attn = tok["attention_mask"][i].bool()
                ids = tok["input_ids"][i]
                keep = attn.clone()
                for sid in special_ids:
                    keep &= (ids != sid)
                if keep.sum() == 0:
                    keep = attn
                buf.append(hs[i, keep, :].detach().cpu().float().numpy())
            del out
    return np.concatenate(buf, axis=0)


def main():
    ap = argparse.ArgumentParser(description="C1 randomized-weights control")
    ap.add_argument("--ref-layer-dir", required=True,
                    help="Trained layer dir to mirror (uids/sequences/META/k/expansion)")
    ap.add_argument("--model", default="esm2", choices=list(MODEL_NAMES))
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=0, help="0 = auto (~4 passes)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--weight-seed", type=int, default=0)
    ap.add_argument("--max-proteins", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out-layer-dir", required=True)
    args = ap.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    ref_dir = Path(args.ref_layer_dir)
    out_dir = Path(args.out_layer_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = _get_device(args.device)

    meta = json.loads((ref_dir / "META.json").read_text())
    embed_dim = meta["embed_dim"]
    k_sparse = meta.get("k_sparse", 256)
    expansion = meta.get("sae_hidden_dim", embed_dim * 8) // embed_dim
    k_aux = meta.get("k_aux", 64)
    dead_threshold = meta.get("dead_threshold", 1_000_000)
    val_uids = set(meta.get("val_uids", []))

    _, uids, lengths = load_layer(ref_dir)
    uids = [str(u) for u in uids]
    ref_seqs = load_ref_seqs(ref_dir)
    sequences = [ref_seqs[u] for u in uids]
    if args.max_proteins and args.max_proteins < len(uids):
        uids, sequences, lengths = uids[:args.max_proteins], sequences[:args.max_proteins], lengths[:args.max_proteins]

    print("=" * 70)
    print(f"  C1 RANDOM-WEIGHTS CONTROL — {args.model} layer {args.layer}")
    print("=" * 70)

    X = extract_random_esm2(sequences, args.layer, device,
                            model_name=MODEL_NAMES[args.model], seed=args.weight_seed)
    total = int(sum(int(l) for l in lengths))
    if X.shape[0] != total:
        raise RuntimeError(f"residue misalignment: X {X.shape[0]} != {total}")
    print(f"  Random embeddings: {X.shape}")

    # protein-level split (val = reference val_uids)
    offsets, off = [], 0
    for L in lengths:
        offsets.append(off); off += int(L)
    train_rows, val_rows = [], []
    for u, L, base in zip(uids, lengths, offsets):
        idx = np.arange(base, base + int(L))
        (val_rows if u in val_uids else train_rows).append(idx)
    train_rows = np.concatenate(train_rows) if train_rows else np.arange(total)
    val_rows = np.concatenate(val_rows) if val_rows else np.arange(0)

    X_train = np.ascontiguousarray(X[train_rows])
    norm_scale = compute_norm_scale(X_train)
    X_train = (X_train * norm_scale).astype(np.float32)
    epochs = args.epochs if args.epochs > 0 else max(4, int(4_000_000 / max(len(X_train), 1)))
    print(f"  Norm scale {norm_scale:.5f} | epochs {epochs} | train {X_train.shape[0]}")

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    sae = train_sae(X_train, input_dim=embed_dim, device=device, epochs=epochs,
                    lr=5e-5, expansion=expansion, k_sparse=k_sparse, k_aux=k_aux,
                    dead_threshold=dead_threshold)
    train_ev = float(compute_explained_variance(sae, X_train, device=device))
    if len(val_rows):
        val_ev = float(compute_explained_variance(sae, (X[val_rows] * norm_scale).astype(np.float32), device=device))
    else:
        val_ev = float("nan")
    print(f"  Train EV {train_ev:.4f} | Val EV {val_ev:.4f}")

    X_norm = (X * norm_scale).astype(np.float32)
    Z, Dd = extract_sae_features(sae, X_norm, device=device, save_dir=str(out_dir))
    print(f"  Z {Z.shape}, sparsity {(Z==0).mean()*100:.1f}%")

    np.save(out_dir / "lengths.npy", np.asarray([int(l) for l in lengths], dtype=np.int32))
    np.save(out_dir / "offsets.npy", np.asarray(offsets, dtype=np.int64))
    (out_dir / "sequences.json").write_text(json.dumps(sequences))
    (out_dir / "uids.json").write_text(json.dumps(uids))
    (out_dir / "META.json").write_text(json.dumps({
        "model": f"RANDOM-INIT {MODEL_NAMES[args.model]}",
        "layer": args.layer, "embed_dim": embed_dim,
        "sae_hidden_dim": embed_dim * expansion, "k_sparse": k_sparse, "k_aux": k_aux,
        "dead_threshold": dead_threshold, "num_proteins": len(uids),
        "total_residues": total, "norm_scale": norm_scale,
        "train_explained_variance": train_ev, "val_explained_variance": val_ev,
        "control": "randomized_weights", "weight_seed": args.weight_seed,
        "val_uids": sorted(val_uids & set(uids)),
    }, indent=2))
    print(f"\n  Control layer dir written: {out_dir}")
    print(f"  Next: run experiment_concept_f1.py and experiment_null_calibration.py on it.")


if __name__ == "__main__":
    main()
