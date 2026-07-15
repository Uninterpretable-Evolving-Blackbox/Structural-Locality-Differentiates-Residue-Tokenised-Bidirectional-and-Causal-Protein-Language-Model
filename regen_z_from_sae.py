#!/usr/bin/env python3
"""
regen_z_from_sae.py — rebuild Z.npy from sae_model.pt + PLM embeddings
========================================================================

Some robustness run dirs kept sae_model.pt / D.npy but lost Z.npy (iCloud
eviction). Concept-F1 needs Z.npy. This script:

  1. Loads (or extracts + caches) raw PLM embeddings for the layer.
  2. Loads the trained SAE checkpoint.
  3. Writes Z.npy (and refreshes D.npy) via extract_sae_features.

Does not retrain the SAE or touch unrelated layer artefacts.

Usage:
  python regen_z_from_sae.py --layer-dir outputs_layerwise_seed43/esm2/layer_16 \
    --model esm2 --layer 16
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import torch

from extract_embeddings import _get_device
from experiment_probe_baseline import get_raw_embeddings
from train_sae import extract_sae_features
from sae import SparseAutoencoder

warnings.filterwarnings("ignore")


def main():
    ap = argparse.ArgumentParser(description="Regenerate Z.npy from sae_model.pt")
    ap.add_argument("--layer-dir", required=True)
    ap.add_argument("--model", required=True, choices=["esm2", "rita"])
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--force", action="store_true",
                    help="Rebuild even if Z.npy already exists")
    args = ap.parse_args()

    layer_dir = Path(args.layer_dir)
    z_path = layer_dir / "Z.npy"
    if z_path.exists() and not args.force:
        size = z_path.stat().st_size
        if size > 1_000_000:
            print(f"  Z.npy already present ({size / 1e6:.0f} MB); skipping {layer_dir}")
            return

    meta_path = layer_dir / "META.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing META.json in {layer_dir}")
    if not (layer_dir / "sae_model.pt").exists():
        raise FileNotFoundError(f"Missing sae_model.pt in {layer_dir}")

    meta = json.loads(meta_path.read_text())
    uids = json.loads((layer_dir / "uids.json").read_text())
    sequences = json.loads((layer_dir / "sequences.json").read_text())
    lengths = np.load(layer_dir / "lengths.npy")
    total = int(sum(int(l) for l in lengths))

    device = _get_device(args.device)
    print(f"  Regenerating Z for {layer_dir} ({args.model} L{args.layer}) on {device}")

    raw = get_raw_embeddings(layer_dir, args.model, args.layer, uids, sequences, cache=True)
    if raw.shape[0] != total:
        raise RuntimeError(f"residue misalignment: raw {raw.shape[0]} != {total}")

    embed_dim = int(meta["embed_dim"])
    hidden_dim = int(meta["sae_hidden_dim"])
    expansion = hidden_dim // embed_dim
    norm_scale = float(meta.get("norm_scale", 1.0))
    X_norm = (raw * norm_scale).astype(np.float32)

    sae = SparseAutoencoder(
        input_dim=embed_dim,
        expansion=expansion,
        k_sparse=int(meta.get("k_sparse", 256)),
        k_aux=int(meta.get("k_aux", 64)),
        dead_threshold=int(meta.get("dead_threshold", 1_000_000)),
    )
    sae.load_state_dict(torch.load(layer_dir / "sae_model.pt", map_location="cpu"))
    sae = sae.to(device).float().eval()

    extract_sae_features(sae, X_norm, device=device, save_dir=str(layer_dir))
    print(f"  Done: {z_path} ({z_path.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
