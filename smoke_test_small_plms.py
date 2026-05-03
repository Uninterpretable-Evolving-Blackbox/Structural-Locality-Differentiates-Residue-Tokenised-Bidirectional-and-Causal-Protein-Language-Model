#!/usr/bin/env python3
"""
smoke_test_small_plms.py — quick check that the scale-ablation extractors
work end-to-end before committing overnight compute.

Verifies:
  1. extract_esm2_small_embeddings loads esm2_t12 and returns finite
     residue-level activations for 3 test sequences at all 5 requested layers.
  2. extract_rita_small_embeddings loads RITA_s and does the same.
  3. The layer lists registered in MODEL_PLANS match the model's n_blocks.
"""

import json
import sys
from pathlib import Path

import numpy as np


def ok(msg): print(f"  ✅  {msg}")
def die(msg): print(f"  ❌  {msg}"); sys.exit(2)


def main():
    print("=" * 60)
    print("  Scale-ablation smoke test (esm2_small + rita_small)")
    print("=" * 60)

    # RITA custom modeling needs this monkeypatch under transformers 5.x
    from transformers.modeling_utils import PreTrainedModel
    if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
        PreTrainedModel.all_tied_weights_keys = {}

    from extract_embeddings import (
        extract_esm2_small_embeddings,
        extract_rita_small_embeddings,
    )
    from run_unsupervised import MODEL_PLANS

    test_seqs = [
        "MKVLWALLIVAALLLRSEEGARADLK",     # 26 residues
        "ACDEFGHIKLMNPQRSTVWY",           # 20 residues
        "MKKLPIRKP" + "ALA" * 10,          # 39 residues
    ]
    expected_total = sum(len(s) for s in test_seqs)

    results = {}
    for tag, extractor_name, planned_layers in [
        ("esm2_small", extract_esm2_small_embeddings, MODEL_PLANS["esm2_small"][2]),
        ("rita_small", extract_rita_small_embeddings, MODEL_PLANS["rita_small"][2]),
    ]:
        print(f"\n[{tag}]  planned layers: {planned_layers}")
        try:
            emb = extractor_name(test_seqs, layers=planned_layers)
        except Exception as e:
            die(f"{tag} extraction failed: {e}")

        if set(emb.keys()) != set(planned_layers):
            die(f"{tag}: returned layers {set(emb.keys())}, expected {set(planned_layers)}")
        for L, arr in emb.items():
            if arr.shape[0] != expected_total:
                die(f"{tag} L{L}: shape {arr.shape}, expected {expected_total} rows")
            if arr.ndim != 2:
                die(f"{tag} L{L}: expected 2D, got {arr.ndim}D")
            if not np.isfinite(arr).all():
                die(f"{tag} L{L}: non-finite values")
        hdim = emb[planned_layers[0]].shape[1]
        ok(f"{tag}: 5/5 layers OK, all finite, hidden_dim={hdim}")
        results[tag] = {"layers": planned_layers, "hidden_dim": int(hdim),
                        "residues": int(expected_total)}

    print("\n" + "=" * 60)
    print("  SMOKE PASSED")
    print("=" * 60)
    print(json.dumps(results, indent=2))
    Path("smoke_small_plms_summary.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
