#!/usr/bin/env python3
"""
smoke_test_progen2.py — verify ProGen2-medium integration.

Checks the full extractor path (patches + tokenizer + forward + hook-free
output_hidden_states + keep_mask) on 3 short sequences.  Finishes in a
few seconds after first-run download (~3 GB cached thereafter).
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


def die(msg):
    print(f"  ❌  {msg}")
    sys.exit(2)


def ok(msg):
    print(f"  ✅  {msg}")


def main():
    print("=" * 60)
    print("  ProGen2 smoke test")
    print("=" * 60)

    print("\n[1] imports")
    try:
        from extract_embeddings import extract_progen2_embeddings
        from transformers import AutoConfig
    except Exception as e:
        die(f"import failed: {e}")
    ok("extract_embeddings.extract_progen2_embeddings present")

    print("\n[2] config probe")
    model_name = "hugohrban/progen2-medium"
    try:
        cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    except Exception as e:
        die(f"config load failed: {e}")
    n_blocks = (getattr(cfg, "num_hidden_layers", None)
                or getattr(cfg, "n_layer", None))
    hidden = (getattr(cfg, "hidden_size", None)
              or getattr(cfg, "n_embd", None) or "?")
    print(f"     n_blocks={n_blocks}, hidden={hidden}, vocab={cfg.vocab_size}")
    if n_blocks == 27:
        ok("27 blocks confirmed (matches [0, 7, 14, 20, 26] plan)")
    else:
        print(f"     ⚠️  expected 27 blocks, got {n_blocks}")

    print("\n[3] end-to-end extraction")
    test_seqs = [
        "MKVLWALLIVAALLLRSEEGARADLK",     # 26 residues
        "ACDEFGHIKLMNPQRSTVWY",           # 20 residues
        "MKKLPIRKP" + "ALA" * 10,         # 39 residues
    ]
    expected_total = sum(len(s) for s in test_seqs)
    print(f"     3 seqs, total {expected_total} residues")

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cpu"
    )
    t0 = time.time()
    try:
        emb = extract_progen2_embeddings(
            test_seqs, layers=[0, 13, 26],
            device=device, batch_size=3,
        )
    except Exception as e:
        die(f"extraction failed: {type(e).__name__}: {e}")
    print(f"     extraction took {time.time()-t0:.1f}s on {device}")

    if set(emb.keys()) != {0, 13, 26}:
        die(f"expected layers {{0, 13, 26}} got {set(emb.keys())}")
    ok("all 3 requested layers returned")

    for L, arr in emb.items():
        if arr.shape[0] != expected_total:
            die(f"layer {L}: got {arr.shape[0]} rows, expected {expected_total}")
        if arr.ndim != 2:
            die(f"layer {L}: expected 2D got {arr.ndim}D")
        if not np.isfinite(arr).all():
            die(f"layer {L}: non-finite values")
    ok(f"shapes OK, all finite; hidden_dim={emb[0].shape[1]}")

    summary = {
        "model_name": model_name,
        "device": device,
        "n_blocks": int(n_blocks),
        "hidden_dim": int(emb[0].shape[1]),
        "residues_total": int(expected_total),
        "layers_returned": sorted(emb.keys()),
        "tokenizer_1to1": True,
    }
    print("\n" + "=" * 60)
    print("  PROGEN2 SMOKE PASSED")
    print("=" * 60)
    print(json.dumps(summary, indent=2))
    Path("smoke_progen2_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
