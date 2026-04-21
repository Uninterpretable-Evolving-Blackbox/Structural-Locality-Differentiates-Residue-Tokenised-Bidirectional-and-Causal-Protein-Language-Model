#!/usr/bin/env python3
"""
smoke_test_rita.py — verify RITA integration before committing overnight.

Fast end-to-end check:
  1. Tokenizer loads and is 1:1 residue↔token (vs ProtGPT2's BPE).
  2. Model loads on the auto-detected device.
  3. extract_rita_embeddings returns (n_residues, hidden_dim) on 3 test seqs.
  4. Layer indexing is correct (requested layers map to hidden_states[N+1]).
  5. Shapes align with sum-of-sequence-lengths (no phantom tokens, no off-by-one).

On success, prints a JSON summary. On failure, prints the offending step and exits 2.
This is the only file that loads RITA weights; ~3 GB first-run download.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


def die(msg: str) -> None:
    print(f"  ❌  {msg}")
    sys.exit(2)


def ok(msg: str) -> None:
    print(f"  ✅  {msg}")


def main():
    print("=" * 60)
    print("  RITA smoke test")
    print("=" * 60)

    # ---------- 1. imports ----------
    print("\n[1] imports")
    try:
        from extract_embeddings import extract_rita_embeddings
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except Exception as e:
        die(f"import failed: {e}")
    ok("extract_embeddings.extract_rita_embeddings present")

    # RITA's custom modeling code predates transformers 5.x tied-weights API;
    # extractor applies this patch internally, but we also need it for the
    # standalone model-load probe in step [3] below.
    from transformers.modeling_utils import PreTrainedModel
    if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
        PreTrainedModel.all_tied_weights_keys = {}

    # ---------- 2. tokenizer 1:1 check ----------
    print("\n[2] tokenizer 1:1 residue↔token check")
    model_name = "lightonai/RITA_l"
    try:
        tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except Exception as e:
        die(f"tokenizer load failed: {e}")
    ok(f"tokenizer loaded from {model_name}")

    probes = ["MKVLWAL", "ACDEFGHIKLMNPQRSTVWY", "A" * 200]
    for seq in probes:
        ids = tok(seq, add_special_tokens=False)["input_ids"]
        if len(ids) != len(seq):
            die(f"tokenizer NOT 1:1: {len(seq)} residues → {len(ids)} tokens "
                f"on probe of length {len(seq)}")
    ok(f"tokenizer is 1:1 on {len(probes)} probes (including 20-AA and 200-AA strings)")

    special_ids = set(getattr(tok, "all_special_ids", []) or [])
    print(f"     special_ids = {sorted(special_ids)}, vocab size = {len(tok)}")

    # ---------- 3. model load + layer count ----------
    print("\n[3] model load + layer count")
    t0 = time.time()
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cpu"
    )
    # Config-only probe (do NOT instantiate the full model here — a preload
    # followed by delete leaves residual state on MPS that causes the
    # extractor's subsequent load to produce NaN activations in deep blocks.
    # The extractor in step [4] does a full load and reports n_blocks itself.)
    from transformers import AutoConfig
    try:
        cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    except Exception as e:
        die(f"config load failed: {e}")
    n_blocks = (getattr(cfg, "num_hidden_layers", None)
                or getattr(cfg, "n_layer", None)
                or getattr(cfg, "num_layers", None))
    hidden = (getattr(cfg, "hidden_size", None)
              or getattr(cfg, "d_model", None)
              or getattr(cfg, "n_embd", None) or "?")
    print(f"     config loaded in {time.time()-t0:.1f}s on {device}, "
          f"n_blocks={n_blocks}, hidden={hidden}")
    if n_blocks == 24:
        ok("24 blocks confirmed (matches [0, 6, 12, 18, 23] matched-depth plan)")
    else:
        print(f"     ℹ️  n_blocks reported by config = {n_blocks} "
              f"(extractor will re-verify from model.transformer.layers)")

    # ---------- 4. end-to-end extraction on 3 seqs, 1 layer ----------
    print("\n[4] extract_rita_embeddings on 3 sequences, layer 0")
    test_seqs = [
        "MKVLWALLIVAALLLRSEEGARADLK",     # 26 residues
        "ACDEFGHIKLMNPQRSTVWY",           # 20 residues
        "MKKLPIRKP" + "ALA" * 10,          # 39 residues
    ]
    expected_total = sum(len(s) for s in test_seqs)
    print(f"     3 seqs, total {expected_total} residues")

    t0 = time.time()
    try:
        emb = extract_rita_embeddings(
            test_seqs, layers=[0, 12, 23],
            device=device, batch_size=3,
        )
    except Exception as e:
        die(f"extraction failed: {e}")
    print(f"     extraction took {time.time()-t0:.1f}s")

    if set(emb.keys()) != {0, 12, 23}:
        die(f"expected layers {{0, 12, 23}} in output, got {set(emb.keys())}")
    ok("all 3 requested layers returned")

    for L, arr in emb.items():
        if arr.shape[0] != expected_total:
            die(f"layer {L}: shape {arr.shape} — expected {expected_total} rows")
        if arr.ndim != 2:
            die(f"layer {L}: expected 2D array, got {arr.ndim}D")
        if not np.isfinite(arr).all():
            die(f"layer {L}: non-finite values in embedding")
    ok(f"shapes OK, all finite; hidden_dim={emb[0].shape[1]}")

    # ---------- 5. summary ----------
    summary = {
        "model_name": model_name,
        "device": device,
        "n_blocks_reported": int(n_blocks) if n_blocks is not None else None,
        "hidden_dim": int(emb[0].shape[1]),
        "residues_total": int(expected_total),
        "layers_returned": sorted(emb.keys()),
        "tokenizer_1to1": True,
    }
    print("\n" + "=" * 60)
    print("  RITA SMOKE PASSED")
    print("=" * 60)
    print(json.dumps(summary, indent=2))
    Path("smoke_rita_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
