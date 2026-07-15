#!/usr/bin/env python3
"""Correctness smoke test for adding ProtBert-BFD and ProGen2.

This does not train SAEs and does not inspect L_struct. It only checks:
  - offline model/tokenizer availability,
  - residue/token alignment,
  - hidden-state indexing at one mid-depth,
  - layer-grid pre-registration for the full 9-depth run.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer

from extract_embeddings import (
    extract_protbert_bfd_embeddings,
    extract_progen2_embeddings,
    verify_progen2_residue_tokenization,
)
from run_unsupervised import load_dataset_from_json


OUT = Path("results_new_plm_pair")
PROTBERT_MODEL = "Rostlab/prot_bert_bfd"
PROGEN2_BASE = os.environ.get("PROGEN2_BASE_MODEL_NAME", "hugohrban/progen2-base")
PROGEN2_FALLBACK = os.environ.get("PROGEN2_FALLBACK_MODEL_NAME", "hugohrban/progen2-small")


def layer_grid(n_blocks: int) -> list[int]:
    """Nine pre-specified relative depths over block indices 0..n_blocks-1."""
    if n_blocks < 2:
        raise ValueError(f"Need at least 2 blocks, got {n_blocks}")
    rel = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
    return [int(round(r * (n_blocks - 1))) for r in rel]


def n_blocks_from_config(model_name: str) -> int:
    try:
        cfg = AutoConfig.from_pretrained(
            model_name, local_files_only=True, trust_remote_code=True)
        cfg = cfg.to_dict()
    except (ValueError, KeyError):
        # Some repos (e.g. Rostlab/prot_bert_bfd) omit `model_type`, which breaks
        # AutoConfig dispatch. Fall back to reading the raw config.json.
        from huggingface_hub import hf_hub_download
        cfg_path = hf_hub_download(model_name, "config.json", local_files_only=True)
        cfg = json.loads(Path(cfg_path).read_text())
    for attr in ("num_hidden_layers", "n_layer", "num_layers"):
        value = cfg.get(attr) if isinstance(cfg, dict) else getattr(cfg, attr, None)
        if value is not None:
            return int(value)
    raise RuntimeError(f"Could not infer transformer block count for {model_name}")


def check_progen2_candidate(model_name: str) -> tuple[int, dict]:
    n_blocks = n_blocks_from_config(model_name)
    tok = AutoTokenizer.from_pretrained(model_name, local_files_only=True, trust_remote_code=True)
    report = verify_progen2_residue_tokenization(tok)
    if not report["is_residue_aligned"]:
        raise RuntimeError(
            f"{model_name} tokenizer is not residue-aligned for the canonical 20-AA probe")
    return n_blocks, report


def main():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    OUT.mkdir(parents=True, exist_ok=True)

    uids, seqs = load_dataset_from_json(Path("cache/sequences.json"))
    seqs = seqs[: int(os.environ.get("NEW_PAIR_SMOKE_PROTEINS", "8"))]
    device = os.environ.get("NEW_PAIR_SMOKE_DEVICE") or None

    report = {
        "purpose": "tokenization/alignment/hidden-state smoke only; no L_struct-based selection",
        "n_smoke_sequences": len(seqs),
        "protbert": {},
        "progen2": {},
    }

    try:
        protbert_blocks = n_blocks_from_config(PROTBERT_MODEL)
    except Exception as exc:
        report["status"] = "failed"
        report["failure_stage"] = "protbert_offline_config"
        report["protbert"].update({
            "model_name": PROTBERT_MODEL,
            "error": str(exc),
        })
        (OUT / "smoke_alignment_report.json").write_text(json.dumps(report, indent=2))
        raise

    protbert_grid = layer_grid(protbert_blocks)
    protbert_mid = protbert_grid[4]
    report["protbert"].update({
        "model_name": PROTBERT_MODEL,
        "n_blocks": protbert_blocks,
        "layer_grid": protbert_grid,
        "smoke_layer": protbert_mid,
        "tokenization": "space-separated uppercase amino acids; U/Z/O/B mapped to X; [CLS]/[SEP] removed",
    })
    protbert = extract_protbert_bfd_embeddings(
        seqs, layers=[protbert_mid], device=device, batch_size=2)
    report["protbert"]["smoke_shape"] = list(protbert[protbert_mid].shape)
    report["protbert"]["expected_residues"] = int(sum(len(s) for s in seqs))
    report["protbert"]["residue_aligned"] = (
        int(protbert[protbert_mid].shape[0]) == report["protbert"]["expected_residues"]
    )
    if not report["protbert"]["residue_aligned"]:
        raise RuntimeError("ProtBert-BFD smoke extraction is not residue-aligned")

    progen2_errors = []
    selected_progen2 = None
    for model_name in (PROGEN2_BASE, PROGEN2_FALLBACK):
        try:
            n_blocks, tok_report = check_progen2_candidate(model_name)
            selected_progen2 = (model_name, n_blocks, tok_report)
            break
        except Exception as exc:
            progen2_errors.append({"model_name": model_name, "error": str(exc)})

    if selected_progen2 is None:
        report["progen2"]["candidate_errors"] = progen2_errors
        (OUT / "smoke_alignment_report.json").write_text(json.dumps(report, indent=2))
        raise RuntimeError("No feasible residue-aligned ProGen2 candidate found offline")

    progen2_model, progen2_blocks, progen2_tok = selected_progen2
    progen2_grid = layer_grid(progen2_blocks)
    progen2_mid = progen2_grid[4]
    report["progen2"].update({
        "model_name": progen2_model,
        "candidate_errors_before_selection": progen2_errors,
        "n_blocks": progen2_blocks,
        "layer_grid": progen2_grid,
        "smoke_layer": progen2_mid,
        "tokenization_probe": progen2_tok,
    })
    progen2 = extract_progen2_embeddings(
        seqs, layers=[progen2_mid], device=device, batch_size=1, model_name=progen2_model)
    report["progen2"]["smoke_shape"] = list(progen2[progen2_mid].shape)
    report["progen2"]["expected_residues"] = int(sum(len(s) for s in seqs))
    report["progen2"]["residue_aligned"] = (
        int(progen2[progen2_mid].shape[0]) == report["progen2"]["expected_residues"]
    )
    if not report["progen2"]["residue_aligned"]:
        raise RuntimeError("ProGen2 smoke extraction is not residue-aligned")

    report["status"] = "passed"
    (OUT / "smoke_alignment_report.json").write_text(json.dumps(report, indent=2))
    (OUT / "new_pair_env.sh").write_text(
        "export PROTBERT_LAYERS=\"{}\"\n"
        "export PROGEN2_MODEL_NAME=\"{}\"\n"
        "export PROGEN2_LAYERS=\"{}\"\n".format(
            ",".join(map(str, protbert_grid)),
            progen2_model,
            ",".join(map(str, progen2_grid)),
        )
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
