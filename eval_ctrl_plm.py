#!/usr/bin/env python3
"""
eval_ctrl_plm.py — turn a trained controlled PLM checkpoint into a layer dir that
the existing L_struct (cpu_stage.py) and concept-F1 (experiment_concept_f1.py)
pipelines can consume, unchanged.

Steps (mirrors run_unsupervised.py exactly, just with our custom PLM as the extractor):
  1. load the checkpoint -> PLM
  2. reuse the SAME eval proteins as ESM-2/RITA (uids + sequences from a ref layer dir)
  3. extract per-residue hidden states at a chosen block index
  4. protein-split train/val (reuse the ref layer dir's val_uids for an identical split)
  5. Bricken norm_scale -> train_sae (expansion 8, k=256, k_aux=64) -> extract Z
  6. write Z.npy / D.npy / sae_model.pt / META.json / lengths / offsets / sequences / uids

Then run:  cpu_stage.py --layer-dir <out>   and   experiment_concept_f1.py --layer-dir <out>

Runs on CPU by default so it does NOT contend with an MPS training job in progress.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch

from model_ctrl_plm import PLM, PLMConfig
from cpu_stage import load_ref_seqs
from train_sae import compute_norm_scale, train_sae, extract_sae_features

AA2ID = {a: 5 + i for i, a in enumerate("ACDEFGHIKLMNPQRSTVWY")}
BOS, EOS = 1, 2


def tokenize(seq):
    return [BOS] + [AA2ID.get(a, 4) for a in seq.upper()] + [EOS]


@torch.no_grad()
def extract_layer(model, uids, seqs, layer, device, batch_size=16, max_len=512):
    """Return X (sum_res, D) of per-residue hidden states at block `layer`, plus token lengths."""
    model.eval()
    feats, lengths = [], []
    order = list(range(len(uids)))
    for i in range(0, len(order), batch_size):
        chunk = order[i:i + batch_size]
        toks = [tokenize(seqs[j])[:max_len] for j in chunk]
        T = max(len(t) for t in toks)
        ids = np.full((len(toks), T), 0, dtype=np.int64)
        am = np.zeros((len(toks), T), dtype=np.int64)
        for r, t in enumerate(toks):
            ids[r, :len(t)] = t
            am[r, :len(t)] = 1
        ids = torch.from_numpy(ids).to(device)
        am_t = torch.from_numpy(am).to(device)
        _, hid = model(ids, am_t, return_hidden=True)
        h = hid[layer].float().cpu().numpy()   # (B,T,D)
        for r, t in enumerate(toks):
            L = len(t) - 2                       # residues between BOS/EOS
            feats.append(h[r, 1:1 + L, :])
            lengths.append(L)
    return np.concatenate(feats, axis=0).astype(np.float32), np.array(lengths, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--name", required=True, help="e.g. ctrl_mlm / ctrl_clm")
    ap.add_argument("--ref-layer-dir", default="outputs_layerwise/esm2/layer_16",
                    help="reuse this dir's uids/sequences/val split (same eval set as ESM-2/RITA)")
    ap.add_argument("--layer", type=int, required=True, help="block index (0..n_layers-1)")
    ap.add_argument("--out-root", default="outputs_ctrl")
    ap.add_argument("--max-proteins", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--sae-device", default="cpu")
    ap.add_argument("--sae-epochs", type=int, default=60)
    ap.add_argument("--expansion", type=int, default=8)
    ap.add_argument("--k-sparse", type=int, default=256)
    args = ap.parse_args()

    ref = Path(args.ref_layer_dir)
    uids = json.loads((ref / "uids.json").read_text())
    ref_seqs = load_ref_seqs(ref)                       # uid -> sequence
    uids = [str(u) for u in uids if str(u) in ref_seqs]
    val_uids = set()
    meta_ref = json.loads((ref / "META.json").read_text())
    val_uids = set(str(u) for u in meta_ref.get("val_uids", []))
    if args.max_proteins and args.max_proteins < len(uids):
        uids = uids[:args.max_proteins]
    seqs = [ref_seqs[u] for u in uids]
    print(f"eval proteins: {len(uids)}  (val held-out: {len(val_uids & set(uids))})")

    ck = torch.load(args.ckpt, map_location="cpu")
    cfg = PLMConfig(**ck["cfg"])
    model = PLM(cfg)
    model.load_state_dict(ck["model"])
    model = model.to(args.device)
    print(f"loaded {args.name} ckpt step={ck.get('step')} tokens={ck.get('tokens',0)/1e6:.0f}M "
          f"| block {args.layer}/{cfg.n_layers} | embed_dim {cfg.d_model}")

    X, lengths = extract_layer(model, uids, seqs, args.layer, args.device)
    D = X.shape[1]
    print(f"extracted X {X.shape}")

    offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
    is_val = np.array([u in val_uids for u in uids], dtype=bool)
    tr_rows = np.concatenate([np.arange(offsets[i], offsets[i] + lengths[i])
                              for i in range(len(uids)) if not is_val[i]]) if (~is_val).any() else np.arange(X.shape[0])
    X_train = X[tr_rows]

    norm_scale = compute_norm_scale(X_train)
    print(f"norm_scale {norm_scale:.6f}")
    sae = train_sae((X_train * norm_scale).astype(np.float32), input_dim=D,
                    device=args.sae_device, epochs=args.sae_epochs,
                    expansion=args.expansion, k_sparse=args.k_sparse, k_aux=64)

    out = Path(args.out_root) / args.name / f"layer_{args.layer}"
    out.mkdir(parents=True, exist_ok=True)
    Z, _ = extract_sae_features(sae, (X * norm_scale).astype(np.float32),
                                device=args.sae_device, save_dir=str(out))
    # save TRUNCATED sequences that match the extracted residues (proteins > 510 res
    # were cut to fit the 512 context) so cpu_stage's sum(len(seq))==Z_rows check holds
    seqs_trunc = [s[:int(lengths[i])] for i, s in enumerate(seqs)]
    np.save(out / "Z.npy", Z.astype(np.float16))
    np.save(out / "lengths.npy", lengths.astype(np.int32))
    np.save(out / "offsets.npy", offsets)
    (out / "sequences.json").write_text(json.dumps(seqs_trunc))
    (out / "uids.json").write_text(json.dumps(uids))
    torch.save(sae.state_dict(), out / "sae_model.pt")
    (out / "META.json").write_text(json.dumps({
        "model": args.name, "layer": args.layer, "embed_dim": D,
        "sae_hidden_dim": D * args.expansion, "k_sparse": args.k_sparse,
        "norm_scale": norm_scale, "val_uids": sorted(val_uids & set(uids)),
        "ckpt": args.ckpt, "ckpt_tokens": int(ck.get("tokens", 0)),
    }, indent=2))
    print(f"Z {Z.shape} sparsity {(Z==0).mean()*100:.1f}%  ->  {out}")


if __name__ == "__main__":
    main()
