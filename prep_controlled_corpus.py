#!/usr/bin/env python3
"""
prep_controlled_corpus.py — build ONE shared tokenized corpus for the controlled
MLM-vs-CLM experiment. Both objectives train on THIS exact corpus (same sequences,
same order) — only the objective differs. This is what isolates the data.

- streams ConvergeBio/uniref50 (cc-by-4.0) from HuggingFace (no full download)
- keeps standard-20-AA sequences, length >= min_len, truncates to context-2
- holds out sequences that EXACTLY match a SCOPe eval domain (pilot-light holdout;
  use mmseqs2 for the publishable homology holdout)
- tokenizes with a clean residue vocab (1 token = 1 residue)
- writes flat uint8 tokens + lengths + offsets (mirrors the existing pipeline layout)

Usage:
  python prep_controlled_corpus.py --smoke            # 2k seqs, quick end-to-end test
  python prep_controlled_corpus.py --n-sequences 3000000
"""
import argparse
import json
from pathlib import Path
import numpy as np

AA = "ACDEFGHIKLMNPQRSTVWY"
PAD, BOS, EOS, MASK, UNK = 0, 1, 2, 3, 4
VOCAB = {"[PAD]": PAD, "[BOS]": BOS, "[EOS]": EOS, "[MASK]": MASK, "[UNK]": UNK}
for _i, _a in enumerate(AA):
    VOCAB[_a] = 5 + _i
VOCAB_SIZE = len(VOCAB)          # 25
AA_SET = set(AA)
AA2ID = {a: 5 + i for i, a in enumerate(AA)}


def load_scope_seqs(fasta):
    seqs = set()
    if not Path(fasta).exists():
        print(f"WARNING: SCOPe fasta {fasta} not found — holdout will be empty")
        return seqs
    cur = []
    with open(fasta) as f:
        for line in f:
            if line.startswith(">"):
                if cur:
                    seqs.add("".join(cur)); cur = []
            else:
                cur.append(line.strip().upper())
    if cur:
        seqs.add("".join(cur))
    return seqs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sequences", type=int, default=3_000_000)
    ap.add_argument("--context", type=int, default=512)
    ap.add_argument("--min-len", type=int, default=16)
    ap.add_argument("--hf-dataset", default="ConvergeBio/uniref50")
    ap.add_argument("--split", default="train")
    ap.add_argument("--scope-fasta", default="cache/scope_40.fa")
    ap.add_argument("--out-dir", default=str(Path.home() / "own_sae_data" / "uniref50_pilot"))
    ap.add_argument("--shuffle-buffer", type=int, default=50000,
                    help="streaming shuffle buffer for a representative sample (0=off)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.n_sequences = 2000
        args.shuffle_buffer = 5000
        args.out_dir = str(Path.home() / "own_sae_data" / "uniref50_smoke")

    from datasets import load_dataset

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    maxres = args.context - 2

    scope = load_scope_seqs(args.scope_fasta)
    print(f"SCOPe holdout sequences: {len(scope)}")

    ds = load_dataset(args.hf_dataset, split=args.split, streaming=True)
    if args.shuffle_buffer > 0:
        ds = ds.shuffle(seed=42, buffer_size=args.shuffle_buffer)
        print(f"streaming shuffle buffer: {args.shuffle_buffer}")

    tokens, lengths = [], []
    kept = seen = dropped_aa = dropped_len = dropped_holdout = 0
    seq_col = None
    for rec in ds:
        seen += 1
        if seq_col is None:
            for c in ("sequence", "Sequence", "text", "seq", "Seq"):
                if c in rec:
                    seq_col = c; break
            if seq_col is None:
                seq_col = [k for k, v in rec.items() if isinstance(v, str)][0]
            print(f"using sequence column: '{seq_col}'  (record keys: {list(rec.keys())})")
        s = str(rec[seq_col]).strip().upper()
        if len(s) > maxres:
            s = s[:maxres]
        if len(s) < args.min_len:
            dropped_len += 1; continue
        if not set(s) <= AA_SET:
            dropped_aa += 1; continue
        if s in scope:
            dropped_holdout += 1; continue
        ids = [BOS] + [AA2ID[a] for a in s] + [EOS]
        tokens.append(np.array(ids, dtype=np.uint8))
        lengths.append(len(ids))
        kept += 1
        if kept >= args.n_sequences:
            break
        if kept % 200000 == 0:
            print(f"  kept {kept} / seen {seen}")

    flat = np.concatenate(tokens)
    lengths = np.array(lengths, dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
    np.save(out / "tokens.npy", flat)
    np.save(out / "lengths.npy", lengths)
    np.save(out / "offsets.npy", offsets)
    meta = dict(vocab=VOCAB, vocab_size=VOCAB_SIZE, context=args.context,
                n_sequences=int(kept), n_tokens=int(flat.shape[0]),
                hf_dataset=args.hf_dataset, seq_col=seq_col,
                dropped=dict(aa=dropped_aa, length=dropped_len, holdout=dropped_holdout),
                pad=PAD, bos=BOS, eos=EOS, mask=MASK, unk=UNK, aa=AA)
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nDONE: kept={kept} seqs | {flat.shape[0]} tokens (~{flat.shape[0]/1e6:.1f}M) | "
          f"mean_len={lengths.mean():.1f}")
    print(f"dropped: non-std-aa={dropped_aa}  too-short={dropped_len}  scope-holdout={dropped_holdout}")
    print(f"wrote -> {out}")


if __name__ == "__main__":
    main()
