#!/usr/bin/env python3
"""
experiment_prott5_densify_analysis.py — H3 (enc vs dec) on a densified grid.

Replaces the 5-point H4_enc_vs_dec analysis with a 9-point grid:
layers {0, 3, 6, 9, 12, 15, 18, 21, 23}.  Produces:

  analysis_results/comparison/H3_enc_vs_dec_dense.csv
  analysis_results/comparison/H3_enc_vs_dec_dense.png / .pdf
  analysis_results/comparison/H3_enc_vs_dec_dense_summary.txt

The plot is a line chart of d_struct vs relative depth with a vertical
crossover-line marking the zero-crossing (interpolated linearly between
the two adjacent probe points).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent

N_BLOCKS = 24  # ProtT5 has 24 transformer blocks → depths in [0, 23]
DEFAULT_LAYERS = [0, 3, 6, 9, 12, 15, 18, 21, 23]


def cohens_d(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    pooled = np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2.0 + 1e-12)
    return float((a.mean() - b.mean()) / pooled)


def compare_enc_dec(outputs_root: Path, layer: int) -> dict | None:
    enc_dir = outputs_root / "prott5_enc" / f"layer_{layer}"
    dec_dir = outputs_root / "prott5_dec" / f"layer_{layer}"
    ss_enc_p = enc_dir / "struct_seq_metrics.csv"
    ss_dec_p = dec_dir / "struct_seq_metrics.csv"
    if not (ss_enc_p.exists() and ss_dec_p.exists()):
        return None

    ss_enc = pd.read_csv(ss_enc_p)
    ss_dec = pd.read_csv(ss_dec_p)

    ve = ss_enc["struct_delta"].values
    vd = ss_dec["struct_delta"].values
    d_struct = cohens_d(ve, vd)
    # Two-sided p — we want to see direction and significance either way.
    _, p_struct = stats.mannwhitneyu(ve, vd, alternative="two-sided")

    ve_seq = ss_enc["seq_delta"].values
    vd_seq = ss_dec["seq_delta"].values
    d_seq = cohens_d(ve_seq, vd_seq)
    _, p_seq = stats.mannwhitneyu(ve_seq, vd_seq, alternative="two-sided")

    return dict(
        layer=layer,
        rel_depth=layer / (N_BLOCKS - 1),
        enc_struct_mean=float(ve.mean()), dec_struct_mean=float(vd.mean()),
        d_struct=d_struct, p_struct=float(p_struct),
        enc_seq_mean=float(ve_seq.mean()), dec_seq_mean=float(vd_seq.mean()),
        d_seq=d_seq, p_seq=float(p_seq),
        n_features=int(len(ve)),
    )


def find_sign_flip(layers: list[int], values: list[float]) -> tuple[int, int, float] | None:
    """Return (L_low, L_high, L_interp) where the sign flips, or None if none."""
    for i in range(len(layers) - 1):
        a, b = values[i], values[i + 1]
        if a == 0 or b == 0:
            continue
        if (a > 0) != (b > 0):
            # Linear interpolation of the zero-crossing in layer space
            L_low, L_high = layers[i], layers[i + 1]
            # x where value = 0 on linear interp between (L_low, a) and (L_high, b)
            t = a / (a - b)
            L_interp = L_low + t * (L_high - L_low)
            return (L_low, L_high, L_interp)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-dir", default="outputs_layerwise")
    ap.add_argument("--out", default="analysis_results/comparison")
    ap.add_argument("--layers", default=",".join(str(x) for x in DEFAULT_LAYERS),
                    help="comma-separated layer indices to probe (both enc and dec)")
    args = ap.parse_args()

    outputs_root = ROOT / args.outputs_dir if not Path(args.outputs_dir).is_absolute() else Path(args.outputs_dir)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    layers = [int(x) for x in args.layers.split(",")]

    rows = []
    for L in layers:
        r = compare_enc_dec(outputs_root, L)
        if r is None:
            print(f"  skip layer {L} (missing enc or dec struct_seq_metrics.csv)")
            continue
        rows.append(r)
    if not rows:
        raise SystemExit("no layers had both enc and dec data — aborting")

    df = pd.DataFrame(rows).sort_values("layer").reset_index(drop=True)
    csv_path = out_dir / "H3_enc_vs_dec_dense.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nWritten → {csv_path}")
    print(df[["layer", "rel_depth", "enc_struct_mean", "dec_struct_mean",
              "d_struct", "p_struct", "d_seq", "p_seq"]].to_string(
        index=False, float_format=lambda v: f"{v:+.4f}"))

    # --- Find crossover (d_struct sign flip) ---
    flip = find_sign_flip(df["layer"].tolist(), df["d_struct"].tolist())
    if flip is None:
        crossover_msg = ("No sign-flip in d_struct across the 9 probes — "
                         "encoder vs decoder structural ordering is monotone.")
        L_interp = None
    else:
        L_low, L_high, L_interp = flip
        relL = L_interp / (N_BLOCKS - 1) * 100
        crossover_msg = (
            f"d_struct flips sign between layer {L_low} and layer {L_high}; "
            f"linearly-interpolated zero-crossing at layer {L_interp:.2f} "
            f"(≈ {relL:.1f}% relative depth). "
            f"{'Crossover lies on a NEW probe point.' if L_interp in (3, 9, 15, 21) else 'Crossover lies BETWEEN existing probe points — new points narrow the localisation.'}"
        )

    # --- Write text summary ---
    summary_lines = [
        "ProtT5 encoder vs decoder — densified H3 (9 depths per side)\n",
        "=" * 72 + "\n\n",
        "Probed relative depths (layer / 23):\n",
        "  " + ", ".join(f"L{r['layer']} ({r['rel_depth']*100:.1f}%)"
                          for _, r in df.iterrows()) + "\n\n",
        "d_struct (enc struct_delta − dec struct_delta):\n",
    ]
    for _, r in df.iterrows():
        marker = "  (NEW)" if r["layer"] in (3, 9, 15, 21) else ""
        summary_lines.append(
            f"  L{int(r['layer']):<2} ({r['rel_depth']*100:5.1f}%): "
            f"d = {r['d_struct']:+.4f}, p = {r['p_struct']:.2e}{marker}\n")
    summary_lines.append("\n" + crossover_msg + "\n")

    (out_dir / "H3_enc_vs_dec_dense_summary.txt").write_text("".join(summary_lines))
    print("\n" + "".join(summary_lines[-6:]).strip())

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), sharex=True)
    x = df["rel_depth"].values * 100

    # Panel A — d_struct (enc vs dec)
    y = df["d_struct"].values
    ax1.axhline(0, color="grey", lw=0.8, ls=":")
    ax1.plot(x, y, "-", color="#444", lw=1.5, zorder=2)
    # enc-wins region (positive d) blue, dec-wins (negative d) orange
    enc_mask = y >= 0
    ax1.scatter(x[enc_mask], y[enc_mask], c="#1f77b4", s=70,
                label="enc wins (d_struct > 0)", zorder=3, edgecolor="white", lw=1)
    ax1.scatter(x[~enc_mask], y[~enc_mask], c="#ff7f0e", s=70,
                label="dec wins (d_struct < 0)", zorder=3, edgecolor="white", lw=1)
    # Highlight NEW points
    new_mask = df["layer"].isin([3, 9, 15, 21]).values
    ax1.scatter(x[new_mask], y[new_mask], s=180, facecolors="none",
                edgecolors="crimson", lw=2, zorder=4, label="new probe (densification)")
    # Crossover line
    if L_interp is not None:
        ax1.axvline(L_interp / (N_BLOCKS - 1) * 100, color="crimson",
                    ls="--", lw=1.4, alpha=0.8,
                    label=f"crossover ≈ {L_interp / (N_BLOCKS - 1) * 100:.1f}% depth")
    ax1.set_xlabel("Relative depth (%)")
    ax1.set_ylabel("Cohen's d (enc − dec, struct_delta)")
    ax1.set_title("Structural locality: encoder vs decoder")
    ax1.legend(loc="best", fontsize=8, framealpha=0.9)
    ax1.grid(alpha=0.3)

    # Panel B — raw means (enc blue, dec orange), for context
    ax2.plot(x, df["enc_struct_mean"].values, "o-", color="#1f77b4",
             lw=2, markersize=8, label="encoder")
    ax2.plot(x, df["dec_struct_mean"].values, "s--", color="#ff7f0e",
             lw=2, markersize=8, label="decoder")
    ax2.scatter(x[new_mask], df["enc_struct_mean"].values[new_mask],
                s=180, facecolors="none", edgecolors="crimson", lw=2, zorder=4)
    ax2.scatter(x[new_mask], df["dec_struct_mean"].values[new_mask],
                s=180, facecolors="none", edgecolors="crimson", lw=2, zorder=4)
    ax2.set_xlabel("Relative depth (%)")
    ax2.set_ylabel("Mean struct_delta")
    ax2.set_title("Per-model structural-locality means")
    ax2.legend(loc="best", fontsize=9)
    ax2.grid(alpha=0.3)

    fig.suptitle(f"ProtT5 encoder vs decoder — 9-layer densified grid "
                 f"({'no crossover' if L_interp is None else f'd_struct = 0 at ~{L_interp/(N_BLOCKS-1)*100:.1f}% depth'})",
                 fontsize=11)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"H3_enc_vs_dec_dense.{ext}", dpi=250, bbox_inches="tight")
    print(f"Plot    → {out_dir / 'H3_enc_vs_dec_dense.png'}")


if __name__ == "__main__":
    main()
