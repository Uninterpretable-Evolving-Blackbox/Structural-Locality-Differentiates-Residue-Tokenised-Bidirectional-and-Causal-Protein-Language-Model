#!/usr/bin/env python3
"""
make_poster_fig_h1_sideways.py — landscape (side-by-side) variant of the H1
locality figure for the poster.

Identical data/styling to make_paper_fig_h1.py, but lays the two panels out
in a single row (L_struct | L_seq) instead of stacked, giving a wide figure
that fits a poster column. The paper figure (h1_locality_dist.*) is left
untouched.

Writes paper_draft/paper_submission-4/figures/h1_locality_dist_sideways.{pdf,png}.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ICML font compliance — must be set BEFORE any figure creation.
plt.rcParams["pdf.fonttype"] = 42     # TrueType in PDF (avoids Type-3)
plt.rcParams["ps.fonttype"]  = 42     # TrueType in PS/EPS (avoids Type-3)

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "paper_draft" / "paper_submission-4" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# RITA-side relative-depth labels (matches paper Table 1).
MATCHED = [
    ("0",    0,   0),
    ("13",   4,   3),
    ("25",   8,   6),
    ("38",   12,  9),
    ("50",   16, 12),
    ("63",   20, 15),
    ("75",   24, 18),
    ("88",   28, 21),
    ("100",  32, 23),
]

ESM_COLOUR  = "#1f77b4"
RITA_COLOUR = "#ff7f0e"


def load_column(model: str, layer: int, col: str) -> np.ndarray:
    p = ROOT / "outputs_layerwise" / model / f"layer_{layer}" / "struct_seq_metrics.csv"
    return pd.read_csv(p)[col].to_numpy()


def paired_violin(ax, depths_label, esm_data, rita_data, ylabel):
    """Paired violin: one pair per depth. Robust to outliers via percentile clip."""
    n = len(depths_label)

    def clip(arr):
        lo, hi = np.percentile(arr, [1, 99])
        return arr[(arr >= lo) & (arr <= hi)]
    esm_clipped  = [clip(a) for a in esm_data]
    rita_clipped = [clip(a) for a in rita_data]

    pos_esm  = np.arange(n) - 0.21
    pos_rita = np.arange(n) + 0.21

    v1 = ax.violinplot(esm_clipped,  positions=pos_esm,  widths=0.38,
                       showmeans=False, showmedians=False, showextrema=False)
    v2 = ax.violinplot(rita_clipped, positions=pos_rita, widths=0.38,
                       showmeans=False, showmedians=False, showextrema=False)
    for b in v1["bodies"]:
        b.set_facecolor(ESM_COLOUR); b.set_alpha(0.75); b.set_edgecolor("black"); b.set_linewidth(0.4)
    for b in v2["bodies"]:
        b.set_facecolor(RITA_COLOUR); b.set_alpha(0.75); b.set_edgecolor("black"); b.set_linewidth(0.4)

    for i, a in enumerate(esm_data):
        q25, q50, q75 = np.percentile(a, [25, 50, 75])
        ax.plot([pos_esm[i], pos_esm[i]], [q25, q75], color="black", lw=1.0)
        ax.plot(pos_esm[i], q50, "o", color="white", markersize=3.0, mec="black", mew=0.6, zorder=5)
    for i, a in enumerate(rita_data):
        q25, q50, q75 = np.percentile(a, [25, 50, 75])
        ax.plot([pos_rita[i], pos_rita[i]], [q25, q75], color="black", lw=1.0)
        ax.plot(pos_rita[i], q50, "o", color="white", markersize=3.0, mec="black", mew=0.6, zorder=5)

    ax.axhline(0, color="grey", lw=0.6, ls=":", zorder=1)
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(depths_label)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Relative depth (%)")
    ax.grid(axis="y", alpha=0.25, zorder=0)


def main():
    depth_labels = [d[0] for d in MATCHED]
    esm_struct   = [load_column("esm2", e, "struct_delta") for _, e, _ in MATCHED]
    rita_struct  = [load_column("rita", r, "struct_delta") for _, _, r in MATCHED]
    esm_seq      = [load_column("esm2", e, "seq_delta")    for _, e, _ in MATCHED]
    rita_seq     = [load_column("rita", r, "seq_delta")    for _, _, r in MATCHED]

    # Landscape: 1 row x 2 columns. Wider than tall for a poster column.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.6),
                                   gridspec_kw={"wspace": 0.22})

    paired_violin(ax1, depth_labels, esm_struct, rita_struct, r"$L_\mathrm{struct}$")
    paired_violin(ax2, depth_labels, esm_seq,    rita_seq,    r"$L_\mathrm{seq}$")

    # Panel labels A / B at the upper-left of each panel.
    ax1.text(-0.12, 1.02, "A", transform=ax1.transAxes,
             fontsize=13, fontweight="bold", va="bottom")
    ax2.text(-0.12, 1.02, "B", transform=ax2.transAxes,
             fontsize=13, fontweight="bold", va="bottom")

    from matplotlib.patches import Patch
    handles = [Patch(facecolor=ESM_COLOUR, alpha=0.75, edgecolor="black", label="ESM-2"),
               Patch(facecolor=RITA_COLOUR, alpha=0.75, edgecolor="black", label="RITA")]
    ax1.legend(handles=handles, loc="upper right", fontsize=9, frameon=True,
               framealpha=0.9, handlelength=1.2, handleheight=1.0)

    for ax in (ax1, ax2):
        ax.tick_params(labelsize=9)
        ax.tick_params(axis="x", which="major", pad=1)
        ax.yaxis.label.set_size(11)
        ax.xaxis.label.set_size(10)

    fig.tight_layout(pad=0.5)
    for ext in ("pdf", "png"):
        out = OUT_DIR / f"h1_locality_dist_sideways.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
