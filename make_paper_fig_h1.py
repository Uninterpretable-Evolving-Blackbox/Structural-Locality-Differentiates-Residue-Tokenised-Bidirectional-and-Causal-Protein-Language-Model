#!/usr/bin/env python3
"""
make_paper_fig_h1.py — Fig. 1 (H1): distribution of L_struct (top) and
L_seq (bottom) across SAE features for ESM-2 (blue) and RITA (orange)
at the nine matched relative depths.

Writes paper_draft/paper_submission-3/figures/h1_locality_dist.{pdf,png}.

ICML submissions disallow Type-3 fonts in embedded figures (matplotlib's
default emits Type-3 DejaVuSans). We force TrueType (Type-42) for PDF and
PS output via rcParams below, which is accepted by the ICML font check.
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
OUT_DIR = ROOT / "paper_draft" / "paper_submission-3" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# x-axis labels are paper-canonical relative-depth %s (RITA-side denominator:
# RITA has 24 blocks → L/23; ESM-2 has 33 blocks → L/32). The paper's Table 1
# uses "13%*" for the L4/L3 pair (13.0% RITA-side), not the 12.5% ESM-2-side
# value — so we label with the RITA %s to match.
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
    # Percentile-clip for visual: use full range for stats but cap the violin tails
    # at [1%, 99%] so a handful of extreme outliers don't squash the bulk.
    def clip(arr):
        lo, hi = np.percentile(arr, [1, 99])
        return arr[(arr >= lo) & (arr <= hi)]
    esm_clipped  = [clip(a) for a in esm_data]
    rita_clipped = [clip(a) for a in rita_data]

    # Narrower violins + smaller offset at 9 depths so 18 violins fit comfortably
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

    # Overlay median + IQR as thin black line
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
    ax.grid(axis="y", alpha=0.25, zorder=0)


def main():
    # ── Collect data ──
    depth_labels = [d[0] for d in MATCHED]
    esm_struct   = [load_column("esm2", e, "struct_delta") for _, e, _ in MATCHED]
    rita_struct  = [load_column("rita", r, "struct_delta") for _, _, r in MATCHED]
    esm_seq      = [load_column("esm2", e, "seq_delta")    for _, e, _ in MATCHED]
    rita_seq     = [load_column("rita", r, "seq_delta")    for _, _, r in MATCHED]

    # ── Figure sized for ICML single column (~3.25 in wide); 9-depth grid
    #     needs a touch more width/height than the 5-depth version ──
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5, 4.3), sharex=True,
                                   gridspec_kw={"hspace": 0.12})

    # Y-axis labels match the paper's locality-score notation L_struct / L_seq.
    # matplotlib's mathtext doesn't support \text; \mathrm renders identically
    # to LaTeX's $L_{\text{struct}}$ / $L_{\text{seq}}$.
    paired_violin(ax1, depth_labels, esm_struct, rita_struct, r"$L_\mathrm{struct}$")
    paired_violin(ax2, depth_labels, esm_seq,    rita_seq,    r"$L_\mathrm{seq}$")
    ax2.set_xlabel("Relative depth (%)")
    # Tighter tick labels at 9 depths
    for ax in (ax1, ax2):
        ax.tick_params(axis="x", which="major", pad=1)

    # Single shared legend
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=ESM_COLOUR, alpha=0.75, edgecolor="black", label="ESM-2"),
               Patch(facecolor=RITA_COLOUR, alpha=0.75, edgecolor="black", label="RITA")]
    ax1.legend(handles=handles, loc="upper right", fontsize=8, frameon=True,
               framealpha=0.9, handlelength=1.2, handleheight=1.0)

    # Smaller tick / label font to fit ICML column
    for ax in (ax1, ax2):
        ax.tick_params(labelsize=8)
        ax.yaxis.label.set_size(9)
    ax2.xaxis.label.set_size(9)

    fig.tight_layout(pad=0.4)
    for ext in ("pdf", "png"):
        out = OUT_DIR / f"h1_locality_dist.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"  wrote {out}")

    # ── Sanity: print per-depth summary stats so we can verify plot matches data ──
    print("\nSanity summary (median and mean per depth):")
    for i, (lab, _, _) in enumerate(MATCHED):
        print(f"  depth {lab:>4}:"
              f"  ESM d_struct med={np.median(esm_struct[i]):+.4f} mean={esm_struct[i].mean():+.4f}"
              f"  |  RITA med={np.median(rita_struct[i]):+.4f} mean={rita_struct[i].mean():+.4f}"
              f"  |  ESM d_seq med={np.median(esm_seq[i]):+.4f}"
              f"  |  RITA d_seq med={np.median(rita_seq[i]):+.4f}")


if __name__ == "__main__":
    main()
