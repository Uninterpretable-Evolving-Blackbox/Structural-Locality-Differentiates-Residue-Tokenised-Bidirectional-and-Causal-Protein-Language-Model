#!/usr/bin/env python3
"""Within-model L_struct trajectory plot with bootstrap CI bands.
Reads v2_cis_trajectory.csv produced by compute_cis_v2.py.
Output: paper_draft/paper_submission-3/figures/within_model_trajectories.{pdf,png}
"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"]  = 42

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "paper_draft" / "paper_submission-4" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(ROOT / "outputs_robustness" / "v2_cis_trajectory.csv")
df["depth_pct"] = df.rel_depth.str.rstrip('%').astype(int)

COLORS = {"esm2": "#1f77b4", "rita": "#ff7f0e",
          "prott5_enc": "#2ca02c", "prott5_dec": "#d62728"}
LABELS = {"esm2": "ESM-2", "rita": "RITA",
          "prott5_enc": "ProtT5 enc", "prott5_dec": "ProtT5 dec"}

fig, ax = plt.subplots(figsize=(3.5, 2.6))
for model in ["esm2", "rita", "prott5_enc", "prott5_dec"]:
    sub = df[df.model == model].sort_values("depth_pct")
    if not len(sub): continue
    x = sub.depth_pct.values
    y = sub.mean_point.values
    ylo = sub.ci_lo.values; yhi = sub.ci_hi.values
    ax.fill_between(x, ylo, yhi, color=COLORS[model], alpha=0.18, linewidth=0)
    ax.plot(x, y, "-o", color=COLORS[model], label=LABELS[model],
            markersize=3, linewidth=1.2)

ax.axhline(0, color="grey", linestyle=":", linewidth=0.6)
ax.set_xlabel("Relative depth (%)")
ax.set_ylabel(r"mean $L_\mathrm{struct}$")
ax.legend(fontsize=7, loc="best", frameon=True, framealpha=0.9, handlelength=1.2)
ax.tick_params(labelsize=8); ax.xaxis.label.set_size(9); ax.yaxis.label.set_size(9)
ax.grid(axis="y", alpha=0.25)

fig.tight_layout(pad=0.4)
for ext in ("pdf", "png"):
    out = OUT / f"within_model_trajectories.{ext}"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"  wrote {out}")

# Sanity print
print("\nWithin-model trajectory summary:")
for model in ["esm2", "rita", "prott5_enc", "prott5_dec"]:
    sub = df[df.model == model].sort_values("depth_pct")
    if not len(sub): continue
    means = sub.mean_point.values
    print(f"  {model:12s}: range=[{means.min():+.4f}, {means.max():+.4f}]  "
          f"argmax depth={sub.iloc[means.argmax()].rel_depth}")
