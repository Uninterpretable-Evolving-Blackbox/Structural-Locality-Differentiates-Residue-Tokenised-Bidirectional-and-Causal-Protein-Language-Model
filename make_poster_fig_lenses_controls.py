#!/usr/bin/env python3
"""
make_poster_fig_lenses_controls.py — poster figures for Section 4.

Fig 1a (lenses agree): concept-F1 vs relative depth, ESM-2 vs RITA.
Fig 1b (lens independence): 4x4 cross-metric Spearman heatmap (ESM-2 L16).
Fig 2 (controls hold): trained vs random concept-F1 at ESM-2 L16 with symmetric
seed dots (SAE seeds 42/43/44 vs PLM weight-init seeds 0/1/2).

Reads results_concept_f1/, results_interp_comparison/, results_concept_f1_multiseed_headline/.
Writes paper_draft/paper_submission-4/figures/poster_{concept_f1_depth,lens_independence,controls}.{pdf,png}.
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "paper_draft" / "paper_submission-4" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ESM_COLOUR = "#1f77b4"
RITA_COLOUR = "#ff7f0e"

# (label, esm_layer, rita_layer)
MATCHED = [
    ("0", 0, 0), ("13", 4, 3), ("25", 8, 6), ("38", 12, 9), ("50", 16, 12),
    ("63", 20, 15), ("75", 24, 18), ("88", 28, 21), ("100", 32, 23),
]


def concept_f1(model: str, layer: int) -> float:
    p = ROOT / "results_concept_f1" / f"{model}_l{layer}" / "summary.json"
    return float(json.loads(p.read_text())["mean_top_test_f1_per_concept"])


def _save(fig, stem: str):
    fig.tight_layout(pad=0.6)
    for ext in ("pdf", "png"):
        out = OUT_DIR / f"{stem}.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"  wrote {out}")
    plt.close(fig)


def fig_concept_f1_depth():
    labels = [m[0] for m in MATCHED]
    x = np.arange(len(labels))
    esm = [concept_f1("esm2", e) for _, e, _ in MATCHED]
    rita = [concept_f1("rita", r) for _, _, r in MATCHED]

    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    ax.plot(x, esm, "o-", color=ESM_COLOUR, lw=2, ms=5, label="ESM-2")
    ax.plot(x, rita, "s-", color=RITA_COLOUR, lw=2, ms=5, label="RITA")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Relative depth (%)")
    ax.set_ylabel("Concept-F1 (mean top, test split)")
    ax.set_ylim(0, 0.85)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower center", fontsize=9, frameon=True, framealpha=0.9)
    ax.set_title("Concept-F1 agrees with $L_\\mathrm{struct}$", fontsize=10, loc="left")
    _save(fig, "poster_concept_f1_depth")


def fig_lens_independence():
    sp = json.loads((ROOT / "results_interp_comparison" / "esm2_l16" /
                     "summary.json").read_text())["cross_metric_spearman"]
    metrics = ["|spearman_rsa|", "|spearman_coord|", "concept_valF1", "struct_delta"]
    short = ["RSA", "Coord", "F1", "Lstruct"]
    M = np.array([[sp[a][b] for b in metrics] for a in metrics])

    fig, ax = plt.subplots(figsize=(3.8, 3.4))
    im = ax.imshow(M, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(4))
    ax.set_xticklabels(short, fontsize=9, rotation=45, ha="right")
    ax.set_yticks(range(4))
    ax.set_yticklabels(short, fontsize=9)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    fontsize=9, color="black" if M[i, j] < 0.6 else "white")
    ax.set_title("Lenses near-independent ($\\rho$)", fontsize=10, loc="left")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save(fig, "poster_lens_independence")


def _load_multiseed_controls():
    p = ROOT / "results_concept_f1_multiseed_headline" / "summary.json"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} missing — run summarize_concept_f1_multiseed_headline.py first"
        )
    s = json.loads(p.read_text())
    trained = np.array(s["trained_esm2_l16_sae_seeds"]["values"], dtype=float)
    rand = np.array(s["random_esm2_l16_weight_seeds"]["values"], dtype=float)
    return trained, rand


def fig_controls():
    trained, rand = _load_multiseed_controls()
    trained_mean, rand_mean = trained.mean(), rand.mean()

    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    ax.bar([0], [trained_mean], width=0.55, color=ESM_COLOUR, alpha=0.85,
           edgecolor="black", lw=0.5, label="Trained ESM-2 (mean)")
    ax.bar([1], [rand_mean], width=0.55, color="grey", alpha=0.7,
           edgecolor="black", lw=0.5, label="Random weights (mean)")
    ax.plot(np.full(len(trained), 0.0) + np.linspace(-0.12, 0.12, len(trained)),
            trained, "o", color="black", ms=5, zorder=5, label="SAE seeds (42,43,44)")
    ax.plot(np.full(len(rand), 1.0) + np.linspace(-0.12, 0.12, len(rand)),
            rand, "o", color="black", ms=5, zorder=5, label="weight seeds (0,1,2)")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Trained", "Random\nweights"])
    ax.set_ylabel("Structural concept-F1 (ESM-2 L16)")
    ax.set_ylim(0, 0.85)
    ax.grid(axis="y", alpha=0.25)
    ax.annotate(f"{trained_mean:.2f}", (0, trained_mean),
                ha="center", va="bottom", fontsize=9)
    ax.annotate(f"{rand_mean:.2f}", (1, rand_mean),
                ha="center", va="bottom", fontsize=9)
    ax.legend(loc="upper right", fontsize=7, frameon=True, framealpha=0.9)

    fig.tight_layout(pad=0.5)
    for ext in ("pdf", "png"):
        out = OUT_DIR / f"poster_controls.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"  wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    fig_concept_f1_depth()
    fig_lens_independence()
    fig_controls()
