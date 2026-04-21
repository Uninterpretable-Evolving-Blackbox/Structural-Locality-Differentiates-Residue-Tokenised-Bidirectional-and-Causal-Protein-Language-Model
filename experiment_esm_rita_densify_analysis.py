#!/usr/bin/env python3
"""
experiment_esm_rita_densify_analysis.py — within-model H5 analysis on a
densified 9-depth grid for ESM-2 and RITA.

Replaces the v5 per-feature Spearman-over-5-depths approach with aggregate
mean(struct_delta) ± bootstrap 95% CI across features, reported per layer.

For each model:
  - Load struct_seq_metrics.csv at each probed layer.
  - mean = mean of struct_delta across features at that layer.
  - CI via 1000 bootstrap resamples (with replacement, n = n_features).
  - Write per-layer row to CSV with point estimate and 95% CI.

Produces:
  analysis_results/comparison/H5_within_model_dense.csv
  analysis_results/comparison/H5_within_model_dense.png / .pdf
  analysis_results/comparison/H5_within_model_dense_summary.txt
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent

# (model, n_blocks, default 9-layer grid) — n_blocks used for relative-depth
# percentages (layer / (n_blocks-1) * 100).
MODELS = {
    "esm2": {
        "n_blocks": 33,
        "layers":   [0, 4, 8, 12, 16, 20, 24, 28, 32],
        "colour":   "#1f77b4",
        "label":    "ESM-2",
        "new":      {4, 12, 20, 28},
    },
    "rita": {
        "n_blocks": 24,
        "layers":   [0, 3, 6, 9, 12, 15, 18, 21, 23],
        "colour":   "#ff7f0e",
        "label":    "RITA",
        "new":      {3, 9, 15, 21},
    },
}


def bootstrap_ci(x: np.ndarray, n_boot: int = 1000, ci: float = 0.95,
                 rng: np.random.Generator | None = None) -> tuple[float, float, float]:
    """Return (mean, lower, upper) — bootstrap CI of the mean."""
    if rng is None:
        rng = np.random.default_rng(42)
    n = len(x)
    means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[i] = x[idx].mean()
    lo = float(np.quantile(means, (1 - ci) / 2))
    hi = float(np.quantile(means, 1 - (1 - ci) / 2))
    return float(x.mean()), lo, hi


def analyse_model(model: str, cfg: dict, outputs_root: Path,
                  n_boot: int) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(42)
    n_blocks = cfg["n_blocks"]
    for L in cfg["layers"]:
        csv_p = outputs_root / model / f"layer_{L}" / "struct_seq_metrics.csv"
        if not csv_p.exists():
            print(f"  skip {model}/layer_{L} (missing struct_seq_metrics.csv)")
            continue
        sd = pd.read_csv(csv_p).struct_delta.values
        mean, lo, hi = bootstrap_ci(sd, n_boot=n_boot, rng=rng)
        rows.append(dict(
            model=model,
            layer=L,
            rel_depth=L / (n_blocks - 1),
            n_features=int(len(sd)),
            mean_struct_delta=mean,
            ci_lo=lo,
            ci_hi=hi,
            is_new=L in cfg["new"],
        ))
    return pd.DataFrame(rows)


def describe_trend(df_model: pd.DataFrame, label: str) -> list[str]:
    """Return a short text summary of depth trend for one model."""
    if df_model.empty:
        return [f"{label}: no data.\n"]
    # Pearson correlation between relative depth and mean_struct_delta across
    # the 9 points — just a descriptive trend check (not a statistical claim).
    x = df_model["rel_depth"].values
    y = df_model["mean_struct_delta"].values
    if len(x) >= 3:
        corr = float(np.corrcoef(x, y)[0, 1])
    else:
        corr = float("nan")
    peak_i = int(np.argmax(y)); trough_i = int(np.argmin(y))
    lines = [
        f"\n{label}:\n",
        f"  9-depth rel-depth ↔ mean(struct_delta) correlation: r = {corr:+.3f}\n",
        f"  peak:   layer {int(df_model.iloc[peak_i]['layer'])} "
        f"({df_model.iloc[peak_i]['rel_depth']*100:5.1f}% depth)  "
        f"mean {y[peak_i]:+.4f}  [CI {df_model.iloc[peak_i]['ci_lo']:+.4f} .. "
        f"{df_model.iloc[peak_i]['ci_hi']:+.4f}]\n",
        f"  trough: layer {int(df_model.iloc[trough_i]['layer'])} "
        f"({df_model.iloc[trough_i]['rel_depth']*100:5.1f}% depth)  "
        f"mean {y[trough_i]:+.4f}  [CI {df_model.iloc[trough_i]['ci_lo']:+.4f} .. "
        f"{df_model.iloc[trough_i]['ci_hi']:+.4f}]\n",
    ]
    # Interpretive note
    if corr < -0.3:
        lines.append(f"  Direction: monotonic decline (negative correlation).\n")
    elif corr > 0.3:
        lines.append(f"  Direction: monotonic increase (positive correlation).\n")
    else:
        lines.append(f"  Direction: no monotonic trend (|r| < 0.3).\n")
    # Early-bias / late-bias note
    early_mean = float(df_model[df_model["rel_depth"] < 0.3]["mean_struct_delta"].mean())
    late_mean  = float(df_model[df_model["rel_depth"] > 0.7]["mean_struct_delta"].mean())
    lines.append(
        f"  Early layers (<30% depth) mean: {early_mean:+.4f}  |  "
        f"Late layers (>70% depth) mean: {late_mean:+.4f}  |  "
        f"Early − Late = {early_mean - late_mean:+.4f}\n"
    )
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-dir", default="outputs_layerwise")
    ap.add_argument("--out", default="analysis_results/comparison")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    outputs_root = ROOT / args.outputs_dir if not Path(args.outputs_dir).is_absolute() else Path(args.outputs_dir)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for model, cfg in MODELS.items():
        print(f"\n=== Analysing {model} ===")
        df = analyse_model(model, cfg, outputs_root, args.n_boot)
        if df.empty:
            print(f"  ⚠  no data for {model}")
            continue
        print(df[["layer", "rel_depth", "n_features", "mean_struct_delta",
                  "ci_lo", "ci_hi", "is_new"]].to_string(index=False,
                  float_format=lambda v: f"{v:+.4f}"))
        all_rows.append(df)
    if not all_rows:
        raise SystemExit("No data for any model — aborting.")

    big = pd.concat(all_rows, ignore_index=True)
    csv_out = out_dir / "H5_within_model_dense.csv"
    big.to_csv(csv_out, index=False)
    print(f"\nWritten → {csv_out}")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for model, cfg in MODELS.items():
        d = big[big.model == model].sort_values("layer").reset_index(drop=True)
        if d.empty:
            continue
        x = d["rel_depth"].values * 100
        y = d["mean_struct_delta"].values
        lo = d["ci_lo"].values; hi = d["ci_hi"].values
        ax.fill_between(x, lo, hi, color=cfg["colour"], alpha=0.2)
        ax.plot(x, y, "o-", color=cfg["colour"], lw=2, markersize=6,
                label=cfg["label"], zorder=3)
        # Ring new probe points
        new_mask = d["is_new"].values
        if new_mask.any():
            ax.scatter(x[new_mask], y[new_mask], s=140, facecolors="none",
                       edgecolors="crimson", lw=2, zorder=5)
    # Legend entry for "new probe"
    ax.scatter([], [], s=140, facecolors="none", edgecolors="crimson", lw=2,
               label="new probe (densification)")
    ax.axhline(0, color="grey", lw=0.8, ls=":")
    ax.set_xlabel("Relative depth (%)")
    ax.set_ylabel("Mean struct_delta  (95% bootstrap CI across features)")
    ax.set_title("Within-model structural-locality depth trend — densified 9-point grid")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"H5_within_model_dense.{ext}", dpi=250, bbox_inches="tight")
    print(f"Plot    → {out_dir / 'H5_within_model_dense.png'}")

    # --- Summary ---
    lines = ["ESM-2 + RITA within-model depth trend — densified 9-point grid\n",
             "=" * 72 + "\n"]
    lines.append(f"\nBootstrap resamples per layer: {args.n_boot}\n")
    lines.append(f"95% CI reported in brackets.\n")
    for model, cfg in MODELS.items():
        d = big[big.model == model].sort_values("layer")
        lines.extend(describe_trend(d, cfg["label"]))
    # ESM-2 front-loading check: is L0 > all other layers, or at least early > late?
    e = big[big.model == "esm2"].sort_values("layer").set_index("layer")
    if len(e) >= 5:
        l0 = e.loc[0, "mean_struct_delta"] if 0 in e.index else float("nan")
        others = e[e.index != 0]["mean_struct_delta"]
        lines.append("\n")
        lines.append("ESM-2 front-loading check (v5 claim):\n")
        lines.append(f"  L0 mean struct_delta = {l0:+.4f}\n")
        lines.append(f"  Max of other 8 depths = {others.max():+.4f}"
                     f"  (at layer {int(others.idxmax())})\n")
        lines.append(f"  L0 > max(others)? {'YES — front-loading preserved' if l0 > others.max() else 'NO — front-loading not preserved'}\n")
    # RITA trend call
    r = big[big.model == "rita"].sort_values("layer")
    if len(r) >= 5:
        x = r["rel_depth"].values; y = r["mean_struct_delta"].values
        corr = float(np.corrcoef(x, y)[0, 1])
        lines.append("\n")
        lines.append("RITA depth-trend call:\n")
        lines.append(f"  Pearson r over 9 points = {corr:+.3f}\n")
        if abs(corr) < 0.3:
            lines.append("  → No systematic depth trend (|r| < 0.3).\n")
        else:
            direction = "increase" if corr > 0 else "decline"
            lines.append(f"  → Monotonic {direction} in structural locality with depth.\n")

    summary_path = out_dir / "H5_within_model_dense_summary.txt"
    summary_path.write_text("".join(lines))
    print("\n" + "".join(lines[-10:]).strip())
    print(f"\nSummary → {summary_path}")


if __name__ == "__main__":
    main()
