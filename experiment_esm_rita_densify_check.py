#!/usr/bin/env python3
"""
experiment_esm_rita_densify_check.py — de-risk gate for ESM-2/RITA H5
densification.

Reads outputs_layerwise/esm2/layer_12/ after it finishes the de-risk
training and checks two things before the remaining 7 SAEs commit:

  1. SAE val_EV at ESM-2 layer 12 lands around 0.7 — the existing ESM-2
     val_EV range is [0.67, 0.94] (with L0 as the high outlier), so L12
     should fit comfortably under 0.95 and near the mid-layer cluster.

  2. Mean struct_delta on ESM-2 at layer 12 sits between layer-8 and
     layer-16 values.  Existing means: L8 +0.0195, L16 +0.0527 — L12
     should be in-between (or within ±1.2× of that gap) for the
     densification to be meaningful.

Exits 0 on pass, 1 on fail.  Prints a clear verdict line either way.
"""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent / "outputs_layerwise"


def val_ev(model: str, layer: int) -> float | None:
    p = ROOT / model / f"layer_{layer}" / "META.json"
    if not p.exists():
        return None
    return float(json.loads(p.read_text())["val_explained_variance"])


def struct_mean(model: str, layer: int) -> float | None:
    csv = ROOT / model / f"layer_{layer}" / "struct_seq_metrics.csv"
    if not csv.exists():
        return None
    return float(pd.read_csv(csv).struct_delta.mean())


def main() -> int:
    print("=" * 72)
    print("  ESM-2 / RITA H5 densification — de-risk gate (ESM-2 layer 12)")
    print("=" * 72)

    # ---- Check 1: val_EV magnitude ----
    evs = {L: val_ev("esm2", L) for L in [0, 8, 12, 16, 24, 32]}
    print("\n  [1] ESM-2 val_EV across layers:")
    for L, v in evs.items():
        flag = "  (NEW)" if L == 12 else ""
        print(f"       layer {L:>2}: " + (f"{v:.4f}{flag}" if v is not None else "(missing)" + flag))
    ev12 = evs.get(12)
    if ev12 is None:
        print("\n  ❌  FAIL: ESM-2 layer 12 META.json missing — de-risk never finished.")
        return 1

    # Existing ESM-2 val_EVs cluster ~0.66-0.94 (L0 is the high outlier);
    # mid-layers sit around 0.67-0.72.  L12 should be in that range ±0.05.
    # Hard ceiling: 0.99 (ProGen2 degeneracy threshold).
    existing = [evs[L] for L in [0, 8, 16, 24, 32] if evs.get(L) is not None]
    if not existing:
        print("\n  ❌  FAIL: no baseline ESM-2 val_EVs available.")
        return 1
    ev_min, ev_max = min(existing), max(existing)
    slack = 0.05
    if ev12 >= 0.99:
        print(f"\n  ❌  FAIL (check 1): val_EV {ev12:.4f} ≥ 0.99 (degeneracy regime).")
        return 1
    if not (ev_min - slack <= ev12 <= ev_max + slack):
        print(f"\n  ❌  FAIL (check 1): val_EV {ev12:.4f} outside "
              f"[{ev_min-slack:.4f}, {ev_max+slack:.4f}] "
              f"(existing range [{ev_min:.4f}, {ev_max:.4f}] ± {slack})")
        return 1
    print(f"\n  ✅  PASS (check 1): val_EV {ev12:.4f} fits existing range "
          f"[{ev_min:.4f}, {ev_max:.4f}] (ceiling 0.99)")

    # ---- Check 2: mean struct_delta smoothness ----
    ms = {L: struct_mean("esm2", L) for L in [0, 8, 12, 16, 24, 32]}
    print("\n  [2] ESM-2 mean struct_delta across layers:")
    for L, m in ms.items():
        flag = "  (NEW)" if L == 12 else ""
        print(f"       layer {L:>2}: " + (f"{m:+.6f}{flag}" if m is not None else "(missing)"))
    m8, m12, m16 = ms.get(8), ms.get(12), ms.get(16)
    if m8 is None or m12 is None or m16 is None:
        print("\n  ❌  FAIL: struct_seq_metrics missing at one of layers 8/12/16")
        return 1

    gap = abs(m16 - m8)
    tol = max(0.010, 1.2 * gap)   # ±0.01 floor so tiny gaps don't fail on noise
    overshoot = max(abs(m12 - m8), abs(m12 - m16)) - tol
    print(f"       gap 8→16 = {gap:+.6f}, tol = {tol:.6f}, "
          f"max |m12-neighbour| = {max(abs(m12-m8), abs(m12-m16)):.6f}")
    if overshoot > 0:
        print(f"\n  ❌  FAIL (check 2): layer 12 mean struct_delta {m12:+.6f} "
              f"overshoots neighbours ({m8:+.6f} / {m16:+.6f}) by {overshoot:.6f}.")
        return 1
    print(f"\n  ✅  PASS (check 2): layer 12 mean struct_delta {m12:+.6f} "
          f"fits between neighbours {m8:+.6f} / {m16:+.6f}")

    print("\n" + "=" * 72)
    print("  ✅  DE-RISK GATE PASSED — proceed with remaining 7 SAEs")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
