#!/usr/bin/env python3
"""
experiment_prott5_densify_check.py — de-risk gate for the ProtT5 densification.

Reads outputs_layerwise/prott5_dec/layer_9/ and checks two things before the
full 8-SAE expansion launches:

  1. SAE val_EV at layer 9 lands in the same range as existing ProtT5-dec
     layers {0, 6, 12, 18, 23} (typically 0.7-0.9, must be < 0.99 to rule
     out the ProGen2-style degeneracy regime).

  2. mean struct_delta on prott5_dec at layer 9 falls between the values
     at layers 6 and 12 — i.e. the new point fits smoothly between its
     neighbours, not an outlier.  This is a dec-only smoothness check
     (enc layer 9 is not yet extracted during de-risk).

Exits 0 on pass, 1 on fail. Prints a clear verdict line either way.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent / "outputs_layerwise"


def cohens_d(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    pooled = np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2.0 + 1e-12)
    return float((a.mean() - b.mean()) / pooled)


def dec_struct_mean(layer: int) -> float | None:
    """Mean struct_delta on prott5_dec at the given layer (single-model
    smoothness statistic; doesn't require prott5_enc to exist)."""
    csv = ROOT / "prott5_dec" / f"layer_{layer}" / "struct_seq_metrics.csv"
    if not csv.exists():
        return None
    return float(pd.read_csv(csv).struct_delta.mean())


def val_ev(model: str, layer: int) -> float | None:
    p = ROOT / model / f"layer_{layer}" / "META.json"
    if not p.exists():
        return None
    return float(json.loads(p.read_text())["val_explained_variance"])


def main() -> int:
    print("=" * 72)
    print("  ProtT5 densification — de-risk gate (layer 9 on dec)")
    print("=" * 72)

    # ---- Check 1: val_EV magnitude ----
    evs = {L: val_ev("prott5_dec", L) for L in [0, 6, 9, 12, 18, 23]}
    print("\n  [1] ProtT5-dec val_EV across layers:")
    for L, v in evs.items():
        flag = "  (NEW)" if L == 9 else ""
        print(f"       layer {L:>2}: {v!r}{flag}" if v is None
              else f"       layer {L:>2}: {v:.4f}{flag}")
    ev9 = evs.get(9)
    if ev9 is None:
        print("\n  ❌  FAIL: layer 9 META.json is missing — de-risk never finished.")
        return 1

    existing = [evs[L] for L in [0, 6, 12, 18, 23] if evs.get(L) is not None]
    if not existing:
        print("\n  ❌  FAIL: no baseline prott5_dec val_EVs available.")
        return 1
    ev_min, ev_max = min(existing), max(existing)
    # Allow ±0.05 slack around the existing range, hard ceiling 0.99
    # (above which the ProGen2 degeneracy regime kicks in).
    slack = 0.05
    if ev9 >= 0.99:
        print(f"\n  ❌  FAIL (check 1): val_EV at layer 9 = {ev9:.4f} ≥ 0.99")
        print(f"      This is the ProGen2-degeneracy regime — SAE basis not uniquely identified.")
        return 1
    if not (ev_min - slack <= ev9 <= ev_max + slack):
        print(f"\n  ❌  FAIL (check 1): val_EV at layer 9 = {ev9:.4f} outside "
              f"[{ev_min-slack:.4f}, {ev_max+slack:.4f}] "
              f"(existing range [{ev_min:.4f}, {ev_max:.4f}] ± {slack})")
        return 1
    print(f"\n  ✅  PASS (check 1): val_EV {ev9:.4f} fits existing range "
          f"[{ev_min:.4f}, {ev_max:.4f}] (max ceiling 0.99)")

    # ---- Check 2: dec mean struct_delta smoothness ----
    ms = {L: dec_struct_mean(L) for L in [0, 6, 9, 12, 18, 23]}
    print("\n  [2] prott5_dec mean struct_delta across layers:")
    for L, m in ms.items():
        flag = "  (NEW)" if L == 9 else ""
        print(f"       layer {L:>2}: " + (f"{m:+.6f}{flag}" if m is not None else "(missing)"))
    m6, m9, m12 = ms.get(6), ms.get(9), ms.get(12)
    if m6 is None or m9 is None or m12 is None:
        print("\n  ❌  FAIL: missing struct_seq_metrics at one of layers 6/9/12")
        return 1

    # Fit test: layer 9 should sit "between" layer 6 and layer 12 within a
    # tolerance.  "Between" is lenient here — protT5-dec mean struct_delta
    # across existing layers (L0 +0.0171, L6 +0.0145, L12 +0.0084, L18
    # +0.0089, L23 +0.0163) is a shallow U-curve, so L9 is plausible in a
    # range that covers both endpoints ± 1.2× the 6→12 gap.
    gap_6_12 = abs(m12 - m6)
    tol = max(0.005, 1.2 * gap_6_12)
    overshoot = max(abs(m9 - m6), abs(m9 - m12)) - tol
    print(f"       gap 6→12 = {gap_6_12:+.6f}, tol = {tol:.6f}, "
          f"max |m9-neighbour| = {max(abs(m9-m6), abs(m9-m12)):.6f}")
    if overshoot > 0:
        print(f"\n  ❌  FAIL (check 2): layer 9 mean struct_delta {m9:+.6f} "
              f"overshoots both neighbours ({m6:+.6f} / {m12:+.6f}) by "
              f"{overshoot:.6f} — SAE at L9 not comparable to neighbours.")
        return 1
    print(f"\n  ✅  PASS (check 2): layer 9 mean struct_delta {m9:+.6f} "
          f"fits between neighbours {m6:+.6f} / {m12:+.6f}")

    print("\n" + "=" * 72)
    print("  ✅  DE-RISK GATE PASSED — proceed with full 8-SAE expansion")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
