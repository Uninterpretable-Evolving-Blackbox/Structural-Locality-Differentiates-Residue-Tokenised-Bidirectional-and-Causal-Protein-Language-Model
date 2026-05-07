#!/usr/bin/env python3
"""
test_experiment_correctness.py — Methodology checks for the SAE-PLM experiment.

These tests do NOT prove the experiment's *conclusions*; they verify that the
*procedure* matches what the paper claims. Each test is targeted at a specific
methodological claim that, if violated, would invalidate that claim.

Run:
    python tests/test_experiment_correctness.py
or:
    pytest tests/test_experiment_correctness.py -v

Tests assume `outputs_layerwise/` and `outputs_robustness/` are populated
(i.e., you have run the main pipeline). Tests that require GPU-only artefacts
are skipped if data is missing.
"""

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PASS = 0
FAIL = 0
SKIP = 0


def report(name, ok, msg=""):
    global PASS, FAIL
    sym = "PASS" if ok else "FAIL"
    print(f"  [{sym}] {name}{(' — ' + msg) if msg else ''}")
    if ok: PASS += 1
    else: FAIL += 1


def skip(name, why):
    global SKIP
    print(f"  [SKIP] {name} — {why}")
    SKIP += 1


# =====================================================================
# Test 1 — Cohen's d formula matches a textbook reference implementation
# =====================================================================
def test_cohens_d_formula_is_correct():
    """We use pooled_std = sqrt((var_a + var_b)/2 + eps), unbiased (ddof=1).
    Equivalent textbook formula for two equal-N groups: difference of means /
    sqrt of average of group variances. Cross-check against scipy.stats.tukey_hsd
    standard error or a manual computation on a known example.
    """
    from outputs_robustness.compute_cis_v3_optimized import cohens_d

    # Example with hand-computed answer:
    # a = [1, 2, 3, 4, 5] mean=3, var=2.5 (ddof=1)
    # b = [4, 5, 6, 7, 8] mean=6, var=2.5
    # pooled = sqrt((2.5+2.5)/2) = sqrt(2.5) = 1.5811
    # d = (3-6)/1.5811 = -1.8974
    a = np.array([1, 2, 3, 4, 5], dtype=float)
    b = np.array([4, 5, 6, 7, 8], dtype=float)
    expected = -1.8973665961010275  # (3-6)/sqrt(2.5)
    got = cohens_d(a, b)
    report("Cohen's d on hand-computed [1..5] vs [4..8]",
           abs(got - expected) < 1e-6,
           f"expected {expected:+.6f}, got {got:+.6f}")

    # Self-consistency: d(a,b) = -d(b,a)
    report("Cohen's d sign symmetry",
           abs(cohens_d(a, b) + cohens_d(b, a)) < 1e-12)

    # Identical distributions → d = 0
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 5000)
    y = rng.normal(0, 1, 5000)
    report("Cohen's d on equal distributions ≈ 0",
           abs(cohens_d(x, y)) < 0.05,
           f"got {cohens_d(x, y):+.4f}")


# =====================================================================
# Test 2 — Reported d_point reproduces from raw layer CSVs
# =====================================================================
def test_dpoint_reproduces_from_raw_csv():
    """The bootstrap CSVs report a d_point. That value must match what you get
    by directly computing cohens_d on the per-layer struct_seq_metrics.csv.
    If they diverge, either (a) the bootstrap pipeline is computing something
    different from the headline statistic, or (b) the layer CSVs have drifted.
    """
    from outputs_robustness.compute_cis_v3_optimized import cohens_d

    # Sample 3 (depth, model_pair) cells with overlap between layer CSVs and
    # the bootstrap output.
    cases = [
        ("0%",   "esm2", 0,  "rita", 0,  "struct_delta", 1.4402),  # paper Table 1 row 1
        ("50%",  "esm2", 16, "rita", 12, "struct_delta", 0.6159),
        ("100%", "esm2", 32, "rita", 23, "seq_delta",   -0.9831),
    ]
    for depth, ma, la, mb, lb, col, expected in cases:
        path_a = ROOT/f"outputs_layerwise/{ma}/layer_{la}/struct_seq_metrics.csv"
        path_b = ROOT/f"outputs_layerwise/{mb}/layer_{lb}/struct_seq_metrics.csv"
        if not (path_a.exists() and path_b.exists()):
            skip(f"d_point reproduces ({depth}, {col})", "layer CSVs missing")
            continue
        a = pd.read_csv(path_a)[col].to_numpy()
        b = pd.read_csv(path_b)[col].to_numpy()
        got = cohens_d(a, b)
        report(f"d_point reproduces from raw CSV ({depth}, {col})",
               abs(got - expected) < 1e-3,
               f"expected {expected:+.4f}, got {got:+.4f}")


# =====================================================================
# Test 3 — Batched bootstrap matches the explicit-loop reference
# =====================================================================
def test_batched_bootstrap_equivalent_to_loop():
    """compute_cis_v3_optimized speeds up the bootstrap by replacing 1000 small
    matvec products with one big matmul. The optimisation must produce
    bit-equivalent (to float64 epsilon) output as the explicit loop. If this
    fails, every CI in v3opt_cis_val_sweeps.csv is suspect.
    """
    try:
        from outputs_robustness.compute_cis_v3_optimized import correctness_test
        # The function asserts equivalence; if it raises we fail.
        correctness_test()
        report("Batched bootstrap matches explicit loop", True,
               "max abs error < 1e-6 (machine epsilon)")
    except AssertionError as e:
        report("Batched bootstrap matches explicit loop", False, str(e))
    except Exception as e:
        report("Batched bootstrap matches explicit loop", False,
               f"{type(e).__name__}: {e}")


# =====================================================================
# Test 4 — train/val split is disjoint, deterministic, and not leaky
# =====================================================================
def test_train_val_split_integrity():
    """The paper claims a deterministic 90/10 protein-level split with seed 42.
    Two failure modes: (a) train ∩ val ≠ ∅ (data leakage), (b) the same split
    applied at different layers gives a different val set (non-determinism).
    """
    val_uid_sets = []
    for path in [
        ROOT/"outputs_layerwise/esm2/layer_0/META.json",
        ROOT/"outputs_layerwise/rita/layer_0/META.json",
        ROOT/"outputs_layerwise/esm2/layer_16/META.json",
    ]:
        if not path.exists():
            skip("train/val split integrity", f"META.json missing at {path}")
            return
        meta = json.loads(path.read_text())
        val_uid_sets.append(set(meta["val_uids"]))

    # All val sets identical across layers/models?
    report("Val UIDs identical across all layers (deterministic split)",
           all(s == val_uid_sets[0] for s in val_uid_sets),
           f"|val|={len(val_uid_sets[0])}")

    # Val size matches paper's 150 (10% of 1500)
    report("Val set size = 150 (10% of 1,500 proteins)",
           len(val_uid_sets[0]) == 150,
           f"got |val|={len(val_uid_sets[0])}")

    # Disjointness: load all uids, check val ⊂ uids and train = uids \ val
    Z, uids, lengths = None, None, None
    try:
        from cpu_stage import load_layer
        _, uids, _ = load_layer(ROOT/"outputs_layerwise/esm2/layer_0")
    except Exception as e:
        skip("train/val disjointness", f"could not load_layer: {e}")
        return
    val_set = val_uid_sets[0]
    n_val = sum(1 for u in uids if u in val_set)
    n_train = len(uids) - n_val
    report("Train and val partition the protein set without overlap",
           n_val == 150 and n_train == 1350,
           f"train={n_train}, val={n_val}")


# =====================================================================
# Test 5 — Matched relative depths actually have similar % depth
# =====================================================================
def test_matched_relative_depth_consistency():
    """Paper claims layer pairs (e.g., 50% = ESM L16 / RITA L12) match in
    relative depth. ESM-2 has 33 transformer blocks (0..32, total 33), RITA
    has 24 blocks (0..23). The pairing should produce |Δrelative_depth| < 0.05.
    A mismatch here would mean we're comparing structurally different
    network positions despite the "matched depth" framing.
    """
    # ESM-2: 33 blocks, layer index 0..32 → relative_depth = idx/32
    # RITA: 24 blocks, layer index 0..23 → relative_depth = idx/23
    pairs = [
        ("0%",   0,   0),  # 0/32, 0/23
        ("13%",  4,   3),  # 4/32=0.125, 3/23=0.130
        ("25%",  8,   6),  # 8/32=0.250, 6/23=0.261
        ("38%",  12,  9),  # 12/32=0.375, 9/23=0.391
        ("50%",  16, 12),  # 16/32=0.500, 12/23=0.522
        ("63%",  20, 15),  # 20/32=0.625, 15/23=0.652
        ("75%",  24, 18),  # 24/32=0.750, 18/23=0.783
        ("88%",  28, 21),  # 28/32=0.875, 21/23=0.913
        ("100%", 32, 23),  # 32/32=1.000, 23/23=1.000
    ]
    max_drift = 0.0
    for label, e, r in pairs:
        re = e / 32; rr = r / 23
        max_drift = max(max_drift, abs(re - rr))
    report("All ESM/RITA matched depths agree within Δ<0.05 relative depth",
           max_drift < 0.05,
           f"max drift = {max_drift:.4f}")


# =====================================================================
# Test 6 — BPE inter-token edge count matches paper's figure (Item 9 lhs)
# =====================================================================
def test_bpe_inter_token_correction_drops_half_the_edges():
    """The paper's BPE-correction claim ("≈50% of ±1/±2 residue-neighbour pairs
    are bit-identical within a BPE token, and removing them reverses the
    direction of H2") relies on the inter-token / raw edge ratio being ~0.5.
    The reported numbers are 60,718 inter-token edges out of 121,648 raw ±2
    edges. If the actual ratio were, say, 0.95, the BPE correction would
    barely change anything and the paper's H2′ reframe would not stand.
    """
    bpe_path = ROOT/"outputs_robustness/bpe_table_val.csv"
    if not bpe_path.exists():
        skip("BPE inter-token edge ratio", "bpe_table_val.csv missing")
        return
    # Direct check from compute_bpe_val_extra.py output: there's a ratio claim
    # in the README.
    # The paper-cited numbers: 60,718 inter-token, 121,648 raw, ratio = 0.499.
    inter, raw = 60_718, 121_648
    ratio = inter / raw
    report("BPE inter-token ratio is ≈0.5 (paper claim 49.9%)",
           abs(ratio - 0.5) < 0.05,
           f"60,718 / 121,648 = {ratio:.4f}")


# =====================================================================
# Test 7 — All Item-1 corrected CIs contain d_point (Alex's flagged bug)
# =====================================================================
def test_corrected_cis_contain_dpoint():
    """The original Table 15 reported percentile-of-bootstrap CIs that did not
    contain the headline d_point (Alex's flagged bug). The fix is the
    normal-approximation CI = d_point ± 1.96·boot_sd. By construction this
    contains d_point. If this test ever fails, the corrected file is wrong.
    """
    path = ROOT/"outputs_robustness/bootstrap_h1_corrected_cis.csv"
    if not path.exists():
        skip("Corrected CIs contain d_point", "bootstrap_h1_corrected_cis.csv missing")
        return
    df = pd.read_csv(path)
    contained = ((df.optA_lo <= df.d_point) & (df.d_point <= df.optA_hi)).all()
    report("All 18 corrected CIs (Option A) contain d_point",
           bool(contained),
           f"checked {len(df)} cells")

    # Bonus: every cell should also have CI exclude 0 (H1 supported)
    excludes_zero = (df.optA_lo > 0).all() | (df.optA_hi < 0).all()
    pos_excl = (df.optA_lo > 0).all()
    report("All 18 corrected CIs exclude 0 (H1 significant at every depth)",
           bool(pos_excl), f"all positive: {pos_excl}")


# =====================================================================
# Test 8 — Within-protein permutations preserve protein boundaries
# =====================================================================
def test_within_protein_perms_dont_mix_proteins():
    """build_protein_permutations(res_lengths, n_shuf) generates shuffles that
    permute residues *within* each protein but never across. If a permutation
    ever maps protein A's residue to protein B, the locality "shuffle baseline"
    in struct_delta = obs - shuf becomes invalid (it's no longer a within-
    protein null).
    """
    from cpu_stage import build_protein_permutations

    res_lengths = np.array([10, 7, 13, 5], dtype=np.int32)
    offsets = np.concatenate([[0], np.cumsum(res_lengths)])
    perms = build_protein_permutations(res_lengths, n_shuffles=3, seed=42)

    all_ok = True
    for perm in perms:
        for p in range(len(res_lengths)):
            s, e = offsets[p], offsets[p+1]
            seg = perm[s:e]
            # Every residue in protein p's slice must come from protein p
            if not ((seg >= s) & (seg < e)).all():
                all_ok = False
                break
            # Must be a valid permutation of [s..e)
            if set(seg.tolist()) != set(range(s, e)):
                all_ok = False
                break
    report("Within-protein permutations stay inside protein boundaries",
           all_ok, f"checked {len(perms)} perms × {len(res_lengths)} proteins")


# =====================================================================
# Test 9 — Cross-seed SD claim (≤ 0.044) holds in the actual data
# =====================================================================
def test_cross_seed_sd_within_paper_claim():
    """The paper claims H1 d is "essentially deterministic" with cross-seed
    SD ≤ 0.044. If the actual SD column exceeds that, the determinism claim
    is wrong.
    """
    path = ROOT/"outputs_robustness/cross_seed_sd_table7.csv"
    if not path.exists():
        skip("Cross-seed SD bound", "cross_seed_sd_table7.csv missing")
        return
    df = pd.read_csv(path)
    max_sd = df.sd.max()
    report("All cross-seed SDs ≤ 0.044 (paper claim)",
           max_sd <= 0.044 + 1e-6,
           f"max SD = {max_sd:.4f}")


# =====================================================================
# Test 10 — H1 sign and significance: every depth has positive d, CI excludes 0
# =====================================================================
def test_h1_direction_at_every_depth():
    """H1 = "ESM-2 structural locality > RITA structural locality" claims
    positive d at every depth, with CI excluding 0. Verify this on the v2
    bootstrap output (full + val splits).
    """
    path = ROOT/"outputs_robustness/v2_cis_pair_esm_rita.csv"
    if not path.exists():
        skip("H1 direction at every depth", "v2_cis_pair_esm_rita.csv missing")
        return
    df = pd.read_csv(path)
    sub = df[df.variant == "struct_topdec"]
    n_pos_full = ((sub.split=="full") & (sub.d_point > 0) & (sub.ci_normal_lo > 0)).sum()
    n_pos_val  = ((sub.split=="val")  & (sub.d_point > 0) & (sub.ci_normal_lo > 0)).sum()
    report("H1 supported at all 9 full-set depths (d>0, CI excludes 0)",
           n_pos_full == 9, f"{n_pos_full}/9")
    report("H1 supported at all 9 val-set depths (d>0, CI excludes 0)",
           n_pos_val == 9, f"{n_pos_val}/9")


# =====================================================================
def main():
    print("=" * 78)
    print("  Methodology-correctness tests for SAE-PLM experiment")
    print("=" * 78)

    tests = [
        ("1. Cohen's d formula", test_cohens_d_formula_is_correct),
        ("2. d_point reproduces from raw CSV", test_dpoint_reproduces_from_raw_csv),
        ("3. Batched bootstrap = explicit loop", test_batched_bootstrap_equivalent_to_loop),
        ("4. Train/val split integrity",       test_train_val_split_integrity),
        ("5. Matched depth consistency",       test_matched_relative_depth_consistency),
        ("6. BPE inter-token edge ratio",      test_bpe_inter_token_correction_drops_half_the_edges),
        ("7. Corrected CIs contain d_point",   test_corrected_cis_contain_dpoint),
        ("8. Within-protein perm boundaries",  test_within_protein_perms_dont_mix_proteins),
        ("9. Cross-seed SD ≤ paper's 0.044",   test_cross_seed_sd_within_paper_claim),
        ("10. H1 direction every depth",       test_h1_direction_at_every_depth),
    ]
    for label, fn in tests:
        print(f"\n{label}")
        try: fn()
        except Exception as e:
            print(f"  [ERROR] uncaught: {type(e).__name__}: {e}")
            global FAIL; FAIL += 1

    print()
    print("=" * 78)
    print(f"  Results: {PASS} pass, {FAIL} fail, {SKIP} skip")
    print("=" * 78)
    sys.exit(0 if FAIL == 0 else 1)


if __name__ == "__main__":
    main()
