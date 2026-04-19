#!/usr/bin/env python3
"""
analyze_hypotheses_v4.py — Cross-model, cross-layer hypothesis testing
=======================================================================

v4 changes from v3:
  - H1/H2 plots now show ±SD bands instead of just lines
  - Added H1_structural_main.png (single-panel figure for main text)
  - Added H1_all_models_structural.png (all 4 models comparison)
  - H5 depth trends now include SD bands

Hypothesis structure:
  - H1: ESM-2 structural Δ > ProtGPT2 (bidirectional > causal)
  - H2: ProtGPT2 sequential Δ > ESM-2 (causal > bidirectional)
  - H3: ESM-2 interpretability > ProtGPT2
  - H4: ProtT5 encoder vs decoder
        — NOT a pure causal/bidirectional contrast: the ProtT5 decoder
          cross-attends to the bidirectional encoder, so each decoder
          position has access to "future" residues via cross-attention.
        — Reframed as: unconstrained encoding (encoder) vs autoregressive
          decoding bottleneck (decoder). The pure causal-vs-bidirectional
          test is H1/H2 (ESM-2 vs ProtGPT2).
  - H5: Structural locality increases with depth
        — Tested per-feature (N ≈ 5×n_features) for full statistical power,
          NOT on the 5-point per-layer means (which would be underpowered).

Relative depth matching:
    ESM-2:     [0, 8, 16, 24, 32]  (33 layers total)
    ProtGPT2:  [0, 9, 18, 27, 35]  (36 layers total)
    ProtT5:    [0, 6, 12, 18, 23]  (24 layers total)

Usage:
    !python analyze_hypotheses_v4.py --root outputs_layerwise --save-dir analysis_results_v4
"""

import os, sys, json, argparse, warnings, re
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(style="whitegrid", font_scale=1.05)

ALPHA = 0.05
FDR_THRESHOLD = 0.05

# =====================================================================
#                  RELATIVE DEPTH MATCHING
# =====================================================================

EXPECTED_LAYERS = {
    "esm2":       [0, 8, 16, 24, 32],
    "protgpt2":   [0, 9, 18, 27, 35],
    "prott5_enc": [0, 6, 12, 18, 23],
    "prott5_dec": [0, 6, 12, 18, 23],
}

DEPTH_LABELS = ["0%", "25%", "50%", "75%", "100%"]

def get_relative_depth_matches():
    esm_layers = EXPECTED_LAYERS["esm2"]
    gpt_layers = EXPECTED_LAYERS["protgpt2"]
    return list(zip(DEPTH_LABELS, esm_layers, gpt_layers))

def get_prott5_matched_layers():
    enc = set(EXPECTED_LAYERS["prott5_enc"])
    dec = set(EXPECTED_LAYERS["prott5_dec"])
    return sorted(enc & dec)


# =====================================================================
#                       AUTO-DISCOVERY
# =====================================================================

def discover_layers(root: Path) -> dict:
    root = Path(root)
    entries = {}
    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        for layer_dir in sorted(model_dir.iterdir()):
            if not layer_dir.is_dir():
                continue
            m = re.match(r"layer_(\d+)", layer_dir.name)
            if not m:
                continue
            layer_num = int(m.group(1))
            required = ["struct_seq_metrics.csv", "feature_interpretability.csv",
                        "fold_enrichment.csv"]
            missing = [f for f in required if not (layer_dir / f).exists()]
            if missing:
                print(f"  ⚠ Skipping {model_name}/layer_{layer_num}: missing {missing}")
                continue
            label = f"{model_name}_layer_{layer_num}"
            entries[label] = {
                "model": model_name,
                "layer": layer_num,
                "dir": layer_dir,
            }
    return entries


def build_model_layer_index(entries: dict) -> dict:
    idx = defaultdict(set)
    for label, info in entries.items():
        idx[info["model"]].add(info["layer"])
    return {m: sorted(layers) for m, layers in idx.items()}


# =====================================================================
#                       DATA LOADING
# =====================================================================

def load_layer_data(layer_dir: Path, label: str):
    ss = pd.read_csv(layer_dir / "struct_seq_metrics.csv")
    ip = pd.read_csv(layer_dir / "feature_interpretability.csv")
    fe = pd.read_csv(layer_dir / "fold_enrichment.csv")
    ss["label"] = label
    ip["label"] = label
    fe["label"] = label
    return {"ss": ss, "ip": ip, "fe": fe}


# =====================================================================
#                  PER-LAYER ANALYSIS
# =====================================================================

def analyze_single_layer(data, label, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    ss, ip, fe = data["ss"], data["ip"], data["fe"]
    n = len(ss)

    lines = []
    def log(msg):
        print(msg); lines.append(msg)

    log(f"\n{'='*65}")
    log(f"  {label}  ({n} features)")
    log(f"{'='*65}")

    summary = {"label": label, "n_features": n}

    for col, tag in [("struct_delta", "struct"), ("seq_delta", "seq")]:
        v = ss[col].values
        summary[f"{tag}_mean"] = v.mean()
        summary[f"{tag}_std"] = v.std()
        summary[f"{tag}_median"] = np.median(v)
        summary[f"{tag}_pct_gt0"] = 100 * (v > 0).mean()
        log(f"    {tag.capitalize():>6s} Δ: mean={v.mean():.4f}±{v.std():.4f}, "
            f">0={100*(v>0).mean():.1f}%")

    for prop in ["helix", "strand", "burial"]:
        q = ip[f"q_{prop}"].values
        sig = (q < FDR_THRESHOLD).sum()
        summary[f"pct_sig_{prop}"] = 100 * sig / n

    any_sig = ((ip["q_helix"] < FDR_THRESHOLD) |
               (ip["q_strand"] < FDR_THRESHOLD) |
               (ip["q_burial"] < FDR_THRESHOLD)).sum()
    summary["pct_any_interp"] = 100 * any_sig / n
    log(f"    Interpretable: {any_sig}/{n} ({100*any_sig/n:.1f}%)")

    dead = ((ss["struct_delta"].abs() < 0.01) & (ss["seq_delta"].abs() < 0.01)).sum()
    summary["pct_dead"] = 100 * dead / n

    n_feat_e = fe["feature_idx"].nunique()
    summary["pct_fold_enriched"] = 100 * n_feat_e / n
    summary["n_folds_enriched"] = fe["fold"].nunique()

    (save_dir / "report.txt").write_text("\n".join(lines))
    return summary


# =====================================================================
#        HELPER: two-sample comparison
# =====================================================================

def _compare_two_distributions(va, vb, direction_a_gt_b=True):
    if direction_a_gt_b:
        u, p_mw = stats.mannwhitneyu(va, vb, alternative="greater")
        d = (va.mean() - vb.mean()) / np.sqrt((va.std()**2 + vb.std()**2)/2 + 1e-10)
    else:
        u, p_mw = stats.mannwhitneyu(vb, va, alternative="greater")
        d = (vb.mean() - va.mean()) / np.sqrt((va.std()**2 + vb.std()**2)/2 + 1e-10)

    return {
        "mean_a": va.mean(), "mean_b": vb.mean(),
        "median_a": np.median(va), "median_b": np.median(vb),
        "cohens_d": d, "MW_U": u, "MW_p": p_mw,
        "significant": p_mw < ALPHA,
    }


# =====================================================================
#          CROSS-MODEL HYPOTHESIS TESTING
# =====================================================================

def run_cross_model(all_summaries, all_data, model_layers, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    lines = []
    def log(msg):
        print(msg); lines.append(msg)

    sdf = pd.DataFrame(all_summaries)
    sdf["model"] = sdf["label"].apply(lambda x: "_".join(x.split("_layer_")[0].split("_")))
    sdf["layer"] = sdf["label"].apply(lambda x: int(x.split("_layer_")[1]))
    sdf = sdf.sort_values(["model", "layer"])
    sdf.to_csv(save_dir / "master_summary.csv", index=False)

    log(f"\n{'='*75}")
    log(f"  HYPOTHESIS TESTING v4 — {len(sdf)} model-layer combos")
    log(f"  Using RELATIVE DEPTH matching (not absolute layer numbers)")
    log(f"{'='*75}")

    families = defaultdict(list)
    for _, row in sdf.iterrows():
        families[row["model"]].append(row.to_dict())

    # ==================================================================
    #  H1 & H2: ESM-2 vs ProtGPT2 at matched RELATIVE depths
    # ==================================================================
    log(f"\n\n{'─'*75}")
    log(f"  H1: ESM-2 structural Δ > ProtGPT2  (relative depth matching)")
    log(f"  H2: ProtGPT2 sequential Δ > ESM-2  (relative depth matching)")
    log(f"{'─'*75}")

    depth_matches = get_relative_depth_matches()
    log(f"\n  Relative depth matches:")
    for depth, esm_l, gpt_l in depth_matches:
        log(f"    {depth:>5s}: ESM-2 L{esm_l} ↔ ProtGPT2 L{gpt_l}")

    h1h2_rows = []
    for depth_label, esm_layer, gpt_layer in depth_matches:
        esm_lab = f"esm2_layer_{esm_layer}"
        gpt_lab = f"protgpt2_layer_{gpt_layer}"
        
        if esm_lab not in all_data:
            log(f"\n  {depth_label}: skipped (missing {esm_lab})")
            continue
        if gpt_lab not in all_data:
            log(f"\n  {depth_label}: skipped (missing {gpt_lab})")
            continue

        ss_e = all_data[esm_lab]["ss"]
        ss_g = all_data[gpt_lab]["ss"]

        log(f"\n  ─── {depth_label} depth (ESM-2 L{esm_layer} vs ProtGPT2 L{gpt_layer}) ───")

        for col, hyp, direction in [
            ("struct_delta", "H1", "esm2 > protgpt2"),
            ("seq_delta",    "H2", "protgpt2 > esm2"),
        ]:
            ve = ss_e[col].values
            vg = ss_g[col].values
            a_gt_b = direction.startswith("esm2")
            res = _compare_two_distributions(ve, vg, direction_a_gt_b=a_gt_b)

            verdict = "✓ SUPPORTED" if res["significant"] else "✗ NOT supported"
            log(f"\n    {hyp} ({direction}):")
            log(f"      ESM-2 L{esm_layer}:    mean={ve.mean():.4f}, med={np.median(ve):.4f}")
            log(f"      ProtGPT2 L{gpt_layer}: mean={vg.mean():.4f}, med={np.median(vg):.4f}")
            log(f"      Cohen's d:  {res['cohens_d']:+.4f}")
            log(f"      MW p={res['MW_p']:.2e}  {verdict}")

            h1h2_rows.append({
                "depth": depth_label, 
                "esm2_layer": esm_layer, "protgpt2_layer": gpt_layer,
                "hypothesis": hyp, "metric": col, "direction": direction,
                "esm2_mean": ve.mean(), "esm2_median": np.median(ve),
                "protgpt2_mean": vg.mean(), "protgpt2_median": np.median(vg),
                **res,
            })

    if h1h2_rows:
        h1h2_df = pd.DataFrame(h1h2_rows)
        h1h2_df.to_csv(save_dir / "H1_H2_all_depths.csv", index=False)

        for hyp in ["H1", "H2"]:
            sub = h1h2_df[h1h2_df["hypothesis"] == hyp]
            n_sig = sub["significant"].sum()
            log(f"\n  {hyp} summary: {n_sig}/{len(sub)} depth levels significant")

    # ==================================================================
    #  H3: Feature Interpretability
    # ==================================================================
    log(f"\n\n{'─'*75}")
    log(f"  H3: ESM-2 interpretability > ProtGPT2 (relative depth matching)")
    log(f"{'─'*75}")

    h3_rows = []
    for depth_label, esm_layer, gpt_layer in depth_matches:
        esm_lab = f"esm2_layer_{esm_layer}"
        gpt_lab = f"protgpt2_layer_{gpt_layer}"
        
        if esm_lab not in all_data or gpt_lab not in all_data:
            continue

        ip_e, ip_g = all_data[esm_lab]["ip"], all_data[gpt_lab]["ip"]
        n_e, n_g = len(ip_e), len(ip_g)
        
        sig_e = ((ip_e["q_helix"] < FDR_THRESHOLD) | 
                 (ip_e["q_strand"] < FDR_THRESHOLD) |
                 (ip_e["q_burial"] < FDR_THRESHOLD)).sum()
        sig_g = ((ip_g["q_helix"] < FDR_THRESHOLD) | 
                 (ip_g["q_strand"] < FDR_THRESHOLD) |
                 (ip_g["q_burial"] < FDR_THRESHOLD)).sum()
        
        pct_e, pct_g = 100 * sig_e / n_e, 100 * sig_g / n_g
        
        table = np.array([[sig_e, n_e - sig_e], [sig_g, n_g - sig_g]])
        chi2, p_chi, _, _ = stats.chi2_contingency(table)
        
        verdict = "✓" if (p_chi < ALPHA and pct_e > pct_g) else "✗"
        log(f"\n  {depth_label} (ESM-2 L{esm_layer} vs ProtGPT2 L{gpt_layer}):")
        log(f"    ESM-2:    {pct_e:.1f}% interpretable ({sig_e}/{n_e})")
        log(f"    ProtGPT2: {pct_g:.1f}% interpretable ({sig_g}/{n_g})")
        log(f"    χ²={chi2:.1f}, p={p_chi:.2e}  {verdict}")
        
        h3_rows.append({
            "depth": depth_label,
            "esm2_layer": esm_layer, "protgpt2_layer": gpt_layer,
            "esm2_pct_interp": pct_e, "protgpt2_pct_interp": pct_g,
            "esm2_n": n_e, "protgpt2_n": n_g,
            "chi2": chi2, "p": p_chi,
            "significant": p_chi < ALPHA,
            "esm2_higher": pct_e > pct_g,
        })

    if h3_rows:
        h3_df = pd.DataFrame(h3_rows)
        h3_df.to_csv(save_dir / "H3_interpretability.csv", index=False)

        # ─── H3 SENSITIVITY ANALYSIS ─────────────────────────────────
        # The default q<0.05 hits a ceiling on this dataset (>92% of features
        # are interpretable in BOTH models, leaving only 1–5pp absolute gaps).
        # Re-compute the same H3 contrasts at progressively tighter q-thresholds
        # to escape the ceiling and reveal the true effect size.
        log(f"\n\n{'─'*75}")
        log(f"  H3 sensitivity analysis at multiple q-thresholds")
        log(f"  (default q<0.05 has ceiling effect; tighter thresholds expose gap)")
        log(f"{'─'*75}")

        h3_sensitivity_rows = []
        for q_thr in (0.05, 0.01, 0.001, 1e-4, 1e-6):
            log(f"\n  ── q < {q_thr:.0e} ──")
            for depth_label, esm_layer, gpt_layer in depth_matches:
                esm_lab = f"esm2_layer_{esm_layer}"
                gpt_lab = f"protgpt2_layer_{gpt_layer}"
                if esm_lab not in all_data or gpt_lab not in all_data:
                    continue
                ip_e = all_data[esm_lab]["ip"]
                ip_g = all_data[gpt_lab]["ip"]
                n_e, n_g = len(ip_e), len(ip_g)
                sig_e = ((ip_e["q_helix"] < q_thr) |
                         (ip_e["q_strand"] < q_thr) |
                         (ip_e["q_burial"] < q_thr)).sum()
                sig_g = ((ip_g["q_helix"] < q_thr) |
                         (ip_g["q_strand"] < q_thr) |
                         (ip_g["q_burial"] < q_thr)).sum()
                pct_e = 100 * sig_e / n_e
                pct_g = 100 * sig_g / n_g
                table = np.array([[sig_e, n_e - sig_e], [sig_g, n_g - sig_g]])
                # Robust contingency test: chi2 fails when a cell forces an
                # expected-frequency of zero (happens at very tight thresholds
                # where one model hits 0% interpretable).  Fall back to
                # Fisher's exact, then to NaN, instead of crashing.
                try:
                    chi2, p_chi, _, _ = stats.chi2_contingency(table)
                except ValueError:
                    try:
                        _, p_chi = stats.fisher_exact(table)
                        chi2 = float("nan")
                    except Exception:
                        chi2, p_chi = float("nan"), float("nan")
                gap = pct_e - pct_g
                v = ("✓" if (not np.isnan(p_chi) and p_chi < ALPHA and pct_e > pct_g)
                     else "✗")
                log(f"    {depth_label:>5s} (L{esm_layer:>2}/{gpt_layer:>2}): "
                    f"esm2={pct_e:5.1f}%  protgpt2={pct_g:5.1f}%  "
                    f"gap={gap:+5.1f}pp  p={p_chi:.1e}  {v}")
                h3_sensitivity_rows.append({
                    "q_threshold": q_thr,
                    "depth": depth_label,
                    "esm2_layer": esm_layer, "protgpt2_layer": gpt_layer,
                    "esm2_pct_interp": pct_e, "protgpt2_pct_interp": pct_g,
                    "esm2_n_interp": int(sig_e), "protgpt2_n_interp": int(sig_g),
                    "gap_pp": gap, "chi2": chi2, "p": p_chi,
                    "significant": bool(p_chi < ALPHA),
                    "esm2_higher": bool(pct_e > pct_g),
                })
        pd.DataFrame(h3_sensitivity_rows).to_csv(
            save_dir / "H3_thresholds.csv", index=False)
        log(f"\n  Saved → {save_dir / 'H3_thresholds.csv'}")

        n_sig = sum(1 for r in h3_rows if r["significant"] and r["esm2_higher"])
        log(f"\n  H3 summary: {n_sig}/{len(h3_rows)} depth levels support ESM-2 > ProtGPT2")

    # ==================================================================
    #  H4: ProtT5 Encoder vs Decoder
    # ==================================================================
    log(f"\n\n{'─'*75}")
    log(f"  H4: ProtT5 Encoder vs Decoder")
    log(f"      (unconstrained encoding vs autoregressive decoding;")
    log(f"       NOT a pure bidirectional/causal test — the decoder")
    log(f"       cross-attends to the bidirectional encoder)")
    log(f"      H4a: Encoder structural Δ > Decoder")
    log(f"      H4b: Decoder sequential Δ > Encoder")
    log(f"      H4c: Encoder interpretability > Decoder")
    log(f"{'─'*75}")

    matched_t5 = get_prott5_matched_layers()
    available_enc = model_layers.get("prott5_enc", [])
    available_dec = model_layers.get("prott5_dec", [])
    matched_t5 = sorted(set(matched_t5) & set(available_enc) & set(available_dec))
    
    log(f"\n  Matched ProtT5 layers: {matched_t5}")

    h4_rows = []
    for layer in matched_t5:
        enc_lab = f"prott5_enc_layer_{layer}"
        dec_lab = f"prott5_dec_layer_{layer}"
        
        if enc_lab not in all_data or dec_lab not in all_data:
            log(f"\n  Layer {layer}: skipped (missing data)")
            continue

        ss_enc = all_data[enc_lab]["ss"]
        ss_dec = all_data[dec_lab]["ss"]
        ip_enc = all_data[enc_lab]["ip"]
        ip_dec = all_data[dec_lab]["ip"]

        log(f"\n  ─── Layer {layer} ───")

        ve = ss_enc["struct_delta"].values
        vd = ss_dec["struct_delta"].values
        res_struct = _compare_two_distributions(ve, vd, direction_a_gt_b=True)
        verdict = "✓" if res_struct["significant"] else "✗"
        log(f"\n    H4a (Structural: enc > dec):")
        log(f"      Encoder: mean={ve.mean():.4f}")
        log(f"      Decoder: mean={vd.mean():.4f}")
        log(f"      Cohen's d={res_struct['cohens_d']:+.4f}, p={res_struct['MW_p']:.2e}  {verdict}")

        ve_seq = ss_enc["seq_delta"].values
        vd_seq = ss_dec["seq_delta"].values
        res_seq = _compare_two_distributions(ve_seq, vd_seq, direction_a_gt_b=False)
        verdict = "✓" if res_seq["significant"] else "✗"
        log(f"\n    H4b (Sequential: dec > enc):")
        log(f"      Encoder: mean={ve_seq.mean():.4f}")
        log(f"      Decoder: mean={vd_seq.mean():.4f}")
        log(f"      Cohen's d={res_seq['cohens_d']:+.4f}, p={res_seq['MW_p']:.2e}  {verdict}")

        n_enc, n_dec = len(ip_enc), len(ip_dec)
        sig_enc = ((ip_enc["q_helix"] < FDR_THRESHOLD) | 
                   (ip_enc["q_strand"] < FDR_THRESHOLD) |
                   (ip_enc["q_burial"] < FDR_THRESHOLD)).sum()
        sig_dec = ((ip_dec["q_helix"] < FDR_THRESHOLD) | 
                   (ip_dec["q_strand"] < FDR_THRESHOLD) |
                   (ip_dec["q_burial"] < FDR_THRESHOLD)).sum()
        pct_enc, pct_dec = 100 * sig_enc / n_enc, 100 * sig_dec / n_dec
        table = np.array([[sig_enc, n_enc - sig_enc], [sig_dec, n_dec - sig_dec]])
        chi2, p_chi, _, _ = stats.chi2_contingency(table)
        verdict = "✓" if (p_chi < ALPHA and pct_enc > pct_dec) else "✗"
        log(f"\n    H4c (Interpretability: enc > dec):")
        log(f"      Encoder: {pct_enc:.1f}%")
        log(f"      Decoder: {pct_dec:.1f}%")
        log(f"      χ²={chi2:.1f}, p={p_chi:.2e}  {verdict}")

        h4_rows.append({
            "layer": layer,
            "h4a_enc_mean": ve.mean(), "h4a_dec_mean": vd.mean(),
            "h4a_cohens_d": res_struct["cohens_d"], "h4a_p": res_struct["MW_p"],
            "h4a_significant": res_struct["significant"],
            "h4b_enc_mean": ve_seq.mean(), "h4b_dec_mean": vd_seq.mean(),
            "h4b_cohens_d": res_seq["cohens_d"], "h4b_p": res_seq["MW_p"],
            "h4b_significant": res_seq["significant"],
            "h4c_enc_pct": pct_enc, "h4c_dec_pct": pct_dec,
            "h4c_chi2": chi2, "h4c_p": p_chi,
            "h4c_significant": p_chi < ALPHA,
            "h4c_enc_higher": pct_enc > pct_dec,
        })

    if h4_rows:
        h4_df = pd.DataFrame(h4_rows)
        h4_df.to_csv(save_dir / "H4_enc_vs_dec.csv", index=False)
        
        n_h4a = sum(1 for r in h4_rows if r["h4a_significant"] and r["h4a_cohens_d"] > 0)
        n_h4b = sum(1 for r in h4_rows if r["h4b_significant"])
        n_h4c = sum(1 for r in h4_rows if r["h4c_significant"] and r["h4c_enc_higher"])
        log(f"\n  H4a summary: {n_h4a}/{len(h4_rows)} layers support enc > dec (structural)")
        log(f"  H4b summary: {n_h4b}/{len(h4_rows)} layers support dec > enc (sequential)")
        log(f"  H4c summary: {n_h4c}/{len(h4_rows)} layers support enc > dec (interpretability)")

    # ==================================================================
    #  H5: Depth Trends
    # ==================================================================
    log(f"\n\n{'─'*75}")
    log(f"  H5: Structural locality increases with depth")
    log(f"{'─'*75}")

    h5_rows = []
    for model, rows_list in sorted(families.items()):
        rows_sorted = sorted(rows_list, key=lambda r: r["layer"])
        if len(rows_sorted) < 3:
            log(f"\n  {model}: {len(rows_sorted)} layers — need ≥3 for trend")
            continue

        layers = [r["layer"] for r in rows_sorted]
        struct_means = [r["struct_mean"] for r in rows_sorted]

        # ── (a) Legacy macroscopic test (N = #layers).  Kept for transparency
        #        and to drive the line plots, but NOT used as the paper's
        #        H5 significance claim — only 5 points → severely underpowered.
        macro_rho, macro_p = stats.spearmanr(layers, struct_means)

        # ── (b) Fully-powered per-feature test.  Stack every feature's
        #        struct_delta against its layer index, then take Spearman.
        #        N = sum over layers of n_features ≈ 5 × 10240 ≈ 51,200.
        layer_ix_chunks, delta_chunks = [], []
        for layer in layers:
            label = f"{model}_layer_{layer}"
            if label not in all_data:
                continue
            deltas = all_data[label]["ss"]["struct_delta"].values.astype(np.float64)
            layer_ix_chunks.append(np.full(deltas.shape[0], float(layer), dtype=np.float64))
            delta_chunks.append(deltas)
        layer_ix = np.concatenate(layer_ix_chunks) if layer_ix_chunks else np.array([])
        deltas = np.concatenate(delta_chunks) if delta_chunks else np.array([])
        n_pairs = int(deltas.size)
        if n_pairs >= 10:
            feat_rho, feat_p = stats.spearmanr(layer_ix, deltas)
        else:
            feat_rho, feat_p = float("nan"), float("nan")

        verdict = "✓" if (feat_p < ALPHA and feat_rho > 0) else "✗"

        log(f"\n  {model}:")
        log(f"    Layers: {layers}")
        log(f"    Struct Δ means: {['%.4f' % m for m in struct_means]}")
        log(f"    Macro (5-layer means) ρ={macro_rho:+.3f}, p={macro_p:.3f}  [descriptive only]")
        log(f"    Per-feature ρ={feat_rho:+.3f}, p={feat_p:.2e}  (N={n_pairs})  {verdict}")

        h5_rows.append({
            "model": model,
            "n_layers": len(layers),
            "layers": str(layers),
            # Legacy (descriptive)
            "macro_struct_spearman_rho": macro_rho,
            "macro_struct_spearman_p": macro_p,
            # Paper's H5 significance test
            "feature_struct_spearman_rho": feat_rho,
            "feature_struct_spearman_p": feat_p,
            "feature_n": n_pairs,
            "significant": (not np.isnan(feat_p)) and feat_p < ALPHA and feat_rho > 0,
        })

    if h5_rows:
        pd.DataFrame(h5_rows).to_csv(save_dir / "H5_depth_trends.csv", index=False)

    # ==================================================================
    #  PLOTS
    # ==================================================================
    _plot_hypotheses(sdf, all_data, model_layers, h1h2_rows, h3_rows, h4_rows, h5_rows, save_dir)

    (save_dir / "hypothesis_report.txt").write_text("\n".join(lines))
    log(f"\n✅ Full report → {save_dir}")


# =====================================================================
#                    PLOTS (v4 with SD bands)
# =====================================================================

def _add_subplot_label(ax, label, x=-0.08, y=1.05, fontsize=16):
    """Add consistent (a), (b), (c) labels to subplot axes."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=fontsize, fontweight="bold", va="bottom", ha="right")


# Display names for plots
DISPLAY_NAMES = {
    "esm2": "ESM-2",
    "protgpt2": "ProtGPT2",
    "prott5_enc": "ProtT5-enc",
    "prott5_dec": "ProtT5-dec",
}


def _plot_hypotheses(sdf, all_data, model_layers, h1h2_rows, h3_rows, h4_rows, h5_rows, save_dir):
    cmap = {"esm2": "#1f77b4", "protgpt2": "#ff7f0e",
            "prott5_enc": "#2ca02c", "prott5_dec": "#d62728"}

    # ---- H1/H2: Means with SD bands ----
    if h1h2_rows:
        h1h2_df = pd.DataFrame(h1h2_rows)
        
        # Get SD values from sdf for each model/layer
        esm_stats = sdf[sdf["model"] == "esm2"].set_index("layer")
        gpt_stats = sdf[sdf["model"] == "protgpt2"].set_index("layer")
        
        depths = ["0%", "25%", "50%", "75%", "100%"]
        esm_layers = [0, 8, 16, 24, 32]
        gpt_layers = [0, 9, 18, 27, 35]
        x = np.arange(len(depths))
        
        # Get stats, handling missing layers gracefully
        def safe_get(stats_df, layer, col):
            if layer in stats_df.index:
                return stats_df.loc[layer, col]
            return np.nan
        
        esm_struct_mean = [safe_get(esm_stats, l, "struct_mean") for l in esm_layers]
        esm_struct_std = [safe_get(esm_stats, l, "struct_std") for l in esm_layers]
        gpt_struct_mean = [safe_get(gpt_stats, l, "struct_mean") for l in gpt_layers]
        gpt_struct_std = [safe_get(gpt_stats, l, "struct_std") for l in gpt_layers]
        esm_seq_mean = [safe_get(esm_stats, l, "seq_mean") for l in esm_layers]
        esm_seq_std = [safe_get(esm_stats, l, "seq_std") for l in esm_layers]
        gpt_seq_mean = [safe_get(gpt_stats, l, "seq_mean") for l in gpt_layers]
        gpt_seq_std = [safe_get(gpt_stats, l, "seq_std") for l in gpt_layers]
        
        # ========== Two-panel figure (H1 & H2) with SD bands ==========
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Panel A: Structural (H1)
        ax = axes[0]
        ax.plot(x, esm_struct_mean, "o-", color=cmap["esm2"], lw=2.5, markersize=10, label="ESM-2", zorder=3)
        ax.fill_between(x, 
                        [m - s for m, s in zip(esm_struct_mean, esm_struct_std)],
                        [m + s for m, s in zip(esm_struct_mean, esm_struct_std)],
                        color=cmap["esm2"], alpha=0.2, zorder=2)
        ax.plot(x, gpt_struct_mean, "s--", color=cmap["protgpt2"], lw=2.5, markersize=10, label="ProtGPT2", zorder=3)
        ax.fill_between(x,
                        [m - s for m, s in zip(gpt_struct_mean, gpt_struct_std)],
                        [m + s for m, s in zip(gpt_struct_mean, gpt_struct_std)],
                        color=cmap["protgpt2"], alpha=0.2, zorder=2)
        ax.axhline(0, color="grey", ls="--", lw=1, zorder=1)
        ax.set_xticks(x); ax.set_xticklabels(depths, fontsize=11)
        ax.set_xlabel("Relative Depth", fontsize=12)
        ax.set_ylabel("Mean Structural Δ", fontsize=12)
        ax.set_title("H1: Structural Locality", fontsize=13, fontweight="bold")
        ax.legend(fontsize=10); ax.grid(alpha=0.3, zorder=0)
        _add_subplot_label(ax, "(a)")
        
        # Panel B: Sequential (H2)
        ax = axes[1]
        ax.plot(x, esm_seq_mean, "o-", color=cmap["esm2"], lw=2.5, markersize=10, label="ESM-2", zorder=3)
        ax.fill_between(x,
                        [m - s for m, s in zip(esm_seq_mean, esm_seq_std)],
                        [m + s for m, s in zip(esm_seq_mean, esm_seq_std)],
                        color=cmap["esm2"], alpha=0.2, zorder=2)
        ax.plot(x, gpt_seq_mean, "s--", color=cmap["protgpt2"], lw=2.5, markersize=10, label="ProtGPT2", zorder=3)
        ax.fill_between(x,
                        [m - s for m, s in zip(gpt_seq_mean, gpt_seq_std)],
                        [m + s for m, s in zip(gpt_seq_mean, gpt_seq_std)],
                        color=cmap["protgpt2"], alpha=0.2, zorder=2)
        ax.axhline(0, color="grey", ls="--", lw=1, zorder=1)
        ax.set_xticks(x); ax.set_xticklabels(depths, fontsize=11)
        ax.set_xlabel("Relative Depth", fontsize=12)
        ax.set_ylabel("Mean Sequential Δ", fontsize=12)
        ax.set_title("H2: Sequential Locality", fontsize=13, fontweight="bold")
        ax.legend(fontsize=10); ax.grid(alpha=0.3, zorder=0)
        _add_subplot_label(ax, "(b)")
        
        fig.tight_layout()
        fig.savefig(save_dir / "H1_H2_means_with_SD.png", dpi=300)
        fig.savefig(save_dir / "H1_H2_means_with_SD.pdf")
        plt.close(fig)
        
        # ========== Single-panel structural figure (main text) ==========
        fig, ax = plt.subplots(figsize=(8, 5.5))
        ax.plot(x, esm_struct_mean, "o-", color=cmap["esm2"], lw=2.5, markersize=12, 
                label="ESM-2 (bidirectional)", zorder=3)
        ax.fill_between(x, 
                        [m - s for m, s in zip(esm_struct_mean, esm_struct_std)],
                        [m + s for m, s in zip(esm_struct_mean, esm_struct_std)],
                        color=cmap["esm2"], alpha=0.2, zorder=2)
        ax.plot(x, gpt_struct_mean, "s--", color=cmap["protgpt2"], lw=2.5, markersize=12, 
                label="ProtGPT2 (causal)", zorder=3)
        ax.fill_between(x,
                        [m - s for m, s in zip(gpt_struct_mean, gpt_struct_std)],
                        [m + s for m, s in zip(gpt_struct_mean, gpt_struct_std)],
                        color=cmap["protgpt2"], alpha=0.2, zorder=2)
        ax.axhline(0, color="grey", ls="-", lw=1.5, zorder=1)
        ax.axhspan(-0.15, 0, alpha=0.05, color="red", zorder=0)
        ax.text(4.5, -0.07, "Anti-correlated\nwith structure", fontsize=10, ha="right", 
                style="italic", color="#666666")
        ax.set_xticks(x); ax.set_xticklabels(depths, fontsize=12)
        ax.set_xlabel("Relative Depth", fontsize=13)
        ax.set_ylabel("Structural Locality (Δ)", fontsize=13)
        ax.set_title("Structural Locality: ESM-2 vs ProtGPT2", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11, loc="upper right"); ax.grid(alpha=0.3, zorder=0)
        ax.set_ylim(-0.15, 0.20)
        fig.tight_layout()
        fig.savefig(save_dir / "H1_structural_main.png", dpi=300)
        fig.savefig(save_dir / "H1_structural_main.pdf")
        plt.close(fig)
        
        # ========== All 4 models structural plot ==========
        enc_stats = sdf[sdf["model"] == "prott5_enc"].set_index("layer")
        dec_stats = sdf[sdf["model"] == "prott5_dec"].set_index("layer")
        t5_layers = [0, 6, 12, 18, 23]
        
        enc_struct_mean = [safe_get(enc_stats, l, "struct_mean") for l in t5_layers]
        enc_struct_std = [safe_get(enc_stats, l, "struct_std") for l in t5_layers]
        dec_struct_mean = [safe_get(dec_stats, l, "struct_mean") for l in t5_layers]
        dec_struct_std = [safe_get(dec_stats, l, "struct_std") for l in t5_layers]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x, esm_struct_mean, "o-", color=cmap["esm2"], lw=2.5, markersize=10, label="ESM-2", zorder=3)
        ax.fill_between(x, [m-s for m,s in zip(esm_struct_mean, esm_struct_std)],
                        [m+s for m,s in zip(esm_struct_mean, esm_struct_std)], color=cmap["esm2"], alpha=0.15, zorder=2)
        ax.plot(x, gpt_struct_mean, "s--", color=cmap["protgpt2"], lw=2.5, markersize=10, label="ProtGPT2", zorder=3)
        ax.fill_between(x, [m-s for m,s in zip(gpt_struct_mean, gpt_struct_std)],
                        [m+s for m,s in zip(gpt_struct_mean, gpt_struct_std)], color=cmap["protgpt2"], alpha=0.15, zorder=2)
        ax.plot(x, enc_struct_mean, "^-", color=cmap["prott5_enc"], lw=2, markersize=9, label="ProtT5-enc", zorder=3)
        ax.fill_between(x, [m-s for m,s in zip(enc_struct_mean, enc_struct_std)],
                        [m+s for m,s in zip(enc_struct_mean, enc_struct_std)], color=cmap["prott5_enc"], alpha=0.15, zorder=2)
        ax.plot(x, dec_struct_mean, "v--", color=cmap["prott5_dec"], lw=2, markersize=9, label="ProtT5-dec", zorder=3)
        ax.fill_between(x, [m-s for m,s in zip(dec_struct_mean, dec_struct_std)],
                        [m+s for m,s in zip(dec_struct_mean, dec_struct_std)], color=cmap["prott5_dec"], alpha=0.15, zorder=2)
        ax.axhline(0, color="grey", ls="-", lw=1.5, zorder=1)
        ax.axhspan(-0.12, 0, alpha=0.05, color="red", zorder=0)
        ax.set_xticks(x); ax.set_xticklabels(depths, fontsize=12)
        ax.set_xlabel("Relative Depth", fontsize=13)
        ax.set_ylabel("Structural Locality (Δ)", fontsize=13)
        ax.set_title("Structural Locality Across All Models", fontsize=14, fontweight="bold")
        ax.legend(fontsize=10, loc="upper right", ncol=2); ax.grid(alpha=0.3, zorder=0)
        ax.set_ylim(-0.12, 0.18)
        fig.tight_layout()
        fig.savefig(save_dir / "H1_all_models_structural.png", dpi=300)
        fig.savefig(save_dir / "H1_all_models_structural.pdf")
        plt.close(fig)

    # ---- H3: Interpretability comparison ----
    if h3_rows:
        h3_df = pd.DataFrame(h3_rows)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(h3_df))
        width = 0.35
        ax.bar(x - width/2, h3_df["esm2_pct_interp"], width, 
               label="ESM-2", color=cmap["esm2"], alpha=0.8)
        ax.bar(x + width/2, h3_df["protgpt2_pct_interp"], width, 
               label="ProtGPT2", color=cmap["protgpt2"], alpha=0.8)
        ax.set_xlabel("Relative Depth"); ax.set_ylabel("% Interpretable Features")
        ax.set_title("H3: Feature Interpretability")
        ax.set_xticks(x); ax.set_xticklabels(h3_df["depth"])
        ax.set_ylim(0, 105)
        ax.legend(loc="lower left"); ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(save_dir / "H3_interpretability.png", dpi=250)
        plt.close(fig)

    # ---- H4: ProtT5 Encoder vs Decoder (with SD bands) ----
    if h4_rows:
        h4_df = pd.DataFrame(h4_rows)
        
        # Get SD from sdf
        enc_stats = sdf[sdf["model"] == "prott5_enc"].set_index("layer")
        dec_stats = sdf[sdf["model"] == "prott5_dec"].set_index("layer")
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # H4a: Structural with SD
        ax = axes[0]
        layers = h4_df["layer"].values
        enc_struct_std = [enc_stats.loc[l, "struct_std"] if l in enc_stats.index else 0 for l in layers]
        dec_struct_std = [dec_stats.loc[l, "struct_std"] if l in dec_stats.index else 0 for l in layers]
        
        ax.plot(layers, h4_df["h4a_enc_mean"], "o-", 
                color=cmap["prott5_enc"], lw=2, markersize=8, label="Encoder")
        ax.fill_between(layers,
                        h4_df["h4a_enc_mean"] - enc_struct_std,
                        h4_df["h4a_enc_mean"] + enc_struct_std,
                        color=cmap["prott5_enc"], alpha=0.2)
        ax.plot(layers, h4_df["h4a_dec_mean"], "s--", 
                color=cmap["prott5_dec"], lw=2, markersize=8, label="Decoder")
        ax.fill_between(layers,
                        h4_df["h4a_dec_mean"] - dec_struct_std,
                        h4_df["h4a_dec_mean"] + dec_struct_std,
                        color=cmap["prott5_dec"], alpha=0.2)
        ax.axhline(0, c="grey", ls="--", lw=0.6)
        ax.set_xlabel("Layer"); ax.set_ylabel("Mean Structural Δ")
        ax.set_title("H4a: Structural Locality"); ax.legend(); ax.grid(alpha=0.3)
        _add_subplot_label(ax, "(a)")
        
        # H4b: Sequential with SD
        ax = axes[1]
        enc_seq_std = [enc_stats.loc[l, "seq_std"] if l in enc_stats.index else 0 for l in layers]
        dec_seq_std = [dec_stats.loc[l, "seq_std"] if l in dec_stats.index else 0 for l in layers]
        
        ax.plot(layers, h4_df["h4b_enc_mean"], "o-", 
                color=cmap["prott5_enc"], lw=2, markersize=8, label="Encoder")
        ax.fill_between(layers,
                        h4_df["h4b_enc_mean"] - enc_seq_std,
                        h4_df["h4b_enc_mean"] + enc_seq_std,
                        color=cmap["prott5_enc"], alpha=0.2)
        ax.plot(layers, h4_df["h4b_dec_mean"], "s--", 
                color=cmap["prott5_dec"], lw=2, markersize=8, label="Decoder")
        ax.fill_between(layers,
                        h4_df["h4b_dec_mean"] - dec_seq_std,
                        h4_df["h4b_dec_mean"] + dec_seq_std,
                        color=cmap["prott5_dec"], alpha=0.2)
        ax.axhline(0, c="grey", ls="--", lw=0.6)
        ax.set_xlabel("Layer"); ax.set_ylabel("Mean Sequential Δ")
        ax.set_title("H4b: Sequential Locality"); ax.legend(); ax.grid(alpha=0.3)
        _add_subplot_label(ax, "(b)")
        
        # H4c: Interpretability (bar chart)
        ax = axes[2]
        x = np.arange(len(h4_df))
        width = 0.35
        ax.bar(x - width/2, h4_df["h4c_enc_pct"], width, 
               label="Encoder", color=cmap["prott5_enc"], alpha=0.8)
        ax.bar(x + width/2, h4_df["h4c_dec_pct"], width, 
               label="Decoder", color=cmap["prott5_dec"], alpha=0.8)
        ax.set_xlabel("Layer"); ax.set_ylabel("% Interpretable")
        ax.set_title("H4c: Interpretability")
        ax.set_xticks(x); ax.set_xticklabels(h4_df["layer"])
        ax.set_ylim(0, 105)
        ax.legend(loc="lower left"); ax.grid(alpha=0.3, axis="y")
        _add_subplot_label(ax, "(c)")
        
        fig.tight_layout()
        fig.savefig(save_dir / "H4_enc_vs_dec.png", dpi=250)
        plt.close(fig)

    # ---- H5: Depth trends (with SD bands) ----
    families = defaultdict(list)
    for _, row in sdf.iterrows():
        families[row["model"]].append(row.to_dict())
    
    multi = {m: r for m, r in families.items() if len(r) >= 3}
    if multi:
        n_fam = len(multi)
        fig, axes = plt.subplots(1, n_fam, figsize=(5 * n_fam, 4.5), squeeze=False)
        subplot_labels = ["(a)", "(b)", "(c)", "(d)"]
        for idx, (model, rows_list) in enumerate(sorted(multi.items())):
            ax = axes[0, idx]
            rs = sorted(rows_list, key=lambda r: r["layer"])
            layers = [r["layer"] for r in rs]
            struct_means = [r["struct_mean"] for r in rs]
            struct_stds = [r["struct_std"] for r in rs]
            
            ax.plot(layers, struct_means, "o-", c=cmap.get(model, "steelblue"), 
                    lw=2, markersize=8)
            ax.fill_between(layers,
                            [m - s for m, s in zip(struct_means, struct_stds)],
                            [m + s for m, s in zip(struct_means, struct_stds)],
                            alpha=0.2, color=cmap.get(model, "steelblue"))
            ax.axhline(0, c="grey", ls="--", lw=0.8)
            ax.set_xlabel("Layer", fontsize=11)
            ax.set_ylabel("Mean Structural Δ", fontsize=11)
            ax.set_title(DISPLAY_NAMES.get(model, model), fontsize=12)
            ax.grid(alpha=0.3)
            if idx < len(subplot_labels):
                _add_subplot_label(ax, subplot_labels[idx])
        
        fig.suptitle("H5: Structural Locality vs Layer Depth", fontsize=13, y=1.02)
        fig.tight_layout()
        fig.savefig(save_dir / "H5_depth_trends.png", dpi=250, bbox_inches="tight")
        plt.close(fig)

    # ---- Heatmaps ----
    if len(sdf) >= 4:
        for col, title, fn in [
            ("struct_mean", "Structural Locality (mean Δ)", "heatmap_struct"),
            ("seq_mean", "Sequential Locality (mean Δ)", "heatmap_seq"),
            ("pct_any_interp", "% Interpretable Features", "heatmap_interp"),
        ]:
            if col not in sdf.columns: continue
            pivot = sdf.pivot_table(index="model", columns="layer", values=col)
            fig, ax = plt.subplots(figsize=(max(8, pivot.shape[1] * 1.2),
                                            max(3, pivot.shape[0] * 0.8)))
            fmt = ".4f" if "mean" in col else ".1f"
            sns.heatmap(pivot, annot=True, fmt=fmt, cmap="YlOrRd", ax=ax,
                        linewidths=0.5, cbar_kws={"label": col})
            ax.set_title(title)
            fig.tight_layout()
            fig.savefig(save_dir / f"{fn}.png", dpi=250)
            plt.close(fig)


# =====================================================================
#                    PAPER TABLE
# =====================================================================

def generate_paper_table(all_data, save_dir):
    save_dir = Path(save_dir)
    rows = []
    for lab in sorted(all_data.keys()):
        d = all_data[lab]
        ss, ip, fe = d["ss"], d["ip"], d["fe"]
        n = len(ss)
        any_sig = ((ip["q_helix"] < 0.05) | (ip["q_strand"] < 0.05) | 
                   (ip["q_burial"] < 0.05)).sum()
        rows.append({
            "Model/Layer": lab.replace("_layer_", " L"),
            "N": n,
            "Struct Δ": f"{ss['struct_delta'].mean():.3f}±{ss['struct_delta'].std():.3f}",
            "Seq Δ": f"{ss['seq_delta'].mean():.3f}±{ss['seq_delta'].std():.3f}",
            "% struct>0": f"{100 * (ss['struct_delta'] > 0).mean():.1f}",
            "% interp": f"{100 * any_sig / n:.1f}",
            "Folds": fe['fold'].nunique(),
        })
    df = pd.DataFrame(rows)
    df.to_csv(save_dir / "paper_table.csv", index=False)
    print(f"✅ Paper table → {save_dir}/paper_table.csv")


# =====================================================================
#                           MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Cross-model SAE hypothesis testing v4")
    parser.add_argument("--root", type=str, required=True,
                        help="Path to outputs_layerwise/ directory")
    parser.add_argument("--save-dir", type=str, default="analysis_results_v4",
                        help="Output directory")
    args = parser.parse_args()

    root = Path(args.root)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"📂 Scanning {root}...")
    entries = discover_layers(root)
    model_layers = build_model_layer_index(entries)

    print(f"   Found {len(entries)} model-layer combinations:")
    for lab, info in sorted(entries.items()):
        print(f"     {lab:30s}  → {info['dir']}")
    print(f"\n   Model layer index:")
    for model, layers in sorted(model_layers.items()):
        print(f"     {model:15s}: {layers}")

    if not entries:
        print("❌ No valid layer directories found!")
        sys.exit(1)

    all_data = {}
    all_summaries = []
    for label, info in sorted(entries.items()):
        print(f"\n{'─'*65}")
        data = load_layer_data(info["dir"], label)
        all_data[label] = data
        summary = analyze_single_layer(data, label, save_dir / "per_layer" / label)
        all_summaries.append(summary)

    if len(all_data) >= 2:
        print(f"\n\n{'━'*75}")
        print(f"  HYPOTHESIS TESTS v4")
        print(f"{'━'*75}")
        run_cross_model(all_summaries, all_data, model_layers, save_dir / "comparison")
        generate_paper_table(all_data, save_dir / "comparison")

    print(f"\n{'━'*75}")
    print(f"✅ All outputs → {save_dir}/")
    print(f"{'━'*75}")


if __name__ == "__main__":
    main()