# Structural Locality Differentiates Residue-Tokenised Bidirectional and Causal Protein Language Model Families

**TopK sparse autoencoders on protein LM activations reveal that bidirectional models encode long-range 3D structure more locally in feature space than matched causal models — a family-level, geometric, correlational pattern, not a claim about training objective alone.**

> **Accepted** to the [ICML 2026 Workshop on Mechanistic Interpretability](https://icml.cc/).  
> This repository is the **living project page** for poster and workshop visitors. A standalone PDF is not required to follow the work; figures, numbers, and reproduction commands are here. Full agent/developer handoff: [`PROJECT_STATUS.md`](PROJECT_STATUS.md).

**Public repo:** [github.com/Uninterpretable-Evolving-Blackbox/Structural-Locality-Differentiates-Residue-Tokenised-Bidirectional-and-Causal-Protein-Language-Model](https://github.com/Uninterpretable-Evolving-Blackbox/Structural-Locality-Differentiates-Residue-Tokenised-Bidirectional-and-Causal-Protein-Language-Model)  
*(git `origin` may still show `submission_2`; content is the same project.)*

---

## Headline result (H1)

At **nine matched relative depths**, SAE features from **ESM-2** (bidirectional, MLM) show higher **structural locality** `L_struct` than **RITA** (causal, CLM) — on the full 1,500-protein set and on the 150-protein validation holdout.

| Scope | What we claim | What we do *not* claim |
|-------|---------------|------------------------|
| **Contrast** | ESM-2 vs RITA at matched depth (650M / 680M, residue-tokenised) | That MLM *alone* causes the effect (architecture, data, and scale co-vary) |
| **Metric** | `L_struct` = shuffle-corrected co-activation contrast for Cα-neighbours (<8 Å, sequence gap ≥12) | General “interpretability” or sequence locality |
| **Evidence type** | Correlational SAE feature geometry (+ convergent extension lenses below) | Proven causal necessity of individual features (steering is a null) |

**Fold-cluster bootstrap (extension, B=1000):** H1 survives phylogenetic non-independence — **9/9 depths** with fold-level 95% CI excluding 0 on the full set (13% depth is the honest marginal cell: lower bound ≈ 0). Effective *n* ≈ 530–640 after design effect (ICC ≈ 0.54–0.74), not 1,500 independent domains.

Canonical CSV: `outputs_robustness/bootstrap_h1_full_bylevel_minact0.csv`

---

## Key numbers (poster-friendly)

Numbers below are from completed runs on disk (`results_*/summary.json`, `outputs_robustness/*.csv`). Concept-F1 headline cells use **symmetric seeds** (trained: SAE init 42/43/44; random: PLM weight-init 0/1/2); other extension metrics are mostly **SAE seed 42** unless noted.

| Lens | ESM-2 (headline layer) | RITA (matched) | Control / honest negative |
|------|------------------------|----------------|---------------------------|
| **`L_struct` H1** | > RITA at all 9 depths | Lower at every depth | Fold bootstrap 9/9 positive (main pair) |
| **Concept-F1** (InterPLM-style, mean best test-F1 per concept) | **0.71 ± 0.009** @ L16 (SAE seeds 42/43/44) | **0.63 ± 0.005** @ L18 | Random PLM weights @ L16: **0.30** mean (range **0.12–0.40**, weight seeds 0/1/2) |
| **Linear probe** (helix AUROC, raw activations) | **0.930** @ L16 | **0.876** @ L18 | Raw **≥** SAE everywhere (SAEs are not for decoding; H1 still reproduces on raw) |
| **`L_struct` null floor** (real vs degree-matched graph) | Real mean **0.053** vs null **−0.105** @ L16 | RITA real ≫ null (~10–22× ratio vs ESM-2 ~150–1350×) | Shuffled-graph null ≈ 0 |
| **Per-feature ablation** (Δ contact precision) | Struct features more harmful than random @ L16, **p=0.022** | — | Effect size **tiny** (weak APC-cosine readout) |
| **Steering** (structural vs random @ 4× dose) | Directionally positive slope | — | **p≈0.10**, CI includes 0 — **not significant** |
| **Second PLM pair** (ProtBert-BFD vs ProGen2) | Descriptive Δ mostly ProtBert > ProGen2 | Fold bootstrap **6/9** depths with CI > 0; **13% depth reverses** | Preliminary generalisation only |

**Concept-F1 headline protocol:** Symmetric comparison at ESM-2 L16 uses **protein-level** val/test split and **80 concepts** for all arms (`results_concept_f1_multiseed_headline/summary.json`). Trained: SAE init seeds {42,43,44}. Random weights: PLM weight-init seeds {0,1,2}. Seed 0 alone gives **0.12** (floor); seeds 1/2 are **~0.40** — report the **triple mean (0.30)** with seed dots, not seed 0 alone. Coarse SCOPe-class signal partly survives on random weights.

---

## Workshop paper vs post-acceptance extensions

| Topic | In submitted workshop paper | Post-acceptance extension (this repo) |
|-------|----------------------------|----------------------------------------|
| **H1** ESM-2 vs RITA `L_struct`, 9 depths | ✓ Main result + metric sweeps, seeds 42/43/44, k/split robustness | Fold-cluster bootstrap; concept-F1; null calibration; probe; ablation; steering |
| **H2** ProtT5 encoder vs decoder crossover (~42% depth) | ✓ | Fold CIs for enc/dec pair (`v2_cis_pair_pt5_fold.csv`) |
| **§4.2** ProtGPT2 BPE sequential-locality artefact | ✓ Inter-token control | — |
| **Within-model depth trajectories** (Appendix G) | ✓ Four PLM signatures | Fold trajectory CIs (`v2_cis_trajectory.csv`) |
| **Multi-seed SAE training** | ✓ Appendix F (cross-seed SD; paper table at 5 depths) | 9-depth grids re-run: `outputs_layerwise_seed{43,44}/` |
| **Randomised PLM weights** | — | ✓ ESM-2 × 9 layers, seed 0; headline L16 seeds 0/1/2 vs trained SAE seeds 42/43/44 |
| **InterPLM-style concept alignment** | — | ✓ 4 residue models × 9 layers |
| **“Do you need SAEs?” probe** | — | ✓ ESM-2 + RITA × 9 (raw wins; ESM-2 raw > RITA raw) |
| **Causal intervention** | Activation clamping (appendix) | Per-feature ablation (E2); steering dose-response (E3, null) |
| **Second PLM pair** (ProtBert-BFD vs ProGen2) | — | ✓ Grid trained; fold bootstrap **partial** (6/9 depths) |
| **Joint k × expansion grid** | Separate single-layer ablations | ✓ 6 headline layers — confirms paper k=256, exp=8× |
| **Fold-level CIs for all paper tables** | Protein-level bootstrap | ✓ ESM/RITA + ProtT5 v2; val sweeps (`v3opt_cis_val_sweeps_fold.csv`) |
| **Cross-attention analysis** | — | ✗ Not started |
| **LLM auto-interp / RAVEL** | — | ✗ Deferred (needs LLM API) |
| **Cross-model summary figure + arXiv draft** | Workshop PDF | ✗ Tables/figures in progress |

---

## Experiment catalogue

Status verified against on-disk artefacts (2026-07-01). “Partial” = done for main configuration but missing robustness breadth or full generalisation.

| ID | Script | Status | Output |
|----|--------|--------|--------|
| **Core pipeline** | `run_all.sh`, `cpu_stage.py`, `analyze_hypotheses.py` | **Done** | `outputs_layerwise/`, `analysis_results/` |
| **H1 fold bootstrap** | `outputs_robustness/compute_h1_bootstrap.py` | **Done** | `bootstrap_h1_*_bylevel_minact0.csv` |
| **E0 Concept-F1** | `experiment_concept_f1.py` | **Done** (54 layers) | `results_concept_f1/` |
| **E1 Probe baseline** | `experiment_probe_baseline.py` | **Done** (ESM-2 + RITA) | `results_probe/` |
| **E2 Causal ablation** | `experiment_causal_features.py` | **Done** (ESM-2 × 9) | `results_causal/` |
| **E3 Steering sweep** | `experiment_steering_sweep.py` | **Done** (null result) | `results_steering_sweep/` |
| **E4 Null calibration** | `experiment_null_calibration.py` | **Done** | `results_null/` |
| **C1 Random weights** | `experiment_random_control.py` | **Done** (seed 0 all layers; seeds 1/2 headline L16) | `outputs_random/`, `outputs_random_weightseed{1,2}/` |
| **Metric independence** | `experiment_interp_comparison.py` | **Done** | `results_interp_comparison/` |
| **SAE diagnostics** | `experiment_sae_diagnostics.py` | **Done** | `results_sae_diagnostics/` (`diagnostics.json`) |
| **LM faithfulness** | `experiment_lm_faithfulness.py` | **Done** (ESM-2) | `results_faithfulness/` |
| **Joint k×exp** | `experiment_joint_sweep.py` | **Done** (6 layers) | `results_joint_sweep/` |
| **Multi-seed SAE** | seeds 43/44 full grid | **Done** | `outputs_layerwise_seed{43,44}/`, `analysis_results_multiseed/` |
| **New PLM pair** | ProtBert-BFD + ProGen2 | **Partial** | `outputs_layerwise_newpair/`, `bootstrap_h1_newpair_*` |
| **Tier 4 fold CIs** | `run_tier4_fold_cis.sh` | **Done** (ESM/RITA, ProtT5, v3 sweeps) | `outputs_robustness/v2_*_fold.csv`, `v3opt_cis_val_sweeps_fold.csv` |
| **Extension multi-seed** | concept-F1 headline seeds 43/44 | **Done** | `results_concept_f1_multiseed_headline/` |
| **Feature viz** | `experiment_feature_viz.py` | **Partial** (sample layers) | `results_feature_viz/` |
| **Cross-attention** | — | **Not done** | — |

---

## What is `L_struct`?

Per SAE feature, shuffle-corrected standardised contrast of neighbour co-activation among **long-range structural contacts** (Cα distance < 8 Å and sequence separation ≥ 12). Implemented in `cpu_stage.py` as `struct_delta`. Sequential variant `L_seq` (±2 neighbours) is reported but **not** an a priori hypothesis — it is confounded (see ProtGPT2 BPE diagnostic in the paper).

---

## Models and depths

Main H1/H2 grid: **nine matched relative depths** {0, 13, 25, 38, 50, 63, 75, 88, 100}%.

| Model | Role | Layers (9-depth grid) |
|-------|------|------------------------|
| ESM-2 t33 | Bidirectional MLM (H1) | 0, 4, 8, 12, 16, 20, 24, 28, 32 |
| RITA-l | Causal CLM (H1 comparator) | 0, 3, 6, 9, 12, 15, 18, 21, 23 |
| ProtT5 enc / dec | Encoder–decoder (H2) | 0, 3, 6, 9, 12, 15, 18, 21, 23 |
| ProtGPT2 | BPE diagnostic only | Not in main H1/H2 |
| ProtBert-BFD / ProGen2 | Second pair (extension) | See `outputs_layerwise_newpair/` |

Dataset: **1,500** SCOPe 2.08 domains (40% ID), ≥80% DSSP coverage, protein-level 90/10 split (seed 42).

---

## Reproducing the work

### Environment

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
source env_local_caches.sh          # HF offline + /tmp pycache (required on iCloud-synced paths)
# DSSP: brew install dssp  (macOS) or conda install -c salilab dssp
```

Always use `.venv/bin/python`. PLM loads expect `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` (set by `env_local_caches.sh`).

### Main workshop pipeline (seed 42, k=256, expansion 8×)

```bash
.venv/bin/python build_dataset.py
.venv/bin/python subsample_dataset.py --n 1500 --min-coverage 0.80 --seed 42
PYTHONUNBUFFERED=1 ./run_all.sh 2>&1 | tee main_run.log
```

Nine-depth densification (if starting from 5-depth defaults): `./run_esm_rita_densify.sh`, `./run_prott5_densify.sh`.

### Extension examples

```bash
# Concept-F1 (one layer)
.venv/bin/python experiment_concept_f1.py \
  --layer-dir outputs_layerwise/esm2/layer_16 \
  --save-dir results_concept_f1/esm2_l16

# H1 fold-cluster bootstrap (canonical)
.venv/bin/python -u outputs_robustness/compute_h1_bootstrap.py \
  --cluster-levels fold,protein --min-active 0 --depths all --n-boot 1000

# Random-weights control (note --weight-seed)
.venv/bin/python experiment_random_control.py \
  --ref-layer-dir outputs_layerwise/esm2/layer_16 \
  --model esm2 --layer 16 --weight-seed 0 \
  --out-layer-dir outputs_random/esm2/layer_16

# Symmetric headline Concept-F1 (SAE seeds 43/44 + aggregate table)
./run_concept_f1_multiseed_headline.sh
.venv/bin/python summarize_concept_f1_multiseed_headline.py

# Poster figures (concept-F1 depth, lens independence, controls with seed dots)
.venv/bin/python make_poster_fig_lenses_controls.py
```

Overnight CPU/MPS runners: `run_overnight_cpu.sh`, `run_overnight_mps.sh`, `run_tier1b_random_weight_seeds.sh`. See [`PROJECT_STATUS.md`](PROJECT_STATUS.md) for the full script index.

---

## Data layout

```
outputs_layerwise/              # Main SAEs (seed 42): esm2, rita, prott5_enc, prott5_dec, protgpt2
outputs_layerwise_seed{43,44}/  # Multi-seed robustness
outputs_layerwise_newpair/      # ProtBert-BFD + ProGen2 extension
outputs_random/                 # Random PLM weights (weight_seed=0)
outputs_random_weightseed{1,2}/ # Additional weight-init seeds (headline L16)

results_concept_f1/             # E0 concept alignment
results_probe/                  # E1 linear probes
results_causal/                 # E2 ablation
results_steering_sweep/         # E3 steering (null)
results_null/                   # E4 graph nulls
results_concept_f1_multiseed_headline/  # Symmetric trained vs random Concept-F1
results_joint_sweep/            # k × expansion grid
analysis_results/               # Workshop hypothesis tables + figures
analysis_results_multiseed/     # Cross-seed aggregates
outputs_robustness/             # Bootstrap CSVs, paper-revision tables
```

Per layer: `Z.npy`, `D.npy`, `sae_model.pt`, `META.json`, `struct_seq_metrics.csv`, …  
Ignore `outputs_layerwise/progen2` — leftover from a removed model.

---

## Figures

| Audience | Location |
|----------|----------|
| **Workshop / poster** | `paper_draft/paper_submission-4/figures/` — `h1_locality_dist_sideways.*`, `poster_concept_f1_depth.*`, `poster_controls.*`, `poster_lens_independence.*` |
| **Main pipeline** | `analysis_results/comparison/` — `H1_structural_main.*`, heatmaps, H2 densification plots |
| **Extension qual.** | `results_feature_viz/` (sample structures) |

---

## Limitations and open work (arXiv summer)

1. **Family-level contrast** — ESM-2 vs RITA varies architecture, objective, and pretraining data together; the second pair (ProtBert vs ProGen2) is only **partially** confirmatory (6/9 fold CIs).
2. **Correlational core** — `L_struct` and concept-F1 are geometric/annotation alignment statistics; steering did not reach significance; ablation effects are small.
3. **Probe honest negative** — Raw activations decode DSSP labels better than SAE features; the SAE value is monosemanticity and sparse structure, not downstream decoding.
4. **Single SAE seed for most extensions** — Workshop H1 had seeds {42,43,44}; **concept-F1 headline** now has symmetric SAE seeds 42/43/44 vs random weight seeds 0/1/2. Probe, steering, ablation, etc. remain mostly seed 42.
5. **Random-weight floor spread** — Weight-init seed 0 gives concept-F1 **0.12**; seeds 1/2 **~0.40** (high variance across inits). Trained mean **0.71** stays well above random mean **0.30**.
6. **Not done:** cross-attention analysis, LLM auto-interp, supervised contact readout for E2, aggregated 4-panel cross-model figure, full arXiv prose.

Principles: convergent evidence, explicit controls (shuffled graph, random weights, random features), protein/fold-cluster bootstrap, val→test for threshold selection — see `.cursor/rules/experimental-rigor.mdc`.

---

## Citation

```bibtex
@inproceedings{anonymous2026structural-locality,
  title={Structural Locality Differentiates Residue-Tokenised Bidirectional and Causal Protein Language Model Families},
  booktitle={ICML 2026 Workshop on Mechanistic Interpretability},
  year={2026}
}
```

---

## For developers

- **Authoritative status & audit:** [`PROJECT_STATUS.md`](PROJECT_STATUS.md) (§11 = latest session)
- **Quick agent snapshot:** [`AGENT_HANDOFF.md`](AGENT_HANDOFF.md)
- **Paper-revision tables:** `outputs_robustness/PAPER_REVISION_SUMMARY.md`
