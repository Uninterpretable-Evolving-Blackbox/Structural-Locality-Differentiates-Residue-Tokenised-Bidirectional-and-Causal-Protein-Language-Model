# SAE-PLM: Sparse Autoencoders Reveal How Training Objectives Shape Structural Representations in Protein Language Models

TopK Sparse Autoencoders (Gao et al., 2024) trained on four protein language model architectures at nine matched relative depths, comparing how bidirectional vs causal training objectives shape learned feature geometry. ESM-2 vs RITA is the main bidirectional-vs-causal contrast (both residue-tokenised, size-matched); ProtGPT2 is included as a BPE-tokenisation diagnostic only; ProtT5 encoder and decoder are probed separately to test how cross-attention propagates structural context under autoregressive decoding.

## Key Results

The paper declares **two a priori hypotheses** (no a priori L_seq hypothesis is stated; sequential results are reported alongside L_struct in Table 2):

| Hypothesis | Result | Effect size |
|---|---|---|
| **H1**: ESM-2 L_struct > RITA L_struct at every matched depth | **9/9 depths supported on full (n=1,500) and 9/9 on val (n=150)** | Cohen's d on L_struct = +0.05 to +1.44 (full); +0.06 to +0.74 (val); bootstrap 95% CI excludes 0 at every cell |
| **H2**: ProtT5 encoder leads decoder on L_struct at late depths; decoder leads at early — depth-dependent reversal | **Crossover localised between L9 and L12 (≈42% relative depth)** | d ∈ [−0.40, −0.13] at depths 0–39%, d ∈ [+0.40, +0.69] at depths 52–100%; flip preserved on val |

### Methodological observation (paper contribution 2, §4.2)

ProtGPT2's BPE tokeniser maps multiple residues into a single token, and the standard residue-projection (uniform 1/L weights) gives every within-token residue **bit-identical SAE activations**. A naïve ProtGPT2-vs-ESM-2 L_seq contrast inherits this artefact: 50.0% of ±2 residue-neighbour pairs are within-token (60,718 / 121,648 directed pairs on val), inflating the apparent causal advantage to d up to +5.9 (Table 3). Restricting to **inter-token** pairs reverses the direction at every depth (27/27 cells across windows ±1/±2/±4). The native-residue RITA-vs-ESM-2 contrast also shows no consistent causal sequential advantage (4/5 depths null or favouring ESM-2). At residue resolution, causal PLMs have no L_seq advantage over bidirectional ones — the apparent advantage is a tokenisation artefact, not architectural. (H1's structural test is unaffected: it requires sequence separation ≥ 12, far beyond any BPE span.)

### Robustness (Appendix sweeps + reproducibility)

| Check | Result |
|---|---|
| Multi-seed reproducibility (seeds 42/43/44) | Cross-seed SD on H1 d ≤ 0.044 (Appendix F Table 7 reports the 5-depth subset {0, 25, 50, 75, 100}%; SAEs are essentially deterministic across initialisation) |
| k-robustness (k=128 vs paper's k=256) | Direction preserved across ESM-2/ProtGPT2/ProtT5/RITA |
| Data-split robustness (split seed 99 vs paper's 42) | H1 direction preserved at every depth |
| Metric sweep (Cα ∈ {6, 8, 10} Å, sep ∈ {8, 12, 24}, quantile ∈ {5, 10, 20}%, window ∈ {±1, ±2, ±4}) | H1 holds at 27/27 Cα×depth cells and 36/45 sep×quantile cells (the 9 non-significant cells all at separation 24 at intermediate depths, where the stricter filter leaves too few edges) |
| Fresh 1,500-protein SCOPe subsample | Direction preserved, max \|Δd\| = 0.08 |
| Smaller PLM pair (ESM-2 t12 vs RITA-s, ~10× smaller) | 5/5 significant in correct direction |
| Within-model L_struct trajectories (Appendix G) | Four qualitatively distinct depth signatures (ESM-2 / RITA / ProtT5-enc / ProtT5-dec) |
| Active-residue threshold sensitivity (Appendix M) | Direction preserved at every threshold; magnitude varies at extreme depths |
| Per-protein cosine baseline at layer 0 (Appendix L) | RITA's raw-embedding geometry favours contacts more than ESM-2's (d = −0.94, opposite to the SAE-feature direction) — sparse coding does substantive work at L0 |

## Models and Depth Matching

H1 (ESM-2 vs RITA) and H2 (ProtT5 enc vs dec) are tested at **nine matched relative depths** {0, 13, 25, 38, 50, 63, 75, 88, 100}%. ProtGPT2 is probed at five depths {0, 25, 50, 75, 100}% for the BPE-tokenisation diagnostic in §4.2 of the paper, and at nine depths in the full metric sweep (Appendix I).

| Model | Architecture | Params | Tokenization | 9-depth grid (paper main) |
|---|---|---|---|---|
| ESM-2 (t33) | Bidirectional encoder (MLM) | 650M | residue | 0, 4, 8, 12, 16, 20, 24, 28, 32 |
| RITA_l | Causal decoder (CLM) | 680M | **residue** | 0, 3, 6, 9, 12, 15, 18, 21, 23 |
| ProtT5-enc | Bidirectional encoder (seq2seq) | ~1.2B | residue | 0, 3, 6, 9, 12, 15, 18, 21, 23 |
| ProtT5-dec | Autoregressive decoder (seq2seq) | ~3B | residue | 0, 3, 6, 9, 12, 15, 18, 21, 23 |
| ProtGPT2 | Causal decoder (CLM) | 738M | BPE | 5-depth (BPE diagnostic): 0, 9, 18, 27, 35; 9-depth (sweep): 0, 4, 9, 13, 18, 22, 27, 31, 35 |

**Why ESM-2 vs RITA is the main contrast:** both are residue-tokenised (1 token per amino acid) and size-matched at 650M / 680M params. ProtGPT2's BPE tokeniser maps multiple residues into a single token (mean ~3 residues/token), which under standard residue-projection gives within-token residues bit-identical SAE activations. This biases the residue-pair L_seq metric (§4.2). RITA preserves a clean apples-to-apples comparison without BPE confounds.

**Historical note:** an earlier paper version probed at five depths only; the current paper's main grid is the nine-depth densification reported in Tables 2 and the within-model trajectories of Appendix G. The 5-point grid persists only for the ProtGPT2 BPE diagnostic.

## Dataset

- 1,500 proteins from SCOPe 2.08 (40% identity filter)
- Stratified by fold (432 folds, all 7 SCOPe classes)
- Filtered to >= 80% DSSP secondary-structure coverage
- 295,240 total residues (mean length 197)
- Protein-level 90/10 train/validation split (seed 42)

## Pipeline

```
build_dataset.py          Step 0: Pull SCOPe, filter, download PDBs, compute DSSP
    |
subsample_dataset.py      Optional: subsample from full SCOPe to N proteins
    |
run_unsupervised.py       Step 1: Extract PLM embeddings + train SAEs (GPU/MPS)
    |                             - Bricken normalization (mean L2 → sqrt(D))
    |                             - Protein-level train/val split
    |                             - TopK SAE with AuxK dead-latent recovery
    |
cpu_stage.py              Step 2: Per-layer structural analysis (CPU)
    |                             - Feature-structure correlations (helix/strand/burial)
    |                             - Fold enrichment
    |                             - Structural vs sequential locality (sparse matrix)
    |                             - UMAP on decoder dictionary + residue activations
    |                             - TopK sensitivity sweep
    |
analyze_hypotheses.py     Step 3: Cross-model hypothesis testing
                                  - H1: ESM-2 vs RITA L_struct at 9 matched depths
                                  - H2: ProtT5 enc vs dec L_struct at 9 matched depths
                                  - BPE diagnostic: ProtGPT2 L_seq with inter-token control
                                  - Within-model L_struct trajectories (Appendix G)
                                  - Feature interpretability sensitivity

run_all.sh                Orchestrates Steps 0-3 end-to-end
```

## Reproducing the Experiments

### Prerequisites

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
# Also need DSSP: brew install dssp (macOS) or conda install -c salilab dssp
```

### Main pipeline (seed=42, k=256, expansion=8)

```bash
# Step 0: Build dataset (downloads SCOPe + PDBs, ~15 min)
.venv/bin/python build_dataset.py

# Subsample to 1500 proteins
.venv/bin/python subsample_dataset.py --n 1500 --min-coverage 0.80 --seed 42

# Steps 1-3: Full pipeline (~3-4 hr on Apple Silicon MPS)
PYTHONUNBUFFERED=1 ./run_all.sh 2>&1 | tee main_run.log
```

### Multi-seed runs (for error bars)

```bash
PYTHONUNBUFFERED=1 SAE_SEED=43 RUN_SUFFIX=_seed43 ./run_all.sh 2>&1 | tee run_seed43.log
PYTHONUNBUFFERED=1 SAE_SEED=44 RUN_SUFFIX=_seed44 ./run_all.sh 2>&1 | tee run_seed44.log
.venv/bin/python aggregate_seeds.py --seeds 42 43 44 --out analysis_results_multiseed
```

### k-robustness check

```bash
PYTHONUNBUFFERED=1 K_SPARSE=128 RUN_SUFFIX=_k128 ./run_all.sh 2>&1 | tee run_k128.log
```

### Ablations (ESM-2 layer 16)

```bash
# k_sparse ablation (k in {64, 128, 256}), ~15 min
.venv/bin/python ablation_k.py

# Expansion factor ablation ({4, 8, 16, 32}x, fixed-k + matched-density), ~50 min
.venv/bin/python ablation_expansion.py
```

### Activation clamping (causal evidence)

```bash
# Uses CPU to avoid MPS fp16/fp32 dtype mismatch in the intervention hook
PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv/bin/python experiment_activation_clamping.py \
    --layer-dir outputs_layerwise/esm2/layer_16 \
    --pdb-dir cache/pdb_files \
    --device cpu \
    --top-k 10 \
    --max-proteins 200 \
    --save-dir results_clamping_esm2_l16
```

### BPE-crossing control (H2′)

```bash
# Single-layer (original single-depth experiment)
.venv/bin/python experiment_bpe_correction.py \
    --layer-dir outputs_layerwise/protgpt2/layer_18 \
    --esm2-layer-dir outputs_layerwise/esm2/layer_16 \
    --save-dir results_bpe_crossing_l18

# All 5 matched depths on the main run (~15 min)
.venv/bin/python experiment_bpe_crossing_all_depths.py \
    --outputs-dir outputs_layerwise \
    --out        results_bpe_crossing \
    --n-shuffles 5

# Same, on any robustness run (seed43/44/k128/split99)
.venv/bin/python experiment_bpe_crossing_all_depths.py \
    --outputs-dir outputs_layerwise_seed43 \
    --out        results_bpe_crossing_seed43 \
    --n-shuffles 5
```

### Val-only checks

```bash
# H1/H2 on 150-protein val subset with the raw (original) sequential metric
.venv/bin/python experiment_val_only_h1h2.py --n-shuffles 5

# H2' on val, using the BPE-corrected inter-token adjacency
.venv/bin/python experiment_h2prime_valonly.py --n-shuffles 5
```

### Data-split robustness (SPLIT_SEED)

```bash
# Full pipeline rerun with a different 150-protein val split (~7 hr)
SAE_SEED=42 SPLIT_SEED=99 RUN_SUFFIX=_split99 ./run_all.sh all

# Then BPE-crossing on the new ProtGPT2 outputs
.venv/bin/python experiment_bpe_crossing_all_depths.py \
    --outputs-dir outputs_layerwise_split99 \
    --out        results_bpe_crossing_split99 \
    --n-shuffles 5
```

### Metric hyperparameter sweep

```bash
# 9-cell grid: seq_gap_min ∈ {8,12,24} × topk_frac ∈ {0.05,0.10,0.20} (~2.5 hr)
.venv/bin/python experiment_metric_sweep.py \
    --outputs-dir outputs_layerwise \
    --out        results_metric_sweep \
    --n-shuffles 5
```

### Overnight robustness chain (split99 + BPE crossing + metric sweep)

```bash
./run_overnight.sh    # runs sequentially; tee to overnight.log for review
```

### Residue-level causal comparator — RITA_l (v6 headline)

```bash
# Smoke test (loads RITA_l ~3 GB from HF on first run; ~30 s if cached)
.venv/bin/python smoke_test_rita.py

# Single-seed main run (MODEL=rita only; ~1.5 hr on MPS)
./run_rita.sh

# Full 5-variant robustness package + val-only + 9-cell metric sweep + aggregator
# (~8 hr on MPS; mirrors the ProtGPT2 overnight chain for ESM-2 vs RITA)
./run_rita_overnight.sh  2>&1 | tee rita_overnight.log
```

Produces `analysis_results_rita{,_seed43,_seed44,_k128,_split99}/comparison/`,
`analysis_results_valonly_rita/`, `results_metric_sweep_rita/`, and the
5-run master table `analysis_results_master_rita/`.

### ProtT5 depth densification — H2 (paper §4.3 / Appendix H)

Builds the **9-depth grid** `{0, 3, 6, 9, 12, 15, 18, 21, 23}` for ProtT5 enc and dec, used to localise the encoder-vs-decoder structural-locality crossover (paper Figure 1 caption / §4.3 reports the crossover at ~42% relative depth, between L9 and L12). Originally an extension over an earlier 5-point grid; **9 depths is now the paper's default** for H2. The de-risk gate trains ProtT5-dec layer 9 first to confirm val_EV and mean struct_delta fit smoothly before committing to the remaining 7 SAEs.

```bash
./run_prott5_densify.sh  2>&1 | tee prott5_densify.log
```

Produces `analysis_results/comparison/H3_enc_vs_dec_dense.{csv,png,pdf,txt}` with per-layer enc-vs-dec Cohen's d for both `struct_delta` and `seq_delta` and a linearly-interpolated zero-crossing marker. (File names retain "H3" prefix for backwards-compatibility with v5 artefacts; the paper now calls this H2.)

### ESM-2 + RITA within-model trajectories — Appendix G

Builds the 9-depth grid for the within-model L_struct trajectory analysis (paper Appendix G, Figure 2, Table 8). ESM-2: `{0,4,8,12,16,20,24,28,32}` (9 layers). RITA: `{0,3,6,9,12,15,18,21,23}` (9 layers). Reports per-layer mean `struct_delta` with bootstrap 95% CI across features (1000 resamples). De-risk gate trains ESM-2 layer 12 first.

```bash
./run_esm_rita_densify.sh  2>&1 | tee esm_rita_densify.log
```

Produces `analysis_results/comparison/H5_within_model_dense.{csv,png,pdf,txt}`. (File name retains "H5" prefix for backwards-compatibility with the v5 hypothesis numbering; the v6 paper reports these as the within-model depth trajectories of Appendix G, not as a numbered hypothesis.)

### Per-model layer-list env-var overrides

The densification experiments use four env vars that override the
default 9-depth matched-depth plan without editing source:

```bash
ESM2_LAYERS="0,4,8,12,16,20,24,28,32"        MODEL=esm2      ./run_all.sh esm2
RITA_LAYERS="0,3,6,9,12,15,18,21,23"          MODEL=rita      ./run_all.sh rita
PROTT5_ENC_LAYERS="0,3,6,9,12,15,18,21,23"    MODEL=prott5_enc ./run_all.sh prott5_enc
PROTT5_DEC_LAYERS="0,3,6,9,12,15,18,21,23"    MODEL=prott5_dec ./run_all.sh prott5_dec
```

Existing layer directories with `META.json` skip at the per-layer level, so
only the new layers re-train; no retraining of committed SAEs.

### Preflight check (30 s)

```bash
.venv/bin/python experiment_preflight.py
```

Validates that cache files, layer outputs, META.val_uids, ProtGPT2 BPE round-trip, and the matched-depth pairs are in place before committing overnight compute.

### Smoke test

```bash
.venv/bin/python smoke_test.py
```

## Environment Variables

| Variable | Default | Used by | Description |
|---|---|---|---|
| `DEVICE` | auto-detect | `run_all.sh`, `run_unsupervised.py` | Compute device: `cuda`, `mps`, or `cpu` |
| `MODEL` | `all` | `run_all.sh`, `run_unsupervised.py` | Which PLM(s): `esm2`, `protgpt2`, `prott5_enc`, `prott5_dec`, `rita`, or `all` |
| `SAE_SEED` | `42` | `run_all.sh`, `run_unsupervised.py` | SAE weight initialization seed |
| `K_SPARSE` | `256` | `run_all.sh`, `run_unsupervised.py` | Number of active latents per token |
| `EXPANSION` | `8` | `run_all.sh`, `run_unsupervised.py` | SAE expansion factor (hidden_dim = input_dim x expansion) |
| `RUN_SUFFIX` | `""` | `run_all.sh`, `run_unsupervised.py` | Output directory suffix (e.g., `_seed43`, `_k128`) |
| `SPLIT_SEED` | `42` | `run_unsupervised.py` | Protein-level 90/10 train/val split seed. Override to probe robustness to protein subset (set RUN_SUFFIX so outputs don't clobber the main run) |
| `ESM2_LAYERS` | `0,8,16,24,32` (5-pt) | `run_unsupervised.py` | Comma-sep ESM-2 layer list. Source default is 5 depths; the paper's 9-depth main grid is set by `run_esm_rita_densify.sh` to `0,4,8,12,16,20,24,28,32`. |
| `RITA_LAYERS` | `0,6,12,18,23` (5-pt) | `run_unsupervised.py` | Comma-sep RITA layer list. Source default is 5 depths; the paper's 9-depth main grid is set by `run_esm_rita_densify.sh` to `0,3,6,9,12,15,18,21,23`. |
| `PROTT5_ENC_LAYERS` | `0,6,12,18,23` (5-pt) | `run_unsupervised.py` | Comma-sep ProtT5-enc layer list. Source default is 5 depths; the paper's 9-depth main grid is set by `run_prott5_densify.sh` to `0,3,6,9,12,15,18,21,23`. |
| `PROTT5_DEC_LAYERS` | `0,6,12,18,23` (5-pt) | `run_unsupervised.py` | Comma-sep ProtT5-dec layer list. Source default is 5 depths; the paper's 9-depth main grid is set by `run_prott5_densify.sh` to `0,3,6,9,12,15,18,21,23`. |
| `SAE_BATCH` | `4096` | `train_sae.py` | SAE training batch size (MPS) |
| `SAE_CPU_BATCH` | `4096` | `train_sae.py` | SAE training batch size (CPU, >= 8 cores) |
| `ESM2_BATCH` | `32` | `extract_embeddings.py` | ESM-2 inference batch size (MPS) |
| `PROTT5_BATCH` | `4` | `extract_embeddings.py` | ProtT5 inference batch size (MPS) |
| `PROTGPT2_BATCH` | `16` | `extract_embeddings.py` | ProtGPT2 inference batch size (MPS) |
| `RITA_BATCH` | `12` | `extract_embeddings.py` | RITA inference batch size (MPS) |
| `CPU_STAGE_MEM_GB` | `100` | `cpu_stage.py` | Memory budget for parallel workers (GB) |
| `SAE_PRECISION` | `fp32` | `train_sae.py` | Override to `fp16` for MPS (not recommended) |

## Output Directory Structure

```
outputs_layerwise{RUN_SUFFIX}/
  {esm2,protgpt2,prott5_enc,prott5_dec}/
    META.json                        # Model-level metadata
    layer_{N}/
      Z.npy                          # SAE activations (N_tokens x hidden_dim, fp16)
      D.npy                          # Decoder dictionary (hidden_dim x input_dim, fp16)
      sae_model.pt                   # Trained SAE state dict
      META.json                      # Per-layer metadata (train/val EV, gap, norm_scale, seed)
      lengths.npy                    # Per-protein token counts (BPE for ProtGPT2, residue for others)
      offsets.npy                    # Cumulative token offsets
      uids.json                     # Protein identifiers
      sequences.json                # Protein sequences
      struct_seq_metrics.csv         # Per-feature structural/sequential locality scores
      feature_interpretability.csv   # Per-feature correlations with helix/strand/burial
      fold_enrichment.csv            # SCOPe fold enrichment per feature
      topk_sensitivity_sweep.csv     # Sensitivity to activation threshold
      plot_struct_seq.png            # Structural vs sequential locality scatter
      umap_decoder_dual.png          # UMAP of decoder dictionary (cosine + euclidean)
      umap_activations_{cosine,euclidean}.png  # UMAP of residue activations

analysis_results{RUN_SUFFIX}/
  comparison/
    hypothesis_report.txt            # Full H1-H5 hypothesis test report
    H1_H2_all_depths.csv             # Per-depth H1/H2 effect sizes
    H3_interpretability.csv          # H3 interpretability comparison
    H3_thresholds.csv                # H3 sensitivity at q in {0.05, 0.01, 0.001, 1e-4, 1e-6}
    H4_enc_vs_dec.csv                # H4 ProtT5 encoder vs decoder
    H5_depth_trends.csv              # H5 per-feature depth Spearman
    master_summary.csv               # Per-(model,layer) summary statistics
    paper_table.csv                  # Publication-ready summary table
    H1_structural_main.{png,pdf}     # Main-text structural locality figure
    H1_H2_means_with_SD.{png,pdf}    # H1+H2 means with SD bands
    H1_all_models_structural.{png,pdf}  # All 4 models structural comparison
    H3_interpretability.png          # H3 bar chart
    H4_enc_vs_dec.png                # H4 layer-wise panel
    H5_depth_trends.png              # H5 depth profiles
    heatmap_{struct,seq,interp}.png  # Cross-model heatmaps

analysis_results_multiseed/
  cross_seed_summary.csv             # Per-layer EV mean +/- std across seeds
  cross_seed_h1h2_summary.csv        # H1/H2 Cohen's d mean +/- std across seeds
  cross_seed_struct_seq.csv          # Struct/seq deltas per seed
  cross_seed_meta.csv                # Raw per-seed per-layer EVs

analysis_results/
  ablation_k_sparse.{csv,png,pdf}    # k_sparse ablation (k in {64, 128, 256})
  ablation_expansion.{csv,png,pdf}   # Expansion factor ablation (4x, 8x, 16x, 32x)

results_clamping_esm2_l16/
  clamping_summary.csv               # Baseline/ablation/amplification precision
  clamping_per_protein.csv           # Per-protein precision under each condition
  clamping_results.{png,pdf}         # Box plot + paired-difference violin
  target_features.csv                # Top-10 structural features used for intervention
```

## File Descriptions

### Core Pipeline
| File | Description |
|---|---|
| `run_all.sh` | End-to-end orchestrator: dataset build -> GPU stage -> CPU stage -> hypothesis tests |
| `build_dataset.py` | Downloads SCOPe proteins, filters by length/coverage, extracts DSSP labels |
| `extract_embeddings.py` | Extracts hidden states from ESM-2, ProtGPT2, ProtT5-enc, ProtT5-dec, RITA_l at specified layers |
| `sae.py` | TopK Sparse Autoencoder model (encoder, decoder, AuxK loss, dead-latent tracking) |
| `train_sae.py` | SAE training loop with mixed precision, hardware auto-config, explained variance computation |
| `run_unsupervised.py` | GPU stage driver: loads dataset, protein-level split, trains SAEs per layer, computes holdout EV |
| `cpu_stage.py` | Structural analysis: feature-structure correlations, locality metrics, fold enrichment, UMAPs |
| `analyze_hypotheses.py` | Cross-model hypothesis testing (H1-H5) with depth matching and publication figures |

### Ablation & Supplementary
| File | Description |
|---|---|
| `ablation_k.py` | k_sparse ablation on ESM-2 layer 16 (Appendix A) |
| `ablation_expansion.py` | Expansion factor sweep with fixed-k and matched-density strategies (Appendix B) |
| `aggregate_seeds.py` | Combines multi-seed (42, 43, 44) runs into cross-seed mean +/- std summaries |
| `experiment_activation_clamping.py` | Causal intervention: ablate/amplify top SAE features during ESM-2 forward pass |
| `subsample_dataset.py` | Deterministic fold-stratified subsampling from full SCOPe to N proteins |

### Paper revision (May 2026): bootstrap CIs and paper-revision tables
Addresses the eight reviewer items raised pre-submission. All point estimates
verified against the paper's published Table 1 / `bootstrap_h1_*` values
(exact match where overlap exists). All bootstrap CIs are normal-approximation
`d_point ± 1.96·boot_sd`, B=1000, protein-level cluster bootstrap.

| File | Description |
|---|---|
| `outputs_robustness/compute_cis_v2.py` | Paired ESM-2 vs RITA + ProtT5 enc-vs-dec bootstrap; 5 active-mask variants (top-decile struct, top-decile seq, struct@0.5/1/2 ×s) per cell, partial-CSV checkpointing each depth. Threaded via OMP/MKL. |
| `outputs_robustness/compute_cis_v3_optimized.py` | Val-only sweep CIs at Cα cutoffs {6, 10} Å and sequence windows ±{1, 4}. Two key optimisations: batched bootstrap (one big `W @ contribs` matmul instead of 1000 small ones, ≈100× speedup; built-in correctness test asserts max-abs error <1e-15 vs explicit loop), and shared sigma/percentile/active-mask across the four adjacency variants for a given (model, layer). |
| `make_within_model_trajectory_plot.py` | Renders Figure 2 (Appendix G): mean L_struct vs depth for all four PLMs with bootstrap CI bands, from `v2_cis_trajectory.csv`. ICML Type-42 fonts. |
| `outputs_robustness/PAPER_REVISION_SUMMARY.md` | Master document — full tables for all 8 items with sources and per-cell numbers. Paper-pasteable. |

CSVs produced (small, paper-quotable):

| File | Reviewer item | Rows |
|---|---|---|
| `bootstrap_h1_corrected_cis.csv` | 1: H1 bootstrap CI fix (Table 15) | 18 |
| `v2_cis_pair_pt5.csv` | 2: ProtT5 enc-vs-dec H2 CIs | 36 |
| `v3opt_cis_val_sweeps.csv` | 3: val cutoff + window CIs | 72 |
| `v2_cis_pair_esm_rita.csv` | 4 + 5: threshold sensitivity + L_seq with CIs | 90 |
| `Lseq_esm_rita.csv`, `prott5_enc_vs_dec.csv` | 5 + 2 (point estimates only) | 9 + 9 |
| `sae_val_ev_table.csv` | 6: RITA-l SAE val EV | 9 |
| `cross_seed_sd_table7.csv` | 7: cross-seed SD per depth | 10 |
| `sweep_significance_markers.csv` | 8: per-cell ✓/✗ for cutoff and window sweeps | 135 |
| `interp_appendixC_table.csv` | 9: per-model %Interp at q<0.05 and q<10⁻⁶ | 10 |
| `v2_cis_trajectory.csv` | within-model L_struct trajectory CIs (4 models × 9 depths) | 36 |
| `_ITEM2/_ITEM3/_ITEM4_*.csv` | sorted-clean per-item paste-in views | as above |

### Post-submission robustness scripts
| File | Description |
|---|---|
| `experiment_bpe_correction.py` | Single-layer BPE intra-token exclusion for ProtGPT2 sequential locality (the H2′ metric) |
| `experiment_bpe_crossing_all_depths.py` | Wrapper: runs the above at all 5 matched depths and aggregates H2′ |
| `experiment_val_only_h1h2.py` | Recomputes H1 and H2 on the 150-protein val subset (raw metric) |
| `experiment_h2prime_valonly.py` | Recomputes H2′ (inter-token) on val, using existing ESM-2 val CSV |
| `experiment_metric_sweep.py` | Joint sweep over `seq_gap_min` × `topk_frac` grid, with H1/H2′ per cell |
| `experiment_preflight.py` | 30 s sanity check for all cached artifacts before committing overnight compute |
| `experiment_expanded_annotations.py` | (Limitation 3) continuous RSA + UniProt functional-site probes |
| `experiment_stability.py` | (Limitations 1 + 5) cross-seed decoder cosine similarity + depth interpolation |
| `run_h2_robustness.sh` | Runs BPE-crossing on the 3 existing non-main runs (seed43/44/k128) |
| `run_overnight.sh` | Sequential chain: split99 full pipeline → BPE crossing on split99 → metric sweep |

### Residue-level causal comparator scripts (v6)
| File | Description |
|---|---|
| `experiment_h1h2_rita.py` | Clean H1/H2 test: ESM-2 vs RITA at 5 matched depths (no BPE correction needed) |
| `experiment_val_only_rita.py` | H1/H2 on the 150-protein val subset for ESM-2 vs RITA |
| `experiment_metric_sweep_rita.py` | 9-cell `seq_gap_min × topk_frac` sweep for ESM-2 vs RITA H1/H2 |
| `aggregate_rita_robustness.py` | Consolidates the 5-run RITA robustness table (main/seed43/seed44/k128/split99) |
| `smoke_test_rita.py` | End-to-end RITA integration check (tokenizer 1:1, model load, layer indexing) |
| `run_rita.sh` | Single-seed RITA pipeline (MODEL=rita only) |
| `run_rita_overnight.sh` | Full 5-variant robustness chain + val-only + metric sweep + aggregator |

### Densification scripts (v6 appendix)
| File | Description |
|---|---|
| `experiment_prott5_densify_check.py` | De-risk gate for the ProtT5 H3 densification (ProtT5-dec L9 val_EV + smoothness check) |
| `experiment_prott5_densify_analysis.py` | H3 enc-vs-dec on 9 probes per side; finds crossover, writes `H3_enc_vs_dec_dense.{csv,png,pdf,txt}` |
| `run_prott5_densify.sh` | Orchestrator: de-risk → gate → full dec → full enc → analysis + plot |
| `experiment_esm_rita_densify_check.py` | De-risk gate for the ESM-2/RITA H5 densification (ESM-2 L12) |
| `experiment_esm_rita_densify_analysis.py` | Bootstrap 95% CI on mean `struct_delta` per layer; writes `H5_within_model_dense.{csv,png,pdf,txt}` |
| `run_esm_rita_densify.sh` | Orchestrator: de-risk → gate → full ESM-2 → full RITA → analysis + plot |

### Utilities
| File | Description |
|---|---|
| `smoke_test.py` | Fast offline verification: tests helpers, SAE training on synthetic data, H5 path, CLI surfaces |
| `draft_paper.py` | Generates the workshop paper draft as a Word .docx file |
| `requirements.txt` | Frozen pip dependencies (Python 3.12, PyTorch 2.11, transformers 5.5) |

### Cached Data (tracked)
| File | Description |
|---|---|
| `cache/sequences.json` | 1,500 subsampled protein sequences (uid -> sequence dict) |
| `cache/residue_features.csv` | Per-residue DSSP labels + burial counts (295,240 rows) |
| `cache/scope_40.fa` | SCOPe 2.08 FASTA with fold annotations |
| `cache/dataset_summary.json` | Dataset statistics (N proteins, folds, length distribution) |

## SAE Hyperparameters

| Parameter | Value | Justification |
|---|---|---|
| Expansion factor | 8x | Ablation over {4, 8, 16, 32}x; 8x balances reconstruction and interpretability |
| k_sparse | 256 | Ablation over {64, 128, 256}; 256 achieves smallest holdout EV gap (0.092) |
| k_aux | 64 | Auxiliary loss for dead-latent recovery |
| Dead threshold | 1,000,000 tokens | ~4 epochs at 265k train tokens |
| Learning rate | 5e-5 | Cosine decay to 5e-6 over 60 epochs |
| Batch size | 4096 | Auto-tuned per device |
| Input normalization | Bricken (mean L2 -> sqrt(D)) | Required: PLM outlier features destabilize SAE training without it |
| Decoder constraint | Unit L2 norm per column | Re-normalized after each optimizer step |
| Tied initialization | Encoder = Decoder^T | Using `param.copy_()` (not `param.data =`, which breaks MPS) |

## Known Issues

- **MPS `.data =` bug**: PyTorch's MPS backend silently corrupts `nn.Linear` forward outputs when `Parameter.data` is reassigned to a new tensor (the deprecated `param.data = new_tensor` pattern). The Parameter reads back as bit-identical to CPU, but `F.linear` produces wrong results (~70 unit max diff). Fixed by using `param.copy_(new_tensor)` throughout `sae.py`.
- **Activation clamping on MPS**: The intervention hook mixes fp16 (ESM-2 model) with fp32 (SAE model) tensors, triggering an MPS matmul dtype assertion. Use `--device cpu` as a workaround.
- **PLM outlier features**: ESM-2/ProtGPT2/ProtT5 hidden states have outlier dimensions with magnitudes 50-100x the typical scale (cf. Dettmers et al., 2022). Without Bricken normalization, SAE training diverges (negative EV, loss U-curves).
- **RITA fp16 checkpoint**: transformers 5.x honours the checkpoint's saved dtype by default, and RITA_l ships fp16. Its custom attention code upcasts softmax to fp32 for numerical stability, producing an `att @ v` dtype mismatch (raises on CPU, silently NaNs middle blocks on MPS). `extract_rita_embeddings` loads with `torch_dtype=torch.float32` and `.float()` to force fp32 end-to-end.
- **RITA L3 high EV**: the H5 densification adds RITA layer 3, where val_EV = 0.9989. RITA SAE bases are cross-seed stable elsewhere, but single-seed L3 sits in a high-EV regime where the SAE basis is not uniquely identified. Aggregate `mean(struct_delta)` with 12k-feature bootstrap CI is robust, but feature-level claims at L3 should be cross-seed-verified before citing.
- **ProtT5 `H3_enc_vs_dec` vs `H4` naming**: the v5 paper labelled this "H4" (in `analyze_hypotheses.py` and `H4_enc_vs_dec.csv`); v6 reframes it as "H3" following the post-submission hypothesis numbering. The densified output is written to `H3_enc_vs_dec_dense.*` and lives alongside the v5 `H4_enc_vs_dec.*` rather than replacing it.

## Citation

If you use this code, please cite:

```
@inproceedings{anonymous2026sae-plm,
  title={Sparse Autoencoders Reveal How Training Objectives Shape Structural Representations in Protein Language Models},
  author={Anonymous},
  booktitle={ICML 2026 Workshop on Mechanistic Interpretability},
  year={2026}
}
```
