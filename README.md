# SAE-PLM: Sparse Autoencoders Reveal How Training Objectives Shape Structural Representations in Protein Language Models

TopK Sparse Autoencoders (Gao et al., 2024) trained on four protein language model architectures at five matched relative depths, comparing how bidirectional vs causal training objectives shape learned feature geometry.

## Key Results

| Hypothesis | Result | Effect size |
|---|---|---|
| **H1**: ESM-2 structural locality > ProtGPT2 | **5/5 depths supported** | Cohen's d = 0.6–1.9 |
| **H2** (raw): ProtGPT2 sequential locality > ESM-2 | **5/5 depths supported** | Cohen's d = 1.7–5.9 |
| **H2′** (BPE-corrected, inter-token): ESM-2 sequential locality > ProtGPT2 | **5/5 depths supported (direction reversed)** | Cohen's d = 1.5–6.1 |
| **H3**: ProtT5 encoder vs decoder | **Depth-dependent reversal** | Early: decoder wins; Late: encoder wins |
| Multi-seed reproducibility | d std < 0.05 across 3 seeds | SAE training essentially deterministic |
| k-robustness (k=128 vs k=256) | 10/10 H1/H2 contrasts preserved | Effect direction robust |
| Data-split robustness (split seed=99) | H1+H2′ direction preserved 5/5 | Per-depth d delta ≤ 0.07 |
| Metric-choice robustness (9 cells) | 45/45 H1 + 45/45 H2′ cells significant | Cohen's d consistent across seq_gap × topk_frac grid |
| Activation clamping (causal) | Ablation drops contact precision | Wilcoxon p = 0.006 |

H2 (raw) on residue-projected ProtGPT2 activations is **substantially a BPE tokenization artifact**: 50.0% of ±1/±2 residue neighbor pairs in the projection are bit-identical by construction, because every residue inside a BPE token shares the same activation vector. The corrected test (H2′) restricts sequential locality to *inter-token* neighbor pairs — the only regime comparable to residue-level ESM-2 — and reverses the direction at all five matched depths. See the **BPE-crossing control** section below.

## Models and Depth Matching

| Model | Architecture | Params | Layers probed (0%, 25%, 50%, 75%, 100%) |
|---|---|---|---|
| ESM-2 (t33) | Bidirectional encoder (MLM) | 650M | 0, 8, 16, 24, 32 |
| ProtGPT2 | Causal decoder (CLM) | 738M | 0, 9, 18, 27, 35 |
| ProtT5-enc | Bidirectional encoder (seq2seq) | ~1.2B | 0, 6, 12, 18, 23 |
| ProtT5-dec | Autoregressive decoder (seq2seq) | ~3B | 0, 6, 12, 18, 23 |

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
                                  - H1/H2 at matched relative depths
                                  - H3 (ProtT5 enc vs dec)
                                  - H5 depth trends
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

### Preflight check (30 s)

```bash
.venv/bin/python experiment_preflight.py
```

Validates that cache files, layer outputs, META.val_uids, ProtGPT2 BPE
round-trip, and all five matched-depth pairs are in place before
committing overnight compute.

### Smoke test

```bash
.venv/bin/python smoke_test.py
```

## Environment Variables

| Variable | Default | Used by | Description |
|---|---|---|---|
| `DEVICE` | auto-detect | `run_all.sh`, `run_unsupervised.py` | Compute device: `cuda`, `mps`, or `cpu` |
| `MODEL` | `all` | `run_all.sh`, `run_unsupervised.py` | Which PLM(s): `esm2`, `protgpt2`, `prott5_enc`, `prott5_dec`, or `all` |
| `SAE_SEED` | `42` | `run_all.sh`, `run_unsupervised.py` | SAE weight initialization seed |
| `K_SPARSE` | `256` | `run_all.sh`, `run_unsupervised.py` | Number of active latents per token |
| `EXPANSION` | `8` | `run_all.sh`, `run_unsupervised.py` | SAE expansion factor (hidden_dim = input_dim x expansion) |
| `RUN_SUFFIX` | `""` | `run_all.sh`, `run_unsupervised.py` | Output directory suffix (e.g., `_seed43`, `_k128`) |
| `SPLIT_SEED` | `42` | `run_unsupervised.py` | Protein-level 90/10 train/val split seed. Override to probe robustness to protein subset (set RUN_SUFFIX so outputs don't clobber the main run) |
| `SAE_BATCH` | `4096` | `train_sae.py` | SAE training batch size (MPS) |
| `SAE_CPU_BATCH` | `4096` | `train_sae.py` | SAE training batch size (CPU, >= 8 cores) |
| `ESM2_BATCH` | `32` | `extract_embeddings.py` | ESM-2 inference batch size (MPS) |
| `PROTT5_BATCH` | `4` | `extract_embeddings.py` | ProtT5 inference batch size (MPS) |
| `PROTGPT2_BATCH` | `16` | `extract_embeddings.py` | ProtGPT2 inference batch size (MPS) |
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
| `extract_embeddings.py` | Extracts hidden states from ESM-2, ProtGPT2, ProtT5-enc, ProtT5-dec at specified layers |
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
