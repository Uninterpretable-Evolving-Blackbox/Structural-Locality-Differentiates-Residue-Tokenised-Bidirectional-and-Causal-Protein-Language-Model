#!/usr/bin/env python3
"""
train_sae.py (A100/H100-Optimized)
==================================

Training utilities for Sparse Autoencoder with full GPU utilization.
Reference: "Scaling and evaluating sparse autoencoders" (Gao et al., 2024)

Optimizations:
- Mixed precision (bf16/fp16) for 2x speedup
- Large batch sizes (4096-8192) to saturate GPU
- Fused AdamW optimizer
- torch.compile for kernel fusion
- Efficient DataLoader with pinned memory
"""

import gc
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import GradScaler
import numpy as np
from typing import Dict, Optional

from sae import SparseAutoencoder


# ============================================================
#       INPUT NORMALIZATION  (Bricken / Anthropic recipe)
# ============================================================
#
# PLM hidden states (ESM-2, ProtGPT2, ProtT5) have OUTLIER FEATURES:
# a few specific dimensions with magnitudes ~50–100× the typical
# activation scale.  Feeding raw activations into a TopK SAE with
# unit-norm decoder columns produces unstable training (negative EV,
# loss U-curve, divergence after epoch ~20) because the optimizer
# can't fit huge outliers with bounded-norm atoms.
#
# Standard fix: rescale every token so the AVERAGE token L2 norm
# becomes √D.  EV is scale-invariant so this changes nothing about
# the metric, but it brings activations into the range Adam can
# handle without thrashing.
#
# We compute the scale ONCE on the train set and apply it to both
# train and val.  The scale is also returned so it can be saved in
# META.json for the cpu_stage / hypothesis test scripts.
#
# Reference: Bricken et al. 2023, "Towards Monosemanticity"; same
# preprocessing used in Anthropic, OpenAI, InterPLM, Goodfire SAE work.
def compute_norm_scale(X: np.ndarray) -> float:
    """Global scalar that rescales mean token L2 norm to sqrt(D)."""
    norms = np.linalg.norm(X, axis=1).astype(np.float64)
    mean_norm = float(norms.mean())
    if mean_norm == 0.0:
        return 1.0
    return float(np.sqrt(X.shape[1]) / mean_norm)


# ============================================================
#              AUTOCAST COMPATIBILITY
# ============================================================

def get_autocast_context(device: str, dtype: torch.dtype, enabled: bool):
    """Get autocast context manager compatible with all PyTorch versions."""
    if not enabled:
        # Return a dummy context manager
        import contextlib
        return contextlib.nullcontext()
    
    # Try new API first (PyTorch 2.0+)
    try:
        return torch.amp.autocast(device_type=device, dtype=dtype, enabled=enabled)
    except (TypeError, AttributeError):
        pass
    
    # Try older API
    try:
        from torch.cuda.amp import autocast
        return autocast(enabled=enabled)
    except:
        pass
    
    # Fallback: no autocast
    import contextlib
    return contextlib.nullcontext()


# ============================================================
#              HARDWARE CONFIGURATION
# ============================================================

_HW_CONFIG = None

def _get_hw_config(device: str) -> dict:
    """Auto-detect hardware and return optimal training config."""
    global _HW_CONFIG
    if _HW_CONFIG is not None:
        return _HW_CONFIG
    
    config = {
        "dtype": torch.float32,
        "use_amp": False,
        "batch_size": 256,
        "num_workers": 0,
        "pin_memory": False,
        "use_compile": False,
        "use_fused_optimizer": False,
    }

    # Beefy multi-core CPU (≥8 cores): bump batch size so BLAS gets a
    # large enough matmul to saturate all cores per step.  This is the
    # path the workshop pipeline uses because PyTorch MPS produces
    # divergent SAE training on Apple Silicon (verified empirically) —
    # we extract PLM embeddings on MPS but train SAEs on CPU.
    if device == "cpu":
        import os
        n_cores = os.cpu_count() or 1
        if n_cores >= 8:
            config["batch_size"] = int(os.environ.get("SAE_CPU_BATCH", 4096))
            print(f"🖥️  CPU ({n_cores} cores), batch_size={config['batch_size']} "
                  f"(BLAS will saturate cores per matmul)")
    
    if device == "cuda" and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / (1024**3)

        print(f"🖥️  GPU: {props.name} ({vram_gb:.1f} GB)")

        # Enable bf16 for Ampere+ (SM 8.0+)
        if props.major >= 8:
            config["dtype"] = torch.bfloat16
            config["use_amp"] = True
            print("✅ Using bfloat16 mixed precision")
        elif props.major >= 7:
            config["dtype"] = torch.float16
            config["use_amp"] = True
            print("✅ Using float16 mixed precision")

        # Batch sizes based on VRAM
        if vram_gb > 70:  # H100
            config["batch_size"] = 8192
            config["num_workers"] = 8
        elif vram_gb > 35:  # A100
            config["batch_size"] = 4096
            config["num_workers"] = 4
        elif vram_gb > 20:
            config["batch_size"] = 2048
            config["num_workers"] = 4
        elif vram_gb > 10:
            config["batch_size"] = 1024
            config["num_workers"] = 2
        else:
            config["batch_size"] = 512
            config["num_workers"] = 2

        config["pin_memory"] = True
        config["use_fused_optimizer"] = True

        # torch.compile for PyTorch 2.0+
        if hasattr(torch, 'compile'):
            config["use_compile"] = True
            print("✅ torch.compile available")

        # TF32 for tensor cores
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

        print(f"✅ Training batch size: {config['batch_size']}")

    elif device == "mps":
        print(f"🖥️  Apple Silicon (MPS backend)")
        # CRITICAL: default to fp32 on MPS.
        #   PyTorch's GradScaler is CUDA-only.  Mixed-precision training
        #   in fp16 on MPS therefore runs the backward pass with NO loss
        #   scaling, which causes gradient underflow/overflow and divergent
        #   SAE training (negative EV, recon loss climbing across epochs).
        #   We hit this empirically: ablation_k.py with fp16+autocast on
        #   MPS produced EV ≈ -2.7 at k=256.
        # The SAE is tiny (~52 MB at fp32) so the memory cost is trivial,
        # and Apple Silicon GPUs have ~equivalent fp32/fp16 throughput
        # for these ops.
        # Override (only after verifying stability) via SAE_PRECISION=fp16.
        import os
        precision = os.environ.get("SAE_PRECISION", "fp32").lower()
        if precision == "fp16":
            config["dtype"] = torch.float16
            config["use_amp"] = True
            print("⚠️  SAE_PRECISION=fp16 OVERRIDE — known unstable on MPS")
        else:
            config["dtype"] = torch.float32
            config["use_amp"] = False
            print("✅ Using fp32 (MPS-stable; no GradScaler available for fp16)")
        config["batch_size"] = int(os.environ.get("SAE_BATCH", 4096))
        config["num_workers"] = 0       # MPS + multiprocessing workers can hang
        config["pin_memory"] = False     # unified memory — no benefit
        config["use_compile"] = False    # torch.compile MPS support is limited
        config["use_fused_optimizer"] = False  # fused AdamW is CUDA-only
        print(f"✅ Training batch size: {config['batch_size']}")

    _HW_CONFIG = config
    return config


# ============================================================
#                   TRAINING FUNCTION
# ============================================================

def train_sae(
    embeddings: np.ndarray,
    input_dim: int,
    device: str = "cuda",
    lr: float = 5e-5,
    epochs: int = 100,
    batch_size: Optional[int] = None,
    expansion: int = 32,
    k_sparse: int = 1024,
    k_aux: int = 512,
    aux_coeff: float = 1/32,
    dead_threshold: int = 1_000_000,
    log_interval: int = 10,
    use_aux_loss: bool = True,
) -> SparseAutoencoder:
    """
    Train a Sparse Autoencoder with A100/H100 optimizations.
    
    Args:
        embeddings: (N, D) array of embeddings
        input_dim: embedding dimension D
        device: compute device
        lr: learning rate
        epochs: training epochs
        batch_size: override auto batch size
        expansion: hidden dimension multiplier
        k_sparse: number of active latents
        k_aux: auxiliary latents for dead recovery
        aux_coeff: auxiliary loss coefficient
        log_interval: logging frequency
        use_aux_loss: enable AuxK loss
        
    Returns:
        Trained SparseAutoencoder
    """
    config = _get_hw_config(device)
    
    if batch_size is None:
        batch_size = config["batch_size"]
    
    dtype = config["dtype"]
    use_amp = config["use_amp"]
    
    # Compute pre-bias as mean
    b_pre = torch.from_numpy(embeddings.mean(axis=0).astype(np.float32))
    
    # Defensive: clamp batch_size to half the dataset so we always get
    # at least 2 complete batches.  This guards against the
    # ZeroDivisionError that hits when batch_size > N (typical only in
    # smoke tests; production runs have hundreds of thousands of tokens).
    n_tokens_total = embeddings.shape[0]
    if n_tokens_total < batch_size * 2:
        new_batch = max(1, n_tokens_total // 2)
        print(f"⚠️  batch_size {batch_size} > N/2 ({n_tokens_total // 2}), "
              f"reducing to {new_batch} for this small dataset")
        batch_size = new_batch

    # Create DataLoader
    dataset = TensorDataset(torch.from_numpy(embeddings).float())
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
        drop_last=True,
        persistent_workers=config["num_workers"] > 0,
    )
    if len(loader) == 0:
        raise RuntimeError(
            f"DataLoader is empty: {n_tokens_total} tokens, batch_size={batch_size}, "
            f"drop_last=True. Need at least batch_size tokens."
        )
    
    # Initialize model
    model = SparseAutoencoder(
        input_dim=input_dim,
        expansion=expansion,
        k_sparse=k_sparse,
        k_aux=k_aux,
        aux_coeff=aux_coeff,
        dead_threshold=dead_threshold,
        tied_init=True,
    ).to(device)
    
    model.b_pre.copy_(b_pre.to(device))
    
    # Compile model
    if config["use_compile"]:
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("✅ Model compiled with torch.compile")
        except Exception as e:
            print(f"⚠️ torch.compile failed: {e}")
    
    # Optimizer with fused implementation
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=0.0,
        fused=(config["use_fused_optimizer"] and device == "cuda"),
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.1
    )
    
    # Mixed precision scaler (CUDA fp16 only — not supported on MPS)
    scaler = GradScaler() if use_amp and dtype == torch.float16 and device == "cuda" else None
    
    n_tokens = embeddings.shape[0]
    n_batches = len(loader)
    
    print(f"Training SAE: {n_tokens} tokens, {input_dim}D -> {model.hidden_dim}D")
    print(f"  k_sparse={k_sparse}, k_aux={k_aux}, expansion={expansion}, lr={lr}")
    print(f"  dead_threshold={dead_threshold:,} tokens (~{dead_threshold/max(n_tokens,1):.1f} epochs)")
    print(f"  batch_size={batch_size}, epochs={epochs}")
    print("-" * 60)
    
    model.train()
    
    for epoch in range(epochs):
        epoch_recon = 0.0
        epoch_aux = 0.0
        epoch_l0 = 0.0
        
        for batch_data in loader:
            batch = batch_data[0].to(device, non_blocking=True)
            
            # Mixed precision forward
            with get_autocast_context(device, dtype, use_amp):
                x_hat, z, z_pre = model(batch)
                losses = model.loss_fn(batch, x_hat, z, z_pre, include_aux=use_aux_loss)
            
            # Backward
            optimizer.zero_grad(set_to_none=True)
            
            if scaler is not None:
                scaler.scale(losses['total']).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                losses['total'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            # Maintain unit norm decoder
            model.normalize_decoder()
            
            # Update dead latent tracking
            if use_aux_loss:
                model._update_activation_tracking(z.detach())
            
            # Accumulate metrics
            epoch_recon += losses['recon'].item()
            epoch_aux += losses['aux'].item()
            epoch_l0 += (z > 0).float().sum(dim=1).mean().item()
        
        scheduler.step()
        
        # Logging
        if (epoch + 1) % log_interval == 0 or epoch == 0:
            avg_recon = epoch_recon / n_batches
            avg_aux = epoch_aux / n_batches
            avg_l0 = epoch_l0 / n_batches
            metrics = model.get_metrics(z)
            
            print(
                f"Epoch {epoch+1:3d}/{epochs} | "
                f"Recon: {avg_recon:.6f} | "
                f"Aux: {avg_aux:.6f} | "
                f"L0: {avg_l0:.1f} | "
                f"Dead: {metrics['pct_dead']:.1f}%"
            )
    
    print("-" * 60)
    print(f"Training complete. Final dead: {metrics['num_dead']}/{model.hidden_dim}")
    
    return model


# ============================================================
#                   EVALUATION FUNCTION
# ============================================================

@torch.no_grad()
def evaluate_sae(
    model: SparseAutoencoder,
    val_loader: DataLoader,
    device: str,
) -> Dict[str, float]:
    """Evaluate SAE on validation set."""
    config = _get_hw_config(device)
    
    model.eval()
    
    total_recon = 0.0
    total_l0 = 0.0
    total_samples = 0
    feature_activations = torch.zeros(model.hidden_dim, device=device)
    
    for batch_data in val_loader:
        batch = batch_data[0].to(device, non_blocking=True)
        batch_size = batch.shape[0]
        
        with get_autocast_context(device, config["dtype"], config["use_amp"]):
            x_hat, z, _ = model(batch)
            recon = F.mse_loss(x_hat, batch, reduction='sum')
        
        total_recon += recon.item()
        total_l0 += (z > 0).float().sum().item()
        feature_activations += (z > 0).float().sum(dim=0)
        total_samples += batch_size
    
    model.train()
    
    return {
        "mse": total_recon / (total_samples * model.input_dim),
        "l0": total_l0 / total_samples,
        "pct_dead_features": 100 * (feature_activations == 0).float().mean().item(),
    }


# ============================================================
#                FEATURE EXTRACTION
# ============================================================

@torch.no_grad()
def extract_sae_features(
    model: SparseAutoencoder,
    X: np.ndarray,
    device: str = "cuda",
    save_dir: Optional[str] = None,
    batch_size: Optional[int] = None,
) -> tuple:
    """
    Extract SAE features from embeddings.
    
    Returns:
        (Z, D) where Z is activations and D is decoder dictionary
    """
    from pathlib import Path
    from tqdm import tqdm
    
    config = _get_hw_config(device)
    
    if batch_size is None:
        batch_size = config["batch_size"] * 2  # Can use larger for inference
    
    # Get the base model (unwrap if compiled)
    base_model = model
    if hasattr(model, '_orig_mod'):
        base_model = model._orig_mod
    
    base_model.eval()
    
    # Simple DataLoader without workers (avoid hanging)
    dataset = TensorDataset(torch.from_numpy(X).float())
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Avoid multiprocessing issues
        pin_memory=False,
    )
    
    Z_chunks = []
    n_batches = len(loader)
    
    print(f"   Extracting {len(X)} tokens in {n_batches} batches...")
    
    with torch.no_grad():
        for batch_data in tqdm(loader, desc="   Features", leave=False):
            batch = batch_data[0].to(device)
            
            # Simple forward without autocast for stability
            z, _ = base_model.encode(batch)
            
            Z_chunks.append(z.cpu().to(torch.float16).numpy())
    
    Z = np.concatenate(Z_chunks, axis=0)
    D = base_model.decoder.weight.detach().cpu().numpy().T.astype(np.float16)
    
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        np.save(save_dir / "Z.npy", Z)
        np.save(save_dir / "D.npy", D)
        torch.save(base_model.state_dict(), save_dir / "sae_model.pt")
    
    return Z, D


# ============================================================
#               EXPLAINED VARIANCE
# ============================================================

@torch.no_grad()
def compute_explained_variance(
    model: SparseAutoencoder,
    embeddings: np.ndarray,
    device: str = "cuda",
    batch_size: int = 4096,
) -> float:
    """Compute R^2 score for SAE reconstruction."""
    config = _get_hw_config(device)
    
    model.eval()
    
    dataset = TensorDataset(torch.from_numpy(embeddings).float())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    total_ss = 0.0
    residual_ss = 0.0
    mean = torch.from_numpy(embeddings.mean(axis=0)).to(device)
    
    for batch_data in loader:
        batch = batch_data[0].to(device, non_blocking=True)
        
        with get_autocast_context(device, config["dtype"], config["use_amp"]):
            x_hat, _, _ = model(batch)
        
        residual_ss += ((batch - x_hat) ** 2).sum().item()
        total_ss += ((batch - mean) ** 2).sum().item()
    
    return 1.0 - (residual_ss / total_ss)