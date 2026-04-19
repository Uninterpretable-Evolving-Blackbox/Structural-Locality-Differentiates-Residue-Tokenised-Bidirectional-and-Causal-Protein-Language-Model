#!/usr/bin/env python3
"""
sae.py (A100/H100-Optimized)
============================

Sparse Autoencoder with TopK activation (OpenAI methodology).
Reference: "Scaling and evaluating sparse autoencoders" (Gao et al., 2024)

Optimizations:
- Efficient TopK with scatter operations
- AuxK loss for dead latent recovery
- Unit norm constraint on decoder
- Compatible with mixed precision training
"""

import torch
from torch import nn
import torch.nn.functional as F
from typing import Dict, Tuple


class SparseAutoencoder(nn.Module):
    """
    TopK Sparse Autoencoder optimized for A100/H100.
    """
    
    def __init__(
        self,
        input_dim: int,
        expansion: int = 32,
        k_sparse: int = 64,
        k_aux: int = 512,
        aux_coeff: float = 1/32,
        dead_threshold: int = 10_000_000,
        tied_init: bool = True,
    ):
        """
        Args:
            input_dim: embedding dimension
            expansion: expansion factor for hidden dimension
            k_sparse: number of active latents per sample
            k_aux: number of dead latents for auxiliary loss
            aux_coeff: coefficient for auxiliary loss
            dead_threshold: tokens since activation to consider latent dead
            tied_init: initialize encoder as decoder.T
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = input_dim * expansion
        self.k_sparse = k_sparse
        self.k_aux = k_aux
        self.aux_coeff = aux_coeff
        self.dead_threshold = dead_threshold
        
        # Decoder: (hidden_dim, input_dim)
        self.decoder = nn.Linear(self.hidden_dim, input_dim, bias=False)
        
        # Encoder: (input_dim, hidden_dim)
        self.encoder = nn.Linear(input_dim, self.hidden_dim, bias=True)
        
        # Pre-bias (learned mean subtraction)
        self.register_buffer("b_pre", torch.zeros(input_dim))
        
        # Dead latent tracking
        self.register_buffer(
            "tokens_since_activation",
            torch.zeros(self.hidden_dim, dtype=torch.long)
        )
        self.register_buffer("total_tokens_seen", torch.tensor(0, dtype=torch.long))
        
        self._init_weights(tied_init)
    
    def _init_weights(self, tied_init: bool):
        """
        Initialize weights following OpenAI methodology.

        CRITICAL: must use `param.copy_(new_tensor)` (or in-place ops on
        `param`), NOT `param.data = new_tensor`.  The .data reassignment
        pattern is deprecated PyTorch and silently corrupts Parameter
        bindings on the MPS backend.  Symptom: after `model.to('mps')`,
        `model.encoder.weight` reads back as bit-identical to the CPU
        version, but `model.encoder(x)` (i.e. `F.linear`) produces wildly
        wrong outputs (max diff ~70 units on a typical activation), while
        the manual `x @ weight.T + bias` route still produces correct
        results.  This is invisible until you actually try to train.
        """
        # Random unit-norm decoder columns
        nn.init.xavier_uniform_(self.decoder.weight)
        with torch.no_grad():
            self.decoder.weight.copy_(F.normalize(self.decoder.weight, dim=0))

        if tied_init:
            # Encoder = Decoder^T
            with torch.no_grad():
                self.encoder.weight.copy_(self.decoder.weight.T)
        else:
            nn.init.xavier_uniform_(self.encoder.weight)

        nn.init.zeros_(self.encoder.bias)
    
    def _topk_activation(self, z_pre: torch.Tensor) -> torch.Tensor:
        """
        TopK activation via torch.where threshold mask.

        ORIGINAL implementation used `torch.zeros_like` + in-place
        `scatter_`, which is buggy on PyTorch's MPS backend: the gradient
        through scatter into a fresh leaf tensor accumulates numerical
        drift over thousands of steps and produces divergent SAE training
        on Apple Silicon (verified empirically on ESM-2 layer 16).

        torch.where + masking is purely elementwise, well-tested on every
        backend, and produces an identical mathematical result for TopK +
        ReLU activation:
            z[b, j] = ReLU(z_pre[b, j])  if z_pre[b, j] is in top-k of row b
                                          else 0
        """
        k = min(self.k_sparse, z_pre.shape[1])

        # k-th largest value per row.  detach() because TopK indices are
        # not differentiable, and the threshold itself is just a selector.
        kth = torch.topk(z_pre, k, dim=1).values[:, -1:].detach()

        # Mask = positions ≥ threshold AND positive (after ReLU).
        # The gradient flows cleanly through z_pre via torch.where.
        return torch.where(
            (z_pre >= kth) & (z_pre > 0),
            z_pre,
            torch.zeros_like(z_pre),
        )
    
    def _get_dead_latent_mask(self) -> torch.Tensor:
        """Returns boolean mask of dead latents."""
        return self.tokens_since_activation >= self.dead_threshold
    
    @torch.no_grad()
    def _update_activation_tracking(self, z: torch.Tensor):
        """Update tracking of latent activations."""
        batch_size = z.shape[0]
        fired = (z > 0).any(dim=0)
        
        self.tokens_since_activation[fired] = 0
        self.tokens_since_activation[~fired] += batch_size
        self.total_tokens_seen += batch_size
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to sparse representation.
        
        Returns:
            z: sparse activations after TopK
            z_pre: pre-activation values (for AuxK)
        """
        x_centered = x - self.b_pre
        z_pre = self.encoder(x_centered)
        z = self._topk_activation(z_pre)
        return z, z_pre
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode sparse latents."""
        return self.decoder(z) + self.b_pre
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            x_hat: reconstruction
            z: sparse activations
            z_pre: pre-activations
        """
        z, z_pre = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z, z_pre
    
    def compute_aux_loss(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        z_pre: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute AuxK loss for dead latent recovery.
        
        Models reconstruction error using top-k_aux DEAD latents only.
        """
        dead_mask = self._get_dead_latent_mask()
        num_dead = dead_mask.sum().item()
        
        if num_dead == 0:
            return torch.tensor(0.0, device=x.device, dtype=x.dtype)
        
        error = x - x_hat
        
        # Mask out alive latents
        z_pre_dead = z_pre.clone()
        z_pre_dead[:, ~dead_mask] = float('-inf')
        
        # TopK over dead latents
        k_aux = min(self.k_aux, num_dead)
        topk_vals, topk_idx = torch.topk(z_pre_dead, k_aux, dim=1)
        
        # Build sparse activation from dead latents
        z_aux = torch.zeros_like(z_pre)
        z_aux.scatter_(1, topk_idx, F.relu(topk_vals))
        
        # Reconstruct error using dead latents
        error_hat = self.decoder(z_aux)
        
        return F.mse_loss(error_hat, error)
    
    def loss_fn(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        z: torch.Tensor,
        z_pre: torch.Tensor,
        include_aux: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute losses.
        
        Returns:
            dict with 'total', 'recon', 'aux' losses
        """
        recon_loss = F.mse_loss(x_hat, x)
        
        if include_aux and self.training:
            aux_loss = self.compute_aux_loss(x, x_hat, z_pre)
        else:
            aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        
        total_loss = recon_loss + self.aux_coeff * aux_loss
        
        return {
            'total': total_loss,
            'recon': recon_loss,
            'aux': aux_loss,
        }
    
    @torch.no_grad()
    def normalize_decoder(self):
        """
        Normalize decoder columns to unit L2 norm, in-place.

        ORIGINAL implementation reassigned `weight.data = F.normalize(...)`,
        which is a deprecated PyTorch anti-pattern that produces a NEW
        tensor and rebinds `.data`.  On MPS this can cause subtle storage
        / view drift that compounds over thousands of optimizer steps and
        manifests as divergent SAE training.

        Idiomatic fix: in-place division by per-column norm, with eps to
        avoid div-by-zero on dead atoms.
        """
        norms = self.decoder.weight.norm(dim=0, keepdim=True).clamp_min(1e-8)
        self.decoder.weight.div_(norms)
    
    @torch.no_grad()
    def get_metrics(self, z: torch.Tensor) -> Dict[str, float]:
        """Compute sparsity metrics."""
        l0 = (z > 0).float().sum(dim=1).mean().item()
        dead_mask = self._get_dead_latent_mask()
        pct_dead = 100.0 * dead_mask.float().mean().item()
        
        return {
            'l0': l0,
            'pct_dead': pct_dead,
            'num_dead': dead_mask.sum().item(),
        }