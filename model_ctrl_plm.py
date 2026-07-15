#!/usr/bin/env python3
"""
model_ctrl_plm.py — shared-backbone protein LM for the controlled MLM-vs-CLM experiment.

ONE modern transformer backbone (RMSNorm Pre-LN, RoPE, SwiGLU, no bias, weight-tied),
instantiated identically for both objectives. The ONLY difference between the MLM and CLM
models is `cfg.causal` (the attention mask) and how the training loss is built downstream —
layers, width, and init are shared. This is what isolates the training objective.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class PLMConfig:
    vocab_size: int = 25
    d_model: int = 480
    n_layers: int = 12
    n_heads: int = 20
    ffn_dim: int = 1280
    max_seq: int = 512
    causal: bool = False
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        dt = x.dtype
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return xf.to(dt) * self.weight.to(dt)


def build_rope(seq, head_dim, theta, device, dtype):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq, device=device).float()
    freqs = torch.outer(t, inv)               # (seq, head_dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)   # (seq, head_dim)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q, k, cos, sin):
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.h = cfg.n_heads
        self.dh = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x, cos, sin, attn_mask):
        B, T, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=2)
        q = q.view(B, T, self.h, self.dh).transpose(1, 2)
        k = k.view(B, T, self.h, self.dh).transpose(1, 2)
        v = v.view(B, T, self.h, self.dh).transpose(1, 2)
        q, k = apply_rope(q, k, cos, sin)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return self.proj(o.transpose(1, 2).reshape(B, T, D))


class SwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.w1 = nn.Linear(cfg.d_model, cfg.ffn_dim, bias=False)  # gate
        self.w3 = nn.Linear(cfg.d_model, cfg.ffn_dim, bias=False)  # up
        self.w2 = nn.Linear(cfg.ffn_dim, cfg.d_model, bias=False)  # down

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n1 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.n2 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn = SwiGLU(cfg)

    def forward(self, x, cos, sin, attn_mask):
        x = x + self.attn(self.n1(x), cos, sin, attn_mask)
        x = x + self.ffn(self.n2(x))
        return x


class PLM(nn.Module):
    def __init__(self, cfg: PLMConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok.weight  # weight tying
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def build_mask(self, attention_mask):
        # attention_mask: (B, T) 1=real, 0=pad -> additive float mask (B, 1, T, T)
        B, T = attention_mask.shape
        allow = attention_mask[:, None, None, :].to(torch.bool).expand(B, 1, T, T).clone()
        if self.cfg.causal:
            causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=attention_mask.device))
            allow = allow & causal[None, None]
        add = torch.zeros(B, 1, T, T, device=attention_mask.device, dtype=torch.float32)
        add.masked_fill_(~allow, float("-inf"))
        return add

    def forward(self, input_ids, attention_mask, return_hidden=False):
        B, T = input_ids.shape
        x = self.tok(input_ids)
        cos, sin = build_rope(T, self.cfg.d_model // self.cfg.n_heads,
                              self.cfg.rope_theta, input_ids.device, x.dtype)
        mask = self.build_mask(attention_mask).to(x.dtype)
        hiddens = []
        for blk in self.blocks:
            x = blk(x, cos, sin, mask)
            if return_hidden:
                hiddens.append(x)
        x = self.norm(x)
        logits = self.head(x)
        if return_hidden:
            return logits, hiddens
        return logits

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
