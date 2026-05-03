#!/usr/bin/env python3
"""
extract_embeddings.py (A100/H100-Optimized) - FIXED LAYER INDEXING
===================================================================

Unified extractors for protein language models:
- ESM-2 encoder (facebook/esm2_t33_650M_UR50D)
- ProtT5 encoder OR decoder (Rostlab/prot_t5_xl_uniref50)
- ProtGPT2 (nferruz/ProtGPT2)

FIX APPLIED: All models now use consistent layer indexing where:
- Layer 0 = output of first transformer block (NOT embedding)
- Layer N = output of (N+1)th transformer block
- Final layer = total_blocks - 1

This matches ESM-2's original convention and ensures cross-model comparisons
are at equivalent relative depths.

Model specs:
- ESM-2 (650M): 33 transformer blocks → layers 0-32
- ProtGPT2: 36 transformer blocks → layers 0-35
- ProtT5-XL: 24 transformer blocks (each) → layers 0-23

A100/H100 Optimizations:
- Mixed precision (bf16) for 2x throughput
- Large batch sizes to saturate GPU
- TF32 tensor core acceleration
- Flash Attention 2 where available
- torch.compile for kernel fusion
"""

import gc
import os
import torch
import numpy as np
from typing import Dict, List, Optional
from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoModel,
    T5Tokenizer,
    T5EncoderModel,
    T5ForConditionalGeneration,
    AutoModelForCausalLM,
)

# ============================================================
#              HARDWARE CONFIGURATION
# ============================================================

_HW_CONFIG = None

def _get_hw_config(device: str) -> dict:
    """Auto-detect hardware and return optimal config."""
    global _HW_CONFIG
    if _HW_CONFIG is not None:
        return _HW_CONFIG
    
    config = {
        "dtype": torch.float32,
        "use_amp": False,
        "esm2_batch": 8,
        "prott5_batch": 1,
        "protgpt2_batch": 4,
        "rita_batch": 4,
        "progen2_batch": 4,
        "use_flash_attn": False,
    }
    
    if device == "cuda" and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / (1024**3)

        print(f"🖥️  GPU: {props.name} ({vram_gb:.1f} GB)")

        # Enable bf16 for Ampere+ (SM 8.0+)
        if props.major >= 8:
            config["dtype"] = torch.bfloat16
            config["use_amp"] = True
        elif props.major >= 7:
            config["dtype"] = torch.float16
            config["use_amp"] = True

        # Batch sizes based on VRAM
        if vram_gb > 70:  # H100 80GB
            config["esm2_batch"] = 64
            config["prott5_batch"] = 8
            config["protgpt2_batch"] = 32
            config["rita_batch"] = 24
            config["progen2_batch"] = 24
        elif vram_gb > 35:  # A100 40/80GB
            config["esm2_batch"] = 32
            config["prott5_batch"] = 4
            config["protgpt2_batch"] = 16
            config["rita_batch"] = 12
            config["progen2_batch"] = 12
        elif vram_gb > 20:  # RTX 3090/4090
            config["esm2_batch"] = 16
            config["prott5_batch"] = 2
            config["protgpt2_batch"] = 8
            config["rita_batch"] = 6
            config["progen2_batch"] = 6
        elif vram_gb > 10:
            config["esm2_batch"] = 8
            config["prott5_batch"] = 1
            config["protgpt2_batch"] = 4
            config["rita_batch"] = 3
            config["progen2_batch"] = 3

        # TF32 for tensor cores
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

        # Check flash attention
        try:
            from transformers.utils import is_flash_attn_2_available
            if is_flash_attn_2_available():
                config["use_flash_attn"] = True
                print("✅ Flash Attention 2 enabled")
        except:
            pass

    elif device == "mps":
        print(f"🖥️  Apple Silicon (MPS backend)")
        # fp16 model loading for 2x memory savings; autocast is
        # automatically skipped for non-CUDA devices (see _get_autocast_context)
        config["dtype"] = torch.float16
        config["use_amp"] = True  # triggers fp16 model loading only
        # Tuned for Apple Silicon Ultra / Max with ≥64 GB unified memory.
        # Override via {ESM2,PROTT5,PROTGPT2}_BATCH env vars if hitting OOM.
        config["esm2_batch"] = int(os.environ.get("ESM2_BATCH", 32))
        config["prott5_batch"] = int(os.environ.get("PROTT5_BATCH", 4))
        config["protgpt2_batch"] = int(os.environ.get("PROTGPT2_BATCH", 16))
        config["rita_batch"] = int(os.environ.get("RITA_BATCH", 12))
        config["progen2_batch"] = int(os.environ.get("PROGEN2_BATCH", 12))
        config["use_flash_attn"] = False
        print(f"   Batches: esm2={config['esm2_batch']} "
              f"prott5={config['prott5_batch']} protgpt2={config['protgpt2_batch']} "
              f"rita={config['rita_batch']} progen2={config['progen2_batch']}")

    _HW_CONFIG = config
    return config


def _clear_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _get_autocast_context(device: str, dtype: torch.dtype, enabled: bool):
    """Get autocast context compatible with all PyTorch versions."""
    if not enabled or device != "cuda":
        import contextlib
        return contextlib.nullcontext()
    
    try:
        # PyTorch 2.0+ API
        return torch.amp.autocast(device_type=device, dtype=dtype, enabled=enabled)
    except (TypeError, AttributeError):
        pass
    
    try:
        # Older API
        return torch.cuda.amp.autocast(enabled=enabled)
    except:
        pass
    
    import contextlib
    return contextlib.nullcontext()


def _get_device(device: Optional[str] = None) -> str:
    if device:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().float().numpy()


def _keep_mask(input_ids: torch.Tensor, attention_mask: torch.Tensor, tokenizer) -> torch.Tensor:
    """Returns boolean mask of positions to KEEP (exclude specials/pad)."""
    ids = input_ids[0] if input_ids.dim() > 1 else input_ids
    attn = attention_mask[0].bool() if attention_mask.dim() > 1 else attention_mask.bool()
    
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    for attr in ("pad_token_id", "eos_token_id", "bos_token_id", 
                 "cls_token_id", "sep_token_id", "mask_token_id"):
        tid = getattr(tokenizer, attr, None)
        if tid is not None:
            special_ids.add(int(tid))
    
    if len(special_ids) == 0:
        return attn
    
    specials_mask = torch.zeros_like(ids, dtype=torch.bool)
    for sid in special_ids:
        specials_mask |= (ids == sid)
    
    keep = attn & (~specials_mask)
    if keep.sum().item() == 0:
        keep = attn
    return keep


# ============================================================
#                    ESM-2 EXTRACTOR
# ============================================================

def extract_esm2_embeddings(
    protein_sequences: List[str],
    layers: List[int],
    device: Optional[str] = None,
    batch_size: Optional[int] = None,
    max_length: int = 1024,
    model_name: str = "facebook/esm2_t33_650M_UR50D",
) -> Dict[int, np.ndarray]:
    """
    Extract ESM-2 embeddings at specified layers.

    Layer indexing: layer N = output of transformer block N+1
      - Layer 0 = first transformer block output
      - Layer (n_blocks-1) = final transformer block output

    Model scales (for PLM-scale robustness):
      facebook/esm2_t6_8M_UR50D    (6 blocks,  8M params)
      facebook/esm2_t12_35M_UR50D  (12 blocks, 35M params)
      facebook/esm2_t30_150M_UR50D (30 blocks, 150M params)
      facebook/esm2_t33_650M_UR50D (33 blocks, 650M params)   [paper default]
      facebook/esm2_t36_3B_UR50D   (36 blocks, 3B params)
    """
    if not layers:
        raise ValueError("Must request at least one ESM-2 layer.")

    layers = sorted(set(int(layer) for layer in layers))
    device = _get_device(device)
    config = _get_hw_config(device)

    if batch_size is None:
        batch_size = config["esm2_batch"]

    print(f"Loading ESM-2 ({model_name}) to {device} (batch={batch_size}, amp={config['use_amp']})...")
    print(f"  Layers requested: {layers}")

    model_kwargs = {}
    if config["use_flash_attn"]:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    if config["use_amp"]:
        model_kwargs["torch_dtype"] = config["dtype"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, **model_kwargs).to(device).eval()
    
    layer_buffers: Dict[int, List[np.ndarray]] = {layer: [] for layer in layers}
    batches = [protein_sequences[i:i + batch_size] for i in range(0, len(protein_sequences), batch_size)]
    
    print(f"Extracting {len(protein_sequences)} sequences in {len(batches)} batches...")
    
    with torch.no_grad():
        for batch_seqs in tqdm(batches, desc="ESM-2"):
            tokens = tokenizer(
                batch_seqs, return_tensors="pt", add_special_tokens=True,
                padding=True, truncation=True, max_length=max_length,
            ).to(device)
            
            with _get_autocast_context(device, config["dtype"], config["use_amp"]):
                outputs = model(**tokens, output_hidden_states=True, return_dict=True)
            hidden_states = outputs.hidden_states  # Tuple of (embedding, block1, block2, ..., block33)
            
            for i in range(len(batch_seqs)):
                keep = _keep_mask(tokens["input_ids"][i:i+1], tokens["attention_mask"][i:i+1], tokenizer)
                for layer in layers:
                    # hidden_states[0] = embedding, hidden_states[1] = block 1, ..., hidden_states[33] = block 33
                    # So layer N (block N+1) is at index N+1
                    idx = layer + 1
                    if idx >= len(hidden_states):
                        raise ValueError(f"ESM-2: requested layer {layer}, only {len(hidden_states)-1} blocks available.")
                    layer_buffers[layer].append(_to_numpy(hidden_states[idx][i, keep, :]))
            
            del outputs, hidden_states
    
    del model
    _clear_cache()
    return {layer: np.vstack(chunks) for layer, chunks in layer_buffers.items()}


# ============================================================
#                 PROTT5 ENCODER EXTRACTOR
# ============================================================

def extract_prott5_encoder_embeddings(
    sequences: List[str],
    layers: List[int],
    device: Optional[str] = None,
    batch_size: Optional[int] = None,
) -> Dict[int, np.ndarray]:
    """
    Extract ProtT5 encoder embeddings.
    
    Layer indexing: layer N = output of transformer block N+1
    - Layer 0 = first transformer block output
    - Layer 23 = final transformer block output (24th block)
    
    ProtT5-XL encoder has 24 transformer blocks, so valid layers are 0-23.
    
    FIXED: Now uses +1 offset to skip embedding layer, matching ESM-2 convention.
    """
    if not layers:
        raise ValueError("Must request at least one ProtT5 encoder layer.")
    
    layers = sorted(set(int(layer) for layer in layers))
    device = _get_device(device)
    config = _get_hw_config(device)
    
    if batch_size is None:
        batch_size = config["prott5_batch"]
    
    model_name = "Rostlab/prot_t5_xl_half_uniref50-enc"
    print(f"Loading ProtT5 encoder to {device} (batch={batch_size}, amp={config['use_amp']})...")
    print(f"  Layers requested: {layers} (ProtT5 encoder has 24 blocks, indices 0-23)")
    
    tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=True)
    model_kwargs = {"torch_dtype": config["dtype"]} if config["use_amp"] else {}
    model = T5EncoderModel.from_pretrained(model_name, **model_kwargs).to(device).eval()
    
    buffers: Dict[int, List[np.ndarray]] = {layer: [] for layer in layers}
    batches = [sequences[i:i + batch_size] for i in range(0, len(sequences), batch_size)]
    
    with torch.no_grad():
        for batch_seqs in tqdm(batches, desc="ProtT5-Enc"):
            texts = [" ".join(seq) for seq in batch_seqs]
            toks = tokenizer(texts, return_tensors="pt", add_special_tokens=False, 
                           padding=True, truncation=True, max_length=1024).to(device)
            
            with _get_autocast_context(device, config["dtype"], config["use_amp"]):
                out = model(**toks, output_hidden_states=True, return_dict=True)
            hidden_states = out.hidden_states  # Tuple of (embedding, block1, ..., block24)
            
            for i in range(len(batch_seqs)):
                keep = toks["attention_mask"][i].bool()
                for layer in layers:
                    # FIXED: Add +1 to skip embedding layer
                    # hidden_states[0] = embedding, hidden_states[1] = block 1, ..., hidden_states[24] = block 24
                    idx = layer + 1
                    if idx >= len(hidden_states):
                        raise ValueError(f"ProtT5 encoder: requested layer {layer}, only {len(hidden_states)-1} blocks available.")
                    buffers[layer].append(_to_numpy(hidden_states[idx][i, keep, :]))
            
            del out, hidden_states
    
    del model
    _clear_cache()
    return {layer: np.vstack(chunks) for layer, chunks in buffers.items()}


# ============================================================
#                 PROTT5 DECODER EXTRACTOR
# ============================================================

def extract_prott5_decoder_embeddings(
    sequences: List[str],
    layers: List[int],
    device: Optional[str] = None,
    batch_size: Optional[int] = None,
) -> Dict[int, np.ndarray]:
    """
    Extract ProtT5 decoder embeddings.
    
    Layer indexing: layer N = output of transformer block N+1
    - Layer 0 = first transformer block output
    - Layer 23 = final transformer block output (24th block)
    
    ProtT5-XL decoder has 24 transformer blocks, so valid layers are 0-23.
    
    FIXED: Now uses +1 offset to skip embedding layer, matching ESM-2 convention.
    """
    if not layers:
        raise ValueError("Must request at least one ProtT5 decoder layer.")
    
    layers = sorted(set(int(layer) for layer in layers))
    device = _get_device(device)
    config = _get_hw_config(device)
    
    model_name = "Rostlab/prot_t5_xl_uniref50"
    print(f"Loading ProtT5 decoder to {device} (amp={config['use_amp']})...")
    print(f"  Layers requested: {layers} (ProtT5 decoder has 24 blocks, indices 0-23)")
    
    tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=True)
    model_kwargs = {"torch_dtype": config["dtype"]} if config["use_amp"] else {}
    model = T5ForConditionalGeneration.from_pretrained(model_name, **model_kwargs).to(device).eval()
    
    bos_id = tokenizer.pad_token_id
    buffers: Dict[int, List[np.ndarray]] = {layer: [] for layer in layers}
    
    with torch.no_grad():
        for seq in tqdm(sequences, desc="ProtT5-Dec"):
            text = " ".join(seq)
            toks = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
            enc_ids, enc_mask = toks["input_ids"], toks["attention_mask"]
            
            dec_ids = torch.full((1, 1), bos_id, dtype=enc_ids.dtype, device=device)
            if enc_ids.shape[1] > 1:
                dec_ids = torch.cat([dec_ids, enc_ids[:, :-1]], dim=1)
            dec_mask = torch.ones_like(dec_ids)
            
            with _get_autocast_context(device, config["dtype"], config["use_amp"]):
                outputs = model(input_ids=enc_ids, attention_mask=enc_mask,
                              decoder_input_ids=dec_ids, decoder_attention_mask=dec_mask,
                              output_hidden_states=True, return_dict=True)
            
            hidden_states = outputs.decoder_hidden_states  # Tuple of (embedding, block1, ..., block24)
            keep = dec_mask.bool()[0]
            
            for layer in layers:
                # FIXED: Add +1 to skip embedding layer
                idx = layer + 1
                if idx >= len(hidden_states):
                    raise ValueError(f"ProtT5 decoder: requested layer {layer}, only {len(hidden_states)-1} blocks available.")
                buffers[layer].append(_to_numpy(hidden_states[idx][0, keep, :]))
            
            del outputs, hidden_states
    
    del model
    _clear_cache()
    return {layer: np.vstack(chunks) for layer, chunks in buffers.items()}


# ============================================================
#                   PROTGPT2 EXTRACTOR
# ============================================================

def extract_protgpt2_embeddings(
    sequences: List[str],
    layers: List[int],
    device: Optional[str] = None,
    batch_size: Optional[int] = None,
    max_length: int = 1024,
) -> Dict[int, np.ndarray]:
    """
    Extract ProtGPT2 embeddings. WARNING: BPE tokens != residues.
    
    Layer indexing: layer N = output of transformer block N+1
    - Layer 0 = first transformer block output
    - Layer 35 = final transformer block output (36th block)
    
    ProtGPT2 has 36 transformer blocks, so valid layers are 0-35.
    
    FIXED: Now uses +1 offset to skip embedding layer, matching ESM-2 convention.
    """
    if not layers:
        raise ValueError("Must request at least one ProtGPT2 layer.")
    
    layers = sorted(set(int(layer) for layer in layers))
    device = _get_device(device)
    config = _get_hw_config(device)
    
    if batch_size is None:
        batch_size = config["protgpt2_batch"]
    
    print(f"Loading ProtGPT2 to {device} (batch={batch_size}, amp={config['use_amp']})...")
    print(f"  Layers requested: {layers} (ProtGPT2 has 36 blocks, indices 0-35)")
    
    tokenizer = AutoTokenizer.from_pretrained("nferruz/ProtGPT2")
    model_kwargs = {"torch_dtype": config["dtype"]} if config["use_amp"] else {}
    model = AutoModelForCausalLM.from_pretrained("nferruz/ProtGPT2", **model_kwargs).to(device).eval()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    buffers: Dict[int, List[np.ndarray]] = {layer: [] for layer in layers}
    batches = [sequences[i:i + batch_size] for i in range(0, len(sequences), batch_size)]
    
    with torch.no_grad():
        for batch_seqs in tqdm(batches, desc="ProtGPT2"):
            inputs = tokenizer(batch_seqs, return_tensors="pt", padding=True,
                             truncation=True, max_length=max_length)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with _get_autocast_context(device, config["dtype"], config["use_amp"]):
                out = model(**inputs, output_hidden_states=True, return_dict=True)
            hidden_states = out.hidden_states  # Tuple of (embedding, block1, ..., block36)
            
            for i in range(len(batch_seqs)):
                keep = inputs["attention_mask"][i].bool()
                for layer in layers:
                    # FIXED: Add +1 to skip embedding layer
                    # hidden_states[0] = embedding, hidden_states[1] = block 1, ..., hidden_states[36] = block 36
                    idx = layer + 1
                    if idx >= len(hidden_states):
                        raise ValueError(f"ProtGPT2: requested layer {layer}, only {len(hidden_states)-1} blocks available.")
                    buffers[layer].append(_to_numpy(hidden_states[idx][i, keep, :]))
            
            del out, hidden_states
    
    del model
    _clear_cache()
    return {layer: np.vstack(chunks) for layer, chunks in buffers.items()}


# ============================================================
#                   PROGEN2 EXTRACTOR
# ============================================================

def _patch_progen2_meta_tensors(model, device: str):
    """Materialise meta-device tensors in ProGen2's custom modelling code.

    ProGen2's ProGenAttention.__init__ creates three tensors that are NOT
    regular parameters or persistent buffers:
      • ``scale_attn``  — a Python attribute holding sqrt(head_dim)
      • ``masked_bias`` — ``register_buffer(..., persistent=False)``
      • ``bias``        — ``register_buffer(..., persistent=False)``
    Under transformers 5.x's meta-device init context, these end up on meta
    and are not moved by ``model.to(device)``.  We rebuild them here with
    their original init values.
    """
    fixed = 0
    for sub in model.modules():
        # scale_attn is a Python attribute (not a registered parameter/buffer)
        if hasattr(sub, "scale_attn") and isinstance(sub.scale_attn, torch.Tensor):
            if sub.scale_attn.device.type == "meta":
                sub.scale_attn = torch.sqrt(torch.tensor(
                    sub.head_dim, dtype=torch.float32)).to(torch.get_default_dtype())
                fixed += 1
        if hasattr(sub, "masked_bias") and isinstance(sub.masked_bias, torch.Tensor):
            if sub.masked_bias.device.type == "meta":
                sub.register_buffer("masked_bias",
                                    torch.tensor(-1e9),
                                    persistent=False)
                fixed += 1
        # Causal mask buffer (name 'bias' is unfortunate — matches Python
        # nn.Module.bias attribute on nn.Linear, so be careful).  ProGen2
        # puts this directly on the attention block, not on a Linear.
        if (hasattr(sub, "bias") and isinstance(sub.bias, torch.Tensor)
                and sub.bias.dtype == torch.bool and sub.bias.dim() == 4):
            if sub.bias.device.type == "meta":
                mp = sub.bias.shape[-1]
                sub.register_buffer("bias",
                    torch.tril(torch.ones((mp, mp), dtype=torch.bool))
                         .view(1, 1, mp, mp),
                    persistent=False)
                fixed += 1
    return fixed


def extract_progen2_embeddings(
    sequences: List[str],
    layers: List[int],
    device: Optional[str] = None,
    batch_size: Optional[int] = None,
    max_length: int = 1024,
    model_name: str = "hugohrban/progen2-medium",
) -> Dict[int, np.ndarray]:
    """
    Extract ProGen2 embeddings.  Residue-level autoregressive PLM (Nijkamp
    et al. 2022): 1 token per amino acid, no BPE merging.

    Matching ESM-2 at residue granularity means the sequential-locality
    metric is directly comparable without the inter-token correction that
    ProtGPT2's BPE requires.

    ProGen2-medium: 764M params, 27 transformer blocks.  Matched relative
    depths for cross-model comparison: [0, 7, 14, 20, 26].

    Implementation notes — three transformers-5.x compatibility patches are
    applied, the RITA-learned pattern:
      (1) ``PreTrainedModel.all_tied_weights_keys = {}`` so the load path
          doesn't crash looking for an attribute the custom class lacks.
      (2) ``PreTrainedModel.get_head_mask`` stub — removed from
          transformers 5.x but ProGen2's forward still calls it.
      (3) Rebuild per-attention-block meta tensors (``scale_attn``,
          ``masked_bias``, ``bias``) that transformers-5.x's meta-init
          leaves stranded.  See ``_patch_progen2_meta_tensors``.

    Unlike RITA, ProGen2 honours ``output_hidden_states=True`` natively
    (returns the full (embedding + N_blocks) tuple), so no forward-hook
    scaffolding is needed.
    """
    if not layers:
        raise ValueError("Must request at least one ProGen2 layer.")

    layers = sorted(set(int(layer) for layer in layers))
    device = _get_device(device)
    config = _get_hw_config(device)

    if batch_size is None:
        batch_size = config.get("progen2_batch", config.get("rita_batch", 12))

    # Patch 1: tied-weights API
    from transformers.modeling_utils import PreTrainedModel
    if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
        PreTrainedModel.all_tied_weights_keys = {}
    # Patch 2: get_head_mask stub
    if not hasattr(PreTrainedModel, "get_head_mask"):
        def _ghm(self, hm, n_layers, is_chunked=False):
            return [None] * n_layers if hm is None else hm
        PreTrainedModel.get_head_mask = _ghm

    print(f"Loading ProGen2 ({model_name}) to {device} (batch={batch_size})...")
    print(f"  Layers requested: {layers}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Padding token: ProGen2's tokenizer has <|pad|> at id 0 but the config
    # leaves pad_token=None.  Set it explicitly so padding=True works.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = ("<|pad|>" if "<|pad|>" in tokenizer.get_vocab()
                               else tokenizer.convert_ids_to_tokens(0))

    # Verify 1:1 residue↔token.  ProGen2 has residue-level tokens and does
    # NOT auto-prepend BOS/EOS for bare amino-acid strings, so add_special
    # tokens True vs False gives identical length.
    probe = sequences[0] if sequences else "MKVLWAL"
    probe_ids = tokenizer(probe, add_special_tokens=False)["input_ids"]
    if len(probe_ids) != len(probe):
        print(f"  ⚠️  ProGen2 tokenizer produced {len(probe_ids)} tokens "
              f"for a {len(probe)}-residue probe — not 1:1.")
    else:
        print(f"  ✓  Tokenizer 1:1 residue↔token verified")

    # Load in fp32 — RITA's fp16 numerical instability doesn't apply to
    # ProGen2 (verified empirically), but fp32 is a minor cost (~3 GB)
    # and removes one potential failure mode.  Keep fp32.
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

    # Patch 3: materialise attention-block meta tensors BEFORE moving
    # to device (rebuilt on CPU, will follow the .to() call below).
    n_patched = _patch_progen2_meta_tensors(model, device)
    print(f"  meta-tensor patch: fixed {n_patched} attention-block tensors")

    model = model.to(device).eval()

    # scale_attn is a Python attribute, NOT a registered buffer — .to()
    # did not move it.  Explicit pass after device-move.
    for sub in model.modules():
        if hasattr(sub, "scale_attn") and isinstance(sub.scale_attn, torch.Tensor):
            if sub.scale_attn.device != torch.device(device):
                sub.scale_attn = sub.scale_attn.to(device)

    # Report block count
    n_blocks = getattr(model.config, "num_hidden_layers",
                       getattr(model.config, "n_layer", None))
    print(f"  Model has {n_blocks} transformer blocks")
    for L in layers:
        if L >= n_blocks:
            raise ValueError(
                f"ProGen2: requested layer {L}, only {n_blocks} blocks available.")

    buffers: Dict[int, List[np.ndarray]] = {layer: [] for layer in layers}
    batches = [sequences[i:i + batch_size]
               for i in range(0, len(sequences), batch_size)]

    print(f"Extracting {len(sequences)} sequences in {len(batches)} batches...")

    with torch.no_grad():
        for batch_seqs in tqdm(batches, desc="ProGen2"):
            inputs = tokenizer(
                batch_seqs, return_tensors="pt", padding=True,
                truncation=True, max_length=max_length,
                add_special_tokens=False,  # 1:1 with residues
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            out = model(**inputs, output_hidden_states=True, return_dict=True)
            hidden_states = out.hidden_states  # (embedding, block1, ..., blockN)

            for i in range(len(batch_seqs)):
                # attention_mask IS the residue mask since no specials added
                keep = inputs["attention_mask"][i].bool()
                for layer in layers:
                    idx = layer + 1  # skip embedding layer
                    if idx >= len(hidden_states):
                        raise ValueError(
                            f"ProGen2: requested layer {layer}, only "
                            f"{len(hidden_states)-1} blocks available.")
                    buffers[layer].append(_to_numpy(hidden_states[idx][i, keep, :]))

            del out, hidden_states

    del model
    _clear_cache()
    return {layer: np.vstack(chunks) for layer, chunks in buffers.items()}


# ============================================================
#                   RITA EXTRACTOR
# ============================================================

def extract_rita_embeddings(
    sequences: List[str],
    layers: List[int],
    device: Optional[str] = None,
    batch_size: Optional[int] = None,
    max_length: int = 1024,
    model_name: str = "lightonai/RITA_l",
) -> Dict[int, np.ndarray]:
    """
    Extract RITA embeddings.  Residue-level autoregressive PLM — 1 token per
    amino acid (vocab ~27: 20 AAs + BOS/EOS/PAD/specials), no BPE.

    Unlike ProtGPT2's BPE (where 50% of residue-level ±1/±2 neighbour pairs
    are bit-identical by construction because they share a token), RITA gives
    every residue its own token.  That makes its sequential-locality metric
    directly comparable to residue-level ESM-2 without any inter-token
    correction.

    RITA_l: 680M params (~exact size-match to ESM-2 650M), 24 transformer
    blocks. Matched relative depths: [0, 6, 12, 18, 23] (same as ProtT5).

    Implementation notes — RITA's custom modeling code (trust_remote_code=True)
    predates transformers 5.x tied-weights API, and it doesn't honour
    output_hidden_states in the standard way (it returns only the final
    hidden state).  Two workarounds:
      • Monkey-patch PreTrainedModel.all_tied_weights_keys so load completes.
      • Register forward hooks on transformer.layers[i] for each requested
        block, capture the block outputs as they pass through.
      • Load without torch_dtype and then uniformly cast to fp16 — RITA's
        attention code mixes fp32 and fp16 intermediates when loaded with
        torch_dtype=fp16, but a post-load .to(fp16) cast is consistent.
    """
    if not layers:
        raise ValueError("Must request at least one RITA layer.")

    layers = sorted(set(int(layer) for layer in layers))
    device = _get_device(device)
    config = _get_hw_config(device)

    if batch_size is None:
        batch_size = config["rita_batch"]

    # Workaround (1/3): give PreTrainedModel a default all_tied_weights_keys
    # so RITA's custom modeling code doesn't fail transformers 5.x's load path.
    from transformers.modeling_utils import PreTrainedModel
    if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
        PreTrainedModel.all_tied_weights_keys = {}

    print(f"Loading RITA ({model_name}) to {device} (batch={batch_size})...")
    print(f"  Layers requested: {layers}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Workaround (2/3): load in fp32 and DO NOT cast to fp16.  RITA's
    # attention implementation mixes fp32-upcast softmax with fp16 value
    # projections when the checkpoint is loaded in fp16 — on CPU this
    # raises "expected m1 and m2 to have the same dtype" at att @ v; on
    # MPS it silently produces NaN in deep blocks (verified empirically
    # at layer 12).  Transformers 5.x honours the checkpoint's saved
    # dtype by default (RITA_l ships fp16), so pass torch_dtype=float32
    # explicitly.  680M fp32 = ~2.7 GB, negligible on Apple Silicon
    # unified memory.
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.float32)
    model = model.float().to(device).eval()

    # RITA's tokenizer has <PAD> at id 1 but its config does not register
    # it as a special token, so tokenizer.pad_token returns None.  Set it
    # explicitly so padding=True works.  Do NOT also set eos_token — that
    # mutates the underlying tokenizer state in a way that propagates into
    # the model's attention-mask handling and produces NaN activations in
    # deep blocks (verified empirically at layer 12+).
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<PAD>" if "<PAD>" in tokenizer.get_vocab() \
                              else tokenizer.convert_ids_to_tokens(1)

    # Probe 1:1 residue↔token alignment
    probe = sequences[0] if sequences else "MKVLWAL"
    probe_ids = tokenizer(probe, add_special_tokens=False)["input_ids"]
    if len(probe_ids) != len(probe):
        print(f"  ⚠️  RITA tokenizer produced {len(probe_ids)} tokens for a "
              f"{len(probe)}-residue probe — not 1:1. Check model_name.")
    else:
        print(f"  ✓  Tokenizer 1:1 residue↔token verified")

    # Locate the transformer blocks
    transformer = getattr(model, "transformer", None)
    if transformer is None or not hasattr(transformer, "layers"):
        raise RuntimeError(
            "RITA load-path changed: expected model.transformer.layers "
            "(ModuleList). Got " + str(type(model).__name__))
    blocks = transformer.layers
    n_blocks = len(blocks)
    print(f"  Model has {n_blocks} transformer blocks (blocks={type(blocks).__name__})")
    for L in layers:
        if L >= n_blocks:
            raise ValueError(
                f"RITA: requested layer {L}, only {n_blocks} blocks available.")

    # Workaround (3/3): RITA's forward does not integrate with HF's
    # output_hidden_states — it emits only the final state.  Use forward
    # hooks on the requested blocks.  The hook receives the block's output
    # tuple (hidden, ...) and we keep index 0.
    buffers: Dict[int, List[np.ndarray]] = {layer: [] for layer in layers}
    batches = [sequences[i:i + batch_size]
               for i in range(0, len(sequences), batch_size)]

    print(f"Extracting {len(sequences)} sequences in {len(batches)} batches...")

    with torch.no_grad():
        for batch_seqs in tqdm(batches, desc="RITA"):
            # RITA's tokenizer does not declare BOS/EOS as special_ids even
            # though add_special_tokens=True prepends a BOS.  _keep_mask
            # therefore cannot filter the BOS position and we'd end up with
            # len(seq)+1 rows per protein instead of len(seq).  Use
            # add_special_tokens=False so every output row corresponds
            # exactly to one residue.
            inputs = tokenizer(
                batch_seqs, return_tensors="pt", padding=True,
                truncation=True, max_length=max_length,
                add_special_tokens=False,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Install hooks only on the layers we need (saves memory per step)
            captured: Dict[int, torch.Tensor] = {}
            hooks = []
            for L in layers:
                def make_hook(idx):
                    def hook(module, inp_args, output):
                        h = output[0] if isinstance(output, tuple) else output
                        captured[idx] = h.detach()
                    return hook
                hooks.append(blocks[L].register_forward_hook(make_hook(L)))

            try:
                _ = model(**inputs)
            finally:
                for h in hooks:
                    h.remove()

            # Per-sequence: attention_mask distinguishes real residues from
            # padding.  No specials were added so attn_mask IS the residue
            # mask, row for row.
            for i in range(len(batch_seqs)):
                keep = inputs["attention_mask"][i].bool()
                for L in layers:
                    buffers[L].append(_to_numpy(captured[L][i, keep, :]))

            del captured

    del model
    _clear_cache()
    return {layer: np.vstack(chunks) for layer, chunks in buffers.items()}


# ============================================================
#                   SMALL-CHECKPOINT WRAPPERS (scale ablation)
# ============================================================

def extract_esm2_small_embeddings(
    protein_sequences: List[str],
    layers: List[int],
    device: Optional[str] = None,
    batch_size: Optional[int] = None,
    max_length: int = 1024,
) -> Dict[int, np.ndarray]:
    """ESM-2 t12 (35M, 12 blocks). Thin wrapper for the scale-ablation arm."""
    return extract_esm2_embeddings(
        protein_sequences, layers, device=device, batch_size=batch_size,
        max_length=max_length, model_name="facebook/esm2_t12_35M_UR50D")


def extract_rita_small_embeddings(
    protein_sequences: List[str],
    layers: List[int],
    device: Optional[str] = None,
    batch_size: Optional[int] = None,
    max_length: int = 1024,
) -> Dict[int, np.ndarray]:
    """RITA-s (smaller residue-level causal). Wrapper around extract_rita_embeddings."""
    return extract_rita_embeddings(
        protein_sequences, layers, device=device, batch_size=batch_size,
        max_length=max_length, model_name="lightonai/RITA_s")


# ============================================================
#                   LAYER RECOMMENDATIONS
# ============================================================

"""
RECOMMENDED LAYERS FOR CROSS-MODEL COMPARISON (relative depth matching):

| Depth | ESM-2 (33) | ProtGPT2 (36) | ProtT5 (24) | RITA_l (24) |
|-------|------------|---------------|-------------|-------------|
| ~0%   | 0          | 0             | 0           | 0           |
| ~25%  | 8          | 9             | 6           | 6           |
| ~50%  | 16         | 18            | 12          | 12          |
| ~75%  | 24         | 27            | 18          | 18          |
| 100%  | 32         | 35            | 23          | 23          |

Example usage:

    # ESM-2
    esm2_layers = [0, 8, 16, 24, 32]
    embeddings = extract_esm2_embeddings(sequences, esm2_layers)

    # ProtGPT2
    protgpt2_layers = [0, 9, 18, 27, 35]
    embeddings = extract_protgpt2_embeddings(sequences, protgpt2_layers)

    # ProtT5
    prott5_layers = [0, 6, 12, 18, 23]
    embeddings = extract_prott5_encoder_embeddings(sequences, prott5_layers)
    embeddings = extract_prott5_decoder_embeddings(sequences, prott5_layers)

    # RITA_l (residue-level autoregressive — direct counterpart to ESM-2)
    rita_layers = [0, 6, 12, 18, 23]
    embeddings = extract_rita_embeddings(sequences, rita_layers)
"""