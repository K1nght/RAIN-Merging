#!/usr/bin/env python3
"""
Efficient Layer-wise Head-wise Null-space Projection Merging Script - Supports Complete Q/K/V/O/FFN Constraints
Core optimizations:
- Sample preprocessing: One-time boundary localization and pair sampling
- Layer-wise forward: Each layer processes each sample only once, then slices by head
- Vectorized A/AT: Use GEMM to replace scalar loops
- Batch task vectors: Extract all head task vectors once per layer
- Support V/O/FFN-Down forward feature constraints, complete model structure protection
"""

import os
import json
import math
import argparse
import re
import random
import gc
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any, Optional
from tqdm import tqdm
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ========== Basic utilities ==========

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def cleanup_memory():
    """Clean up memory and GPU cache"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def print_memory_status(stage: str):
    """Print memory status"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"🔧 [{stage}] GPU memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    else:
        print(f"🔧 [{stage}] Using CPU mode")

def read_json_samples(path: str, tokenizer, max_n: Optional[int] = None) -> List[str]:
    """Read samples from JSON file and build complete conversations"""
    with open(path, "r", encoding="utf-8") as f:
        samples = json.load(f)
    
    full_prompts = []
    for sample in samples:
        if max_n is not None and len(full_prompts) >= max_n:
            break

        prompt = sample['prompt']
        reasoning = sample.get('reasoning', '')
        response = sample.get('response', '')
        
        # Build chat messages
        messages = [{"role": "user", "content": prompt}]
        
        # Apply chat template
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Build complete conversation
        full_prompt = formatted_prompt + reasoning + "\n</think>\n\n"  # + response
        full_prompts.append(full_prompt)
    
    return full_prompts

@dataclass
class PreparedSample:
    """Preprocessed sample"""
    input_ids: torch.Tensor
    nbr: List[int]
    pairs_q: List[Tuple[int, int]] = None
    pairs_k: List[Tuple[int, int]] = None
    # New: sampling based only on t
    v_t: List[int] = None
    o_t: List[int] = None
    ffn_t: List[int] = None

def locate_segments(text: str, tokenizer) -> List[int]:
    """Locate thinking boundary token indices in text"""
    # Find thinking boundary markers
    THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
    bound_char = []
    for pat in [THINK_OPEN, THINK_CLOSE]:
        bound_char.extend([m.start() for m in re.finditer(re.escape(pat), text)])
    
    # Character to token mapping
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc["offset_mapping"]
    
    def char2tok(c):
        for i, (s, e) in enumerate(offsets):
            if s <= c < e:
                return i
        return None
    
    bound_idx = []
    for c in bound_char:
        t = char2tok(c)
        if t is not None:
            bound_idx.append(t)
    
    return sorted(list(set(bound_idx)))

def prepare_samples_unified(texts: List[str], tokenizer, radius: int, merge_types: str,
                   q_rows_per_text: int, k_rows_per_text: int,
                   v_rows_per_text: int, o_rows_per_text: int, ffn_rows_per_text: int,
                   rng: random.Random) -> List[PreparedSample]:
    """Unified sample preprocessing: locate boundaries, build neighborhoods and pairs based on merge types"""
    print("🔄 Preprocessing samples...")
    prepped = []
    
    for text in tqdm(texts, desc="Preprocessing"):
        bound_idx = locate_segments(text, tokenizer)
        if len(bound_idx) != 2:
            continue

        think_start, think_end = bound_idx[0], bound_idx[1]
        enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        T = enc["input_ids"].shape[1]
        
        # Build neighborhoods around boundaries
        start_nbr = set()
        end_nbr = set()
        for t in range(think_start, min(T, think_start+radius+1)):
            start_nbr.add(t)
        for t in range(max(0, think_end-radius), think_end+1):
            end_nbr.add(t)
        nbr = sorted(list(start_nbr) + list(end_nbr))
        
        if not nbr:
            continue
        
        # Parse required types
        merge_q = 'q' in merge_types.lower()
        merge_k = 'k' in merge_types.lower()
        merge_v = 'v' in merge_types.lower()
        merge_o = 'o' in merge_types.lower()
        merge_f = 'f' in merge_types.lower()
        
        # Generate pairs and sampling based on requirements
        sample_data = {
            "input_ids": enc["input_ids"],
            "start_nbr": start_nbr,
            "end_nbr": end_nbr,
            "nbr": nbr
        }
        
        if merge_q or merge_k:
            start_pairs = [(think_start, i) for i in start_nbr]
            end_pairs = [(i, think_end) for i in end_nbr]
            pairs = start_pairs + end_pairs
            rng.shuffle(pairs)
            if merge_q:
                sample_data["pairs_q"] = pairs[:q_rows_per_text]
            if merge_k:
                sample_data["pairs_k"] = pairs[:k_rows_per_text]
        
        if merge_v or merge_o or merge_f:
            ts = list(nbr)
            rng.shuffle(ts)
            if merge_v:
                sample_data["v_t"] = ts[:v_rows_per_text]
            if merge_o:
                sample_data["o_t"] = ts[:o_rows_per_text]
            if merge_f:
                sample_data["ffn_t"] = ts[:ffn_rows_per_text]

        # Only pass fields supported by PreparedSample
        valid_sample_data = {
            "input_ids": sample_data["input_ids"],
            "nbr": sample_data["nbr"]
        }
        
        # Optional fields
        if "pairs_q" in sample_data:
            valid_sample_data["pairs_q"] = sample_data["pairs_q"]
        if "pairs_k" in sample_data:
            valid_sample_data["pairs_k"] = sample_data["pairs_k"]
        if "v_t" in sample_data:
            valid_sample_data["v_t"] = sample_data["v_t"]
        if "o_t" in sample_data:
            valid_sample_data["o_t"] = sample_data["o_t"]
        if "ffn_t" in sample_data:
            valid_sample_data["ffn_t"] = sample_data["ffn_t"]
            
        prepped.append(PreparedSample(**valid_sample_data))
    
    print(f"✅ Preprocessing completed, valid samples: {len(prepped)}")
    return prepped



def set_strict_runtime():
    """Set strict runtime environment to ensure numerical consistency"""
    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass
    torch.use_deterministic_algorithms(False)

def collect_layer_features_with_hooks(model, input_ids: torch.Tensor, selected_layers: List[int], merge_types: str = "qkvof", max_seq_len: int = 7168):
    """Collect layer internal features using a strict-consistency hook method.
    
    Args:
        model: Model
        input_ids: Input token ids
        selected_layers: Selected layers
        merge_types: Merge types "qkvof", used to determine which features to collect
        max_seq_len: Maximum sequence length; skip if exceeded to avoid OOM (based on BF16 optimization, default 7168)
        
    Memory optimization strategies:
        1. Immediately offload features to CPU after extraction to reduce GPU memory usage
        2. Use BF16 for attention weights (50% memory saving)
        3. Automatically move back to the compute device when needed (QK/VO/FFN devices)
    """
    # Check sequence length to avoid OOM
    seq_length = input_ids.shape[-1]
    if seq_length > max_seq_len:
        print(f"⚠️  Sequence length {seq_length} exceeds the BF16 optimization limit {max_seq_len}; skipping feature extraction to avoid OOM")
        # Return an empty feature dict to keep API consistent
        return {layer_idx: {} for layer_idx in selected_layers}
    
    set_strict_runtime()  # Ensure numerical consistency
    features = {}
    hooks = []
    
    # Parse required feature types
    need_qk = 'q' in merge_types.lower() or 'k' in merge_types.lower()
    need_vo = 'v' in merge_types.lower() or 'o' in merge_types.lower()
    need_ffn = 'f' in merge_types.lower()
    
    def register_strict_layer_hooks(layer_idx, layer):
        """Register strict-consistency hooks for a single layer"""
        feat_bucket = features.setdefault(layer_idx, {})
        layer_hooks = []
        
        # 1) pre-LN output (for QK/VO attention input)
        if need_qk or need_vo:
            def hook_attn_input_ln(module, inp, out):
                # Offload to CPU immediately to reduce memory footprint
                feat_bucket["attn_input"] = out[0].detach().cpu()  # [T, d_model]
            h1 = layer.input_layernorm.register_forward_hook(hook_attn_input_ln)
            layer_hooks.append(h1)
        
        # 2) post-attention LN output (for FFN input)
        if need_ffn:
            def hook_ffn_input_ln(module, inp, out):
                # Offload to CPU immediately to reduce memory footprint
                feat_bucket["ffn_input"] = out[0].detach().cpu()  # [T, d_model]
            h2 = layer.post_attention_layernorm.register_forward_hook(hook_ffn_input_ln)
            layer_hooks.append(h2)
            
            # 3) Gate and Up projection outputs (for constraint construction)
            def hook_gate_output(module, inp, out):
                # Offload to CPU immediately to reduce memory footprint
                feat_bucket["gate_output"] = out.detach().cpu()  # [B, T, d_ff]
            def hook_up_output(module, inp, out):
                # Offload to CPU immediately to reduce memory footprint
                feat_bucket["up_output"] = out.detach().cpu()    # [B, T, d_ff]
            
            layer_mlp = layer.mlp
            h_gate = layer_mlp.gate_proj.register_forward_hook(hook_gate_output)
            h_up = layer_mlp.up_proj.register_forward_hook(hook_up_output)
            layer_hooks.extend([h_gate, h_up])
        
        # 3) For VO constraints, we need true Q/K/V linear outputs to compute attention weights
        if need_vo:
            def hook_q_proj(module, inp, out):
                # Keep on GPU for attention weight computation; will be offloaded after compute
                feat_bucket["q_proj_out"] = out.detach()
            def hook_k_proj(module, inp, out):
                feat_bucket["k_proj_out"] = out.detach()
            def hook_v_proj(module, inp, out):
                feat_bucket["v_proj_out"] = out.detach()
            
            h3 = layer.self_attn.q_proj.register_forward_hook(hook_q_proj)
            h4 = layer.self_attn.k_proj.register_forward_hook(hook_k_proj)
            h5 = layer.self_attn.v_proj.register_forward_hook(hook_v_proj)
            layer_hooks.extend([h3, h4, h5])
        
        return layer_hooks
    
    def stable_softmax_with_masks(scores: torch.Tensor, causal: bool = True, attn_mask: torch.Tensor = None) -> torch.Tensor:
        """Stable softmax computation that avoids NaNs"""
        H, T, _ = scores.shape
        device = scores.device
        
        # Causal lower-triangular mask
        if causal:
            tril = torch.tril(torch.ones(T, T, device=device, dtype=torch.bool))
        else:
            tril = torch.ones(T, T, device=device, dtype=torch.bool)
        valid = tril

        if attn_mask is not None:
            # attn_mask: 1=keep, 0=pad
            key_keep = attn_mask.to(torch.bool)  # [T]
            valid = valid & key_keep.unsqueeze(0).expand(T, T)

        # Assign -inf to masked positions
        scores_masked = scores.clone()
        minus_inf = torch.finfo(scores_masked.dtype).min
        scores_masked[~valid.unsqueeze(0).expand_as(scores_masked)] = minus_inf

        # Check whether rows are fully masked
        row_valid = valid.any(dim=-1).unsqueeze(0).expand(H, T)  # [H, T]
        
        # Stabilization: subtract row-wise max
        row_max = scores_masked.max(dim=-1, keepdim=True).values  # [H, T, 1]
        row_max = torch.where(row_valid.unsqueeze(-1), row_max, torch.zeros_like(row_max))
        scores_shifted = scores_masked - row_max

        # exp
        exp_scores = torch.exp(scores_shifted)
        exp_scores = exp_scores * valid.unsqueeze(0)

        # Normalization
        denom = exp_scores.sum(dim=-1, keepdim=True)  # [H, T, 1]
        
        # For rows that are all masked, put unit mass on diagonal to avoid div-by-zero
        empty_rows = (denom == 0)  # [H, T, 1]
        if empty_rows.any():
            eye = torch.eye(T, device=device, dtype=exp_scores.dtype)
            exp_scores = torch.where(empty_rows, eye.unsqueeze(0), exp_scores)
            denom = torch.where(empty_rows, torch.ones_like(denom), denom)

        attn = exp_scores / denom
        return attn

    def compute_attention_weights_from_qkv(layer_features, layer_idx, config):
        """Compute attention weights from true Q/K/V linear outputs"""
        q_out = layer_features[layer_idx]["q_proj_out"]  # [B, T, H*head_dim] or [T, H*head_dim]
        k_out = layer_features[layer_idx]["k_proj_out"]  # [B, T, H_kv*head_dim] or [T, H_kv*head_dim]
        
        # Handle batch dimension
        if q_out.dim() == 3:
            q_out = q_out[0]  # [T, H*head_dim]
            k_out = k_out[0]  # [T, H_kv*head_dim]
        
        T = q_out.shape[0]
        num_heads = config.num_attention_heads
        num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
        head_dim = config.hidden_size // num_heads
        
        # Rearrange to multi-head format, keep original dtype
        q = q_out.view(T, num_heads, head_dim).permute(1,0,2).contiguous()  # [H, T, head_dim]
        k = k_out.view(T, num_kv_heads, head_dim).permute(1,0,2).contiguous()  # [H_kv, T, head_dim]
        
        # GQA: expand K to match Q's head count
        if num_kv_heads < num_heads:
            rep = num_heads // num_kv_heads
            k = k.repeat_interleave(rep, dim=0)  # [H, T, head_dim]
        
        # Use bfloat16 for compute to reduce memory (keep model precision where applicable)
        original_dtype = q.dtype
        compute_dtype = torch.bfloat16 if original_dtype in [torch.float16, torch.bfloat16] else torch.float32
        
        q = q.to(compute_dtype)
        k = k.to(compute_dtype)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)  # [H, T, T]
        
        # Stable softmax
        attn_weights = stable_softmax_with_masks(scores, causal=True, attn_mask=None)
        
        # Immediately offload to CPU to free GPU memory
        return attn_weights.to(compute_dtype).cpu()
    
    # Register hooks for specified layers
    for layer_idx in selected_layers:
        layer_obj = model.model.layers[layer_idx]
        layer_hooks = register_strict_layer_hooks(layer_idx, layer_obj)
        hooks.extend(layer_hooks)
    
    try:
        with torch.no_grad():
            # Run forward to trigger hooks
            print(input_ids.shape)
            out = model.model(input_ids=input_ids)
            
            # Post-processing: compute attention weights for VO constraints
            for layer_idx in selected_layers:
                if need_vo and layer_idx in features:
                    layer_feats = features[layer_idx]
                    if ("q_proj_out" in layer_feats and 
                        "k_proj_out" in layer_feats and 
                        "v_proj_out" in layer_feats):
                        # Compute attention weights from true Q/K outputs
                        attention_weights = compute_attention_weights_from_qkv(features, layer_idx, model.config)
                        features[layer_idx]["attention_weights"] = attention_weights
                        
                        # Offload Q/K/V projection outputs to CPU and free GPU
                        features[layer_idx]["q_proj_out"] = features[layer_idx]["q_proj_out"].cpu()
                        features[layer_idx]["k_proj_out"] = features[layer_idx]["k_proj_out"].cpu()
                        features[layer_idx]["v_proj_out"] = features[layer_idx]["v_proj_out"].cpu()
                
    finally:
        # Clean up hooks
        for hook in hooks:
            hook.remove()
    
    return features


# ========== Unified constraint construction function (supports QK/VO/FFN) ==========

def build_constraints_single_layer_unified(
    model_R, prepped_samples: List[PreparedSample],
    layer: int, selected_heads: List[int],
    merge_types: str = "qkvof",
    # QK parameters
    w_q: float = 1.0, w_k: float = 1.0, q_rows_per_text: int = 8, k_rows_per_text: int = 8,
    # VO parameters
    w_v: float = 1.0, w_o: float = 1.0, v_rows_per_text: int = 4, o_rows_per_text: int = 4,
    # FFN parameters
    w_ffn: float = 1.0, ffn_rows_per_text: int = 4, readout_dirs: int = 2,
    # Device configuration
    qk_device: str = "cuda:0", vo_device: str = "cuda:0", ffn_device: str = "cuda:0",
    compute_dtype: torch.dtype = torch.float32, use_hooks: bool = True,
    # Sequence length limit
    max_seq_len: int = 7168
) -> Dict[str, Any]:
    """Unified per-layer constraint construction; supports QK/VO/FFN constraints"""
    device_R = next(model_R.parameters()).device
    d_model = model_R.config.hidden_size
    H = model_R.config.num_attention_heads
    hD = d_model // H
    KV = getattr(model_R.config, "num_key_value_heads", H)
    d_ff = getattr(model_R.config, "intermediate_size", d_model * 4)  # FFN hidden size

    # Parse required processing types
    merge_q = 'q' in merge_types.lower()
    merge_k = 'k' in merge_types.lower()
    merge_v = 'v' in merge_types.lower()
    merge_o = 'o' in merge_types.lower()
    merge_f = 'f' in merge_types.lower()
    
    print(f"  Building constraint types: {merge_types.upper()} (Q={merge_q}, K={merge_k}, V={merge_v}, O={merge_o}, F={merge_f})")
    
    # Get layer weights (keep original dtype)
    layer_obj = model_R.model.layers[layer]
    attn = layer_obj.self_attn
    
    # Load weights onto the corresponding devices as needed
    WQ = attn.q_proj.weight.data.clone().to(qk_device) if (merge_q or merge_k) else None
    WK = attn.k_proj.weight.data.clone().to(qk_device) if (merge_q or merge_k) else None
    WV = attn.v_proj.weight.data.clone().to(vo_device) if (merge_v or merge_o) else None
    WO = attn.o_proj.weight.data.clone().to(vo_device) if (merge_v or merge_o) else None
    
    # FFN weights
    if merge_f:
        mlp = layer_obj.mlp
        Wg = mlp.gate_proj.weight.data.clone().to(ffn_device)
        Wu = mlp.up_proj.weight.data.clone().to(ffn_device)
        Wd = mlp.down_proj.weight.data.clone().to(ffn_device)
    else:
        Wg = Wu = Wd = None

    # Initialize constraint collector
    constraints = {
        "qk": {h: {
        "Xi_q": [], "kj": [], "sc_q": [],
        "Xj_k": [], "qi": [], "sc_k": []
        } for h in selected_heads} if (merge_q or merge_k) else {},
        "vo": {h: {
        "Xi_v": [], "rv": [], "sc_v": [],
        "c_vec": [], "z_h": [], "sc_o": []
        } for h in selected_heads} if (merge_v or merge_o) else {},
        "ffn": {
            # down_proj constraints (current)
            "H": [], "c": [], "sc": [],
            # gate_proj constraints (new)
            "X_gate": [], "c_gate": [], "sc_gate": [],
            # up_proj constraints (new)
            "X_up": [], "c_up": [], "sc_up": []
        } if merge_f else {}
    }
    
    # Random readout directions
    if merge_v or merge_o or merge_f:
        rng = random.Random(1234)
        if merge_v or merge_o:
            C_out_vo = [torch.randn(d_model, device=vo_device).to(next(model_R.parameters()).dtype) for _ in range(readout_dirs)]
            C_out_vo = [c / (c.norm() + 1e-6) for c in C_out_vo]
        else:
            C_out_vo = []
            
        if merge_f:
            # FFN Gate/Up readout directions live in FFN hidden space (d_ff)
            C_out_ffn_gate_up = [torch.randn(d_ff, device=ffn_device).to(next(model_R.parameters()).dtype) for _ in range(readout_dirs)]
            C_out_ffn_gate_up = [c / (c.norm() + 1e-6) for c in C_out_ffn_gate_up]
            # FFN Down readout directions live in model output space (d_model)
            C_out_ffn_down = [torch.randn(d_model, device=ffn_device).to(next(model_R.parameters()).dtype) for _ in range(readout_dirs)]
            C_out_ffn_down = [c / (c.norm() + 1e-6) for c in C_out_ffn_down]
        else:
            C_out_ffn_gate_up = []
            C_out_ffn_down = []
    
    # Iterate through samples (unified feature extraction)
    skipped_samples = 0
    for samp in tqdm(prepped_samples, desc=f"Build constraints for layer {layer} ({merge_types.upper()})", leave=False):
        ids = samp.input_ids.to(device_R)
        
        # Skip too-long sequences
        seq_length = ids.shape[-1]
        if seq_length > max_seq_len:
            skipped_samples += 1
            continue
        
        if use_hooks:
            # Use hook-based feature capture
            layer_features = collect_layer_features_with_hooks(model_R, ids, [layer], merge_types, max_seq_len)
            if layer not in layer_features or not layer_features[layer]:
                # If empty feature dict or no features, skip this sample
                continue
                
            # Gather features
            X_attn = layer_features[layer].get("attn_input") if (merge_q or merge_k or merge_v or merge_o) else None
            X_ffn = layer_features[layer].get("ffn_input") if merge_f else None
            A_weights = layer_features[layer].get("attention_weights") if (merge_v or merge_o) else None
        else:
            # Fallback (may be inaccurate); for VO/QK we still prefer hooks
            if merge_v or merge_o:
                X_attn = None
                A_weights = None
            elif merge_q or merge_k:
                X_attn = None
                A_weights = None
            else:
                X_attn = None
                A_weights = None
                
            if merge_f:
                X_ffn = None
            else:
                X_ffn = None
        
        T = ids.shape[1]  # sequence length
        
        # ========== QK constraints ==========
        Q_full = None
        K_full = None
        if (merge_q or merge_k) and X_attn is not None:
            X_qk = X_attn.to(qk_device)
            Q_full = X_qk @ WQ.T if WQ is not None else None  # [T, H*hD]
            K_full = X_qk @ WK.T if WK is not None else None  # [T, KV*hD]
            
        for h in selected_heads:
            if Q_full is not None and K_full is not None:
                Q_h = Q_full[:, h*hD:(h+1)*hD]  # [T, hD]
                kvh = h % KV
                K_h = K_full[:, kvh*hD:(kvh+1)*hD]  # [T, hD]
                
                # Q constraint: use K-head outputs (attention-consistent)
                if merge_q and hasattr(samp, 'pairs_q') and samp.pairs_q:
                    Xi_q = X_qk[[i for i, _ in samp.pairs_q]]  # [m_q, d_model]
                    kj = K_h[[j for _, j in samp.pairs_q]]     # [m_q, hD_kv]
                    sc_q = torch.full((Xi_q.size(0), 1), w_q / math.sqrt(hD))
                    
                    constraints["qk"][h]["Xi_q"].append(Xi_q.cpu())
                    constraints["qk"][h]["kj"].append(kj.cpu())
                    constraints["qk"][h]["sc_q"].append(sc_q)
                
                # K constraint
                if merge_k and hasattr(samp, 'pairs_k') and samp.pairs_k:
                    Xj_k = X_qk[[j for _, j in samp.pairs_k]]  # [m_k, d_model]
                    qi = Q_h[[i for i, _ in samp.pairs_k]]     # [m_k, hD]
                    sc_k = torch.full((Xj_k.size(0), 1), w_k / math.sqrt(hD))
                    
                    constraints["qk"][h]["Xj_k"].append(Xj_k.cpu())
                    constraints["qk"][h]["qi"].append(qi.cpu())
                    constraints["qk"][h]["sc_k"].append(sc_k)
        
        # ========== VO constraints ==========
        if (merge_v or merge_o) and X_attn is not None:
            X_vo = X_attn.to(vo_device)
            A_vo = A_weights.to(vo_device) if A_weights is not None else None
            
            # Silent stability check (NaNs already handled upstream)
            V_full = X_vo @ WV.T if WV is not None else None  # [T, KV*hD]
            
            for h in selected_heads:
                if V_full is not None and A_vo is not None:
                    kvh = h % KV
                    V_h = V_full[:, kvh*hD:(kvh+1)*hD]  # [T, hD]
                    A_h = A_vo[h]  # [T, T]
                    
                    # V constraint
                    if merge_v and hasattr(samp, 'v_t') and samp.v_t:
                        for t in samp.v_t:
                            if t >= T: continue
                            # Ensure dtype consistency
                            A_h_compute = A_h[t].unsqueeze(0).to(X_vo.dtype)
                            S_th = A_h_compute @ X_vo
                            S_th = S_th.squeeze(0)  # [d_model]
                            
                            O_h = WO[:, h*hD:(h+1)*hD] if WO is not None else None
                            if O_h is not None:
                                # Skip NaNs
                                if torch.isnan(S_th).any():
                                    continue
                                    
                                for c in C_out_vo:
                                    r_h = (O_h.T @ c)
                                    if torch.isnan(r_h).any():
                                        continue
                                        
                                    sc = w_v / math.sqrt(hD)
                                    constraints["vo"][h]["Xi_v"].append(S_th.cpu())
                                    constraints["vo"][h]["rv"].append(r_h.cpu())
                                    constraints["vo"][h]["sc_v"].append(torch.tensor([sc], dtype=torch.float32))
                    
                    # O constraint
                    if merge_o and hasattr(samp, 'o_t') and samp.o_t:
                        for t in samp.o_t:
                            if t >= T: continue
                            # Ensure dtype consistency
                            A_h_compute = A_h[t].unsqueeze(0).to(V_h.dtype)
                            u_th = (A_h_compute @ V_h).squeeze(0)  # [hD]
                            
                            # Skip NaNs
                            if torch.isnan(u_th).any():
                                continue
                                
                            for c in C_out_vo:
                                sc = w_o / math.sqrt(hD)
                                constraints["vo"][h]["c_vec"].append(c.detach().cpu())
                                constraints["vo"][h]["z_h"].append(u_th.detach().cpu())
                                constraints["vo"][h]["sc_o"].append(torch.tensor([sc], dtype=torch.float32))
        
        # ========== FFN constraints ==========
        if merge_f and X_ffn is not None:
            X_ffn_device = X_ffn.to(ffn_device)
            
            # Get gate and up outputs (from hooks)
            if use_hooks and layer in layer_features:
                feat = layer_features[layer]
                gate_outputs = feat.get("gate_output")  # [B, T, d_ff]
                up_outputs = feat.get("up_output")      # [B, T, d_ff]
                
                if gate_outputs is not None and up_outputs is not None:
                    # Remove batch dim (assume B=1) and move to FFN device
                    gate_out = gate_outputs[0].to(ffn_device)  # [T, d_ff]
                    up_out = up_outputs[0].to(ffn_device)      # [T, d_ff]
                else:
                    gate_out = up_out = None
            else:
                gate_out = up_out = None
            
            if hasattr(samp, 'ffn_t') and samp.ffn_t:
                for t in samp.ffn_t:
                    if t >= T: continue
                    x = X_ffn_device[t]  # [d_model]
                    
                    # Gate constraint: based on gate_proj output
                    if gate_out is not None and Wg is not None:
                        gate_t = gate_out[t]  # [d_ff]
                        
                        # Skip NaNs
                        if not torch.isnan(gate_t).any():
                            for c in C_out_ffn_gate_up:
                                # Constraint: c^T @ gate_output = 0, where c is a random direction in FFN hidden space
                                sc_gate = w_ffn / math.sqrt(gate_t.numel())
                                constraints["ffn"]["X_gate"].append(x.detach().cpu())
                                constraints["ffn"]["c_gate"].append(c.detach().cpu())
                                constraints["ffn"]["sc_gate"].append(torch.tensor([sc_gate], dtype=torch.float32))
                    
                    # Up constraint: based on up_proj output
                    if up_out is not None and Wu is not None:
                        up_t = up_out[t]  # [d_ff]
                        
                        # Skip NaNs
                        if not torch.isnan(up_t).any():
                            for c in C_out_ffn_gate_up:
                                # Constraint: c^T @ up_output = 0
                                sc_up = w_ffn / math.sqrt(up_t.numel())
                                constraints["ffn"]["X_up"].append(x.detach().cpu())
                                constraints["ffn"]["c_up"].append(c.detach().cpu())
                                constraints["ffn"]["sc_up"].append(torch.tensor([sc_up], dtype=torch.float32))
                    
                    # Down constraint: based on SwiGLU output (original)
                    a_g = (Wg @ x) if Wg is not None else None
                    a_u = (Wu @ x) if Wu is not None else None
                    if a_g is not None and a_u is not None:
                        h = torch.nn.functional.silu(a_g) * a_u  # [d_ff]
                        
                        # Skip NaNs in FFN
                        if torch.isnan(h).any():
                            continue
                            
                        for c in C_out_ffn_down:
                            sc = w_ffn / math.sqrt(h.numel())
                            constraints["ffn"]["H"].append(h.detach().cpu())
                            constraints["ffn"]["c"].append(c.detach().cpu())
                            constraints["ffn"]["sc"].append(torch.tensor([sc], dtype=torch.float32))
    
    # Merge to batched tensors
    def stack_constraints(cons_dict, keys_to_stack, keys_to_cat):
        for h in selected_heads:
            if h in cons_dict:
                for key in keys_to_stack:
                    if cons_dict[h][key]:
                        cons_dict[h][key] = torch.stack(cons_dict[h][key], dim=0).contiguous()
                    else:
                        cons_dict[h][key] = torch.empty(0, dtype=torch.float32)
                        
                for key in keys_to_cat:
                    if cons_dict[h][key]:
                        cons_dict[h][key] = torch.cat(cons_dict[h][key], dim=0).contiguous()
                    else:
                        cons_dict[h][key] = torch.empty(0, dtype=torch.float32)
    
    # QK stacking (use cat to keep parity with original)
    if merge_q or merge_k:
        stack_constraints(
            constraints["qk"],
            keys_to_stack=[],  # QK uses cat
            keys_to_cat=["Xi_q", "kj", "Xj_k", "qi", "sc_q", "sc_k"]
        )
    
    # VO stacking
    if merge_v or merge_o:
        stack_constraints(
            constraints["vo"],
            keys_to_stack=["Xi_v", "rv", "c_vec", "z_h"],
            keys_to_cat=["sc_v", "sc_o"]
        )
    
    # FFN stacking
    if merge_f:
        ffn_cons = constraints["ffn"]
        # Down (original)
        for key in ["H", "c"]:
            if ffn_cons[key]:
                ffn_cons[key] = torch.stack(ffn_cons[key], dim=0).contiguous()
            else:
                ffn_cons[key] = torch.empty(0, dtype=torch.float32)
        # Gate (new)
        for key in ["X_gate", "c_gate"]:
            if ffn_cons[key]:
                ffn_cons[key] = torch.stack(ffn_cons[key], dim=0).contiguous()
            else:
                ffn_cons[key] = torch.empty(0, dtype=torch.float32)
        # Up (new)
        for key in ["X_up", "c_up"]:
            if ffn_cons[key]:
                ffn_cons[key] = torch.stack(ffn_cons[key], dim=0).contiguous()
            else:
                ffn_cons[key] = torch.empty(0, dtype=torch.float32)
        # Scalar weights
        for key in ["sc", "sc_gate", "sc_up"]:
            if ffn_cons[key]:
                ffn_cons[key] = torch.cat(ffn_cons[key], dim=0).contiguous()
            else:
                ffn_cons[key] = torch.empty(0, dtype=torch.float32)
    
    # Report skipped samples
    if skipped_samples > 0:
        print(f"  ⚠️  Skipped {skipped_samples}/{len(prepped_samples)} long-sequence samples (>{max_seq_len} tokens, BF16 optimization)")
    
    return constraints


# ========== Unified task vector extraction function ==========

def task_vectors_single_layer_unified(
    model_base, model_instruct, layer: int, selected_heads: List[int],
    merge_types: str = "qkvof", scaling_factor: float = 1.0
) -> Dict[str, Any]:
    """Unified single-layer task vector extraction; supports QK/VO/FFN"""
    d_model = model_base.config.hidden_size
    H = model_base.config.num_attention_heads
    hD = d_model // H
    KV = getattr(model_base.config, "num_key_value_heads", H)
    
    # Parse required processing types
    merge_q = 'q' in merge_types.lower()
    merge_k = 'k' in merge_types.lower()
    merge_v = 'v' in merge_types.lower()
    merge_o = 'o' in merge_types.lower()
    merge_f = 'f' in merge_types.lower()
    
    print(f"  Extract task vector types: {merge_types.upper()}")
    
    # Get layer objects
    attn_base = model_base.model.layers[layer].self_attn
    attn_instruct = model_instruct.model.layers[layer].self_attn
    mlp_base = model_base.model.layers[layer].mlp
    mlp_instruct = model_instruct.model.layers[layer].mlp
    
    task_vectors = {"qk": {}, "vo": {}, "ffn": {}}
    
    with torch.no_grad():
        # QK task vectors
        if merge_q or merge_k:
            dQ = (attn_instruct.q_proj.weight - attn_base.q_proj.weight) * scaling_factor if merge_q else None
            dK = (attn_instruct.k_proj.weight - attn_base.k_proj.weight) * scaling_factor if merge_k else None
        else:
            dQ = dK = None
            
        # VO task vectors
        if merge_v or merge_o:
            dV = (attn_instruct.v_proj.weight - attn_base.v_proj.weight) * scaling_factor if merge_v else None
            dO = (attn_instruct.o_proj.weight - attn_base.o_proj.weight) * scaling_factor if merge_o else None
        else:
            dV = dO = None
            
        # FFN task vectors (complete gate/up/down)
        if merge_f:
            dGate = (mlp_instruct.gate_proj.weight - mlp_base.gate_proj.weight) * scaling_factor  # [d_ff, d_model]
            dUp   = (mlp_instruct.up_proj.weight   - mlp_base.up_proj.weight)   * scaling_factor  # [d_ff, d_model]
            dDown = (mlp_instruct.down_proj.weight - mlp_base.down_proj.weight) * scaling_factor  # [d_model, d_ff]
            # Use transposed for Down to match CG implementations
            dDown_T = dDown.T.contiguous()  # [d_ff, d_model]
        else:
            dGate = dUp = dDown_T = None
        
        # Slice QK task vectors by head
        if merge_q or merge_k:
            for h in selected_heads:
                qk_head = {}
                
                if merge_q and dQ is not None:
                    q_start, q_end = h * hD, (h + 1) * hD
                    dQ_h = dQ[q_start:q_end, :].T.contiguous()  # [d_model, hD]
                    qk_head["dQ"] = dQ_h.cpu()
        
                if merge_k and dK is not None:
                    kvh = h % KV
                    k_start, k_end = kvh * hD, (kvh + 1) * hD
                    dK_h = dK[k_start:k_end, :].T.contiguous()  # [d_model, hD]
                    qk_head["dK"] = dK_h.cpu()
                    
                task_vectors["qk"][h] = qk_head
        
        # Slice VO task vectors by head
        if merge_v or merge_o:
            for h in selected_heads:
                vo_head = {}
                
                if merge_v and dV is not None:
                    kvh = h % KV
                    v_rows = slice(kvh*hD, (kvh+1)*hD)
                    dV_h = dV[v_rows, :].T.contiguous()  # [d_model, hD]
                    vo_head["dV"] = dV_h.cpu()

                if merge_o and dO is not None:
                    o_cols = slice(h*hD, (h+1)*hD)
                    dO_h = dO[:, o_cols].contiguous()  # [d_model, hD]
                    vo_head["dO"] = dO_h.cpu()
                    
                task_vectors["vo"][h] = vo_head
        
        # FFN task vectors (not per head)
        if merge_f:
            task_vectors["ffn"] = {}
            if dGate is not None:
                task_vectors["ffn"]["dGate"] = dGate.cpu()
            if dUp is not None:
                task_vectors["ffn"]["dUp"] = dUp.cpu()
            if dDown_T is not None:
                task_vectors["ffn"]["dDown_T"] = dDown_T.cpu()
    
    return task_vectors




# ========== Vectorized A/AT operations (Q/K - original) ==========

@torch.no_grad()
def A_times_delta_qk_batched(delta_dQ: torch.Tensor, delta_dK: torch.Tensor,
                            cons_h: Dict[str, torch.Tensor], device: str = "cpu",
                            compute_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Vectorized A·Δ (use GEMM instead of scalar loops)"""
    y_list = []
    
    # Q constraint: y_q = scale_q * diag(Xi @ dQ @ kj^T)
    if cons_h["Xi_q"].numel() > 0:
        Xi = cons_h["Xi_q"].to(device, compute_dtype)     # [m_q, d_model]
        kj = cons_h["kj"].to(device, compute_dtype)       # [m_q, hD]
        sc = cons_h["sc_q"].to(device, compute_dtype).squeeze(-1)  # [m_q]
        
        # Matrix multiply: Xi @ dQ -> [m_q, hD]
        M = Xi @ delta_dQ.to(device, compute_dtype)       # [m_q, hD]
        yq = sc * (M * kj).sum(dim=1)                     # [m_q]
        y_list.append(yq)
    
    # K constraint: y_k = scale_k * diag(Xj @ dK @ qi^T)
    if cons_h["Xj_k"].numel() > 0:
        Xj = cons_h["Xj_k"].to(device, compute_dtype)     # [m_k, d_model]
        qi = cons_h["qi"].to(device, compute_dtype)       # [m_k, hD]
        sc = cons_h["sc_k"].to(device, compute_dtype).squeeze(-1)  # [m_k]
        
        M = Xj @ delta_dK.to(device, compute_dtype)       # [m_k, hD]
        yk = sc * (M * qi).sum(dim=1)                     # [m_k]
        y_list.append(yk)
    
    return torch.cat(y_list, dim=0) if y_list else torch.zeros(0, device=device, dtype=compute_dtype)

@torch.no_grad()
def AT_times_y_qk_batched(y: torch.Tensor, cons_h: Dict[str, torch.Tensor],
                         shapes: Tuple[int, int], device: str = "cpu",
                         compute_dtype: torch.dtype = torch.float32) -> Tuple[torch.Tensor, torch.Tensor]:
    """Vectorized A^T·y"""
    d_model, hD = shapes
    dQ = torch.zeros((d_model, hD), device=device, dtype=compute_dtype)
    dK = torch.zeros((d_model, hD), device=device, dtype=compute_dtype)
    idx = 0
    
    # Transpose for Q constraints: dQ += Xi^T @ diag(w * sc_q) @ kj
    if cons_h["Xi_q"].numel() > 0:
        m_q = cons_h["Xi_q"].shape[0]
        w = (y[idx:idx+m_q] * cons_h["sc_q"].squeeze(-1).to(device)).unsqueeze(1)  # [m_q, 1]
        Xi = cons_h["Xi_q"].to(device, compute_dtype)                              # [m_q, d_model]
        kj = cons_h["kj"].to(device, compute_dtype)                                # [m_q, hD]
        
        # Compute Xi^T @ (w * kj) via GEMM
        dQ += Xi.T @ (w * kj)                                                      # [d_model, hD]
        idx += m_q
    
    # Transpose for K constraints
    if cons_h["Xj_k"].numel() > 0:
        m_k = cons_h["Xj_k"].shape[0]
        w = (y[idx:idx+m_k] * cons_h["sc_k"].squeeze(-1).to(device)).unsqueeze(1)  # [m_k, 1]
        Xj = cons_h["Xj_k"].to(device, compute_dtype)                              # [m_k, d_model]
        qi = cons_h["qi"].to(device, compute_dtype)                                # [m_k, hD]
        
        dK += Xj.T @ (w * qi)                                                      # [d_model, hD]
        idx += m_k
    
    return dQ, dK


# ========== Vectorized A/AT operations (V/O/FFN - new) ==========

# ===== V: ΔW_V^{h'} ∈ R[d_model, hD]
@torch.no_grad()
def A_times_delta_v(delta_dV, cons_h, device="cpu", compute_dtype=torch.float32):
    y = []
    if cons_h["Xi_v"].numel():
        Xi = cons_h["Xi_v"].to(device, compute_dtype)      # [m, d_model]
        rv = cons_h["rv"].to(device, compute_dtype)        # [m, hD]
        sc = cons_h["sc_v"].to(device, compute_dtype).squeeze(-1)  # [m]
        M  = Xi @ delta_dV.to(device, compute_dtype)       # [m, hD]
        yv = sc * (M * rv).sum(dim=1)                      # [m]
        y.append(yv)
    return torch.cat(y, dim=0) if y else torch.zeros(0, device=device, dtype=compute_dtype)

@torch.no_grad()
def AT_times_y_v(y, cons_h, d_model, hD, device="cpu", compute_dtype=torch.float32):
    dV = torch.zeros((d_model, hD), device=device, dtype=compute_dtype)
    idx = 0
    if cons_h["Xi_v"].numel():
        m = cons_h["Xi_v"].shape[0]
        w = (y[idx:idx+m] * cons_h["sc_v"].squeeze(-1).to(device)).unsqueeze(1)
        Xi = cons_h["Xi_v"].to(device, compute_dtype)    # [m,d_model]
        rv = cons_h["rv"].to(device, compute_dtype)      # [m,hD]
        dV += Xi.T @ (w * rv)             # [d_model,hD]
        idx += m
    return dV

def cg_v(cons_h, task_dV, lam=1e-4, maxit=100, tol=1e-5, device="cpu", compute_dtype=torch.float32):
    # Convert task_dV to compute_dtype for CG
    task_dV_compute = task_dV.to(device, compute_dtype)
    rhs = A_times_delta_v(task_dV_compute, cons_h, device, compute_dtype)
    if rhs.numel()==0:
        return task_dV, {"rhs":rhs.cpu(), "z":torch.tensor([]), "residual_norm":0.0, "iterations":0}
    def Mv(z):
        dV = AT_times_y_v(z, cons_h, task_dV_compute.size(0), task_dV_compute.size(1), device, compute_dtype)
        Az = A_times_delta_v(dV, cons_h, device, compute_dtype)
        return Az + lam * z
    # CG
    x = torch.zeros_like(rhs); r=rhs.clone(); p=r.clone(); rs=(r*r).sum()
    it=0
    for it in range(maxit):
        Ap = Mv(p); alpha = rs / ((p*Ap).sum()+1e-12)
        x = x + alpha*p; r = r - alpha*Ap
        rs_new = (r*r).sum()
        if torch.sqrt(rs_new) <= tol * torch.sqrt((rhs*rhs).sum()+1e-12): break
        p = r + (rs_new/(rs+1e-12))*p; rs = rs_new
    dV_w = AT_times_y_v(x, cons_h, task_dV_compute.size(0), task_dV_compute.size(1), device, compute_dtype)
    dV_proj = task_dV_compute - dV_w
    res = A_times_delta_v(dV_proj, cons_h, device, compute_dtype)
    # Back to original dtype
    return dV_proj.to(task_dV.dtype), {"rhs":rhs.cpu(),"z":x.cpu(),"residual_norm":res.norm().item(),"iterations":it+1}

# ===== O: ΔW_{O,h} ∈ R[d_model, hD] (by column block)
@torch.no_grad()
def A_times_delta_o(delta_dO, cons_h, device="cpu", compute_dtype=torch.float32):
    y = []
    if cons_h["c_vec"].numel():
        C  = cons_h["c_vec"].to(device, compute_dtype)   # [m,d_model]
        zh = cons_h["z_h"].to(device, compute_dtype)     # [m,hD]
        sc = cons_h["sc_o"].to(device, compute_dtype).squeeze(-1)
        M  = C @ delta_dO.to(device, compute_dtype)      # [m,hD]
        yo = sc * (M * zh).sum(dim=1)
        y.append(yo)
    return torch.cat(y, dim=0) if y else torch.zeros(0, device=device, dtype=compute_dtype)

@torch.no_grad()
def AT_times_y_o(y, cons_h, d_model, hD, device="cpu", compute_dtype=torch.float32):
    dO = torch.zeros((d_model, hD), device=device, dtype=compute_dtype)
    idx = 0
    if cons_h["c_vec"].numel():
        m = cons_h["c_vec"].shape[0]
        w = (y[idx:idx+m] * cons_h["sc_o"].squeeze(-1).to(device)).unsqueeze(1)
        C = cons_h["c_vec"].to(device, compute_dtype)
        zh= cons_h["z_h"].to(device, compute_dtype)
        dO += C.T @ (w * zh)
        idx += m
    return dO

def cg_o(cons_h, task_dO, lam=1e-4, maxit=100, tol=1e-5, device="cpu", compute_dtype=torch.float32):
    # Convert task_dO to compute_dtype for CG
    task_dO_compute = task_dO.to(device, compute_dtype)
    rhs = A_times_delta_o(task_dO_compute, cons_h, device, compute_dtype)
    if rhs.numel()==0:
        return task_dO, {"rhs":rhs.cpu(),"z":torch.tensor([]),"residual_norm":0.0,"iterations":0}
    def Mv(z):
        dO = AT_times_y_o(z, cons_h, task_dO_compute.size(0), task_dO_compute.size(1), device, compute_dtype)
        Az = A_times_delta_o(dO, cons_h, device, compute_dtype)
        return Az + lam * z
    # CG
    x = torch.zeros_like(rhs); r=rhs.clone(); p=r.clone(); rs=(r*r).sum()
    it=0
    for it in range(maxit):
        Ap = Mv(p); alpha = rs / ((p*Ap).sum()+1e-12)
        x = x + alpha*p; r = r - alpha*Ap
        rs_new = (r*r).sum()
        if torch.sqrt(rs_new) <= tol * torch.sqrt((rhs*rhs).sum()+1e-12): break
        p = r + (rs_new/(rs+1e-12))*p; rs = rs_new
    dO_w = AT_times_y_o(x, cons_h, task_dO_compute.size(0), task_dO_compute.size(1), device, compute_dtype)
    dO_proj = task_dO_compute - dO_w
    res = A_times_delta_o(dO_proj, cons_h, device, compute_dtype)
    # Back to original dtype
    return dO_proj.to(task_dO.dtype), {"rhs":rhs.cpu(),"z":x.cpu(),"residual_norm":res.norm().item(),"iterations":it+1}

# ===== FFN-Gate: ΔW_gate ∈ R[d_ff, d_model]
@torch.no_grad()
def A_times_delta_ffn_gate(delta_dGate, cons, device="cpu", compute_dtype=torch.float32):
    """A(dW_gate) = [c_i^T @ (dW_gate @ x_i)] for Gate constraints"""
    y = []
    if cons["X_gate"].numel() > 0:
        X = cons["X_gate"].to(device, compute_dtype)        # [m, d_model]
        C = cons["c_gate"].to(device, compute_dtype)        # [m, d_ff] (directions in FFN hidden space)
        sc = cons["sc_gate"].to(device, compute_dtype).squeeze(-1)
        
        # dW_gate @ X.T = [d_ff, d_model] @ [d_model, m] = [d_ff, m]
        M = delta_dGate.to(device, compute_dtype) @ X.T     # [d_ff, m]
        # For each i: c_i^T @ (dW_gate @ x_i) = C[i] @ M[:, i]
        yf = sc * (C * M.T).sum(dim=1)  # [m]
        y.append(yf)
    return torch.cat(y, dim=0) if y else torch.zeros(0, device=device, dtype=compute_dtype)

@torch.no_grad()
def AT_times_y_ffn_gate(y, cons, d_ff, d_model, device="cpu", compute_dtype=torch.float32):
    """Vectorized A^T·y for Gate constraints"""
    if cons["X_gate"].numel() == 0:
        return torch.zeros((d_ff, d_model), device=device, dtype=compute_dtype)
    
    # y, sc_gate: [m]; X_gate: [m,d_model]; c_gate: [m,d_ff]
    w = (y * cons["sc_gate"].squeeze(-1).to(device)).to(compute_dtype)        # [m]
    X = cons["X_gate"].to(device, compute_dtype)                               # [m, d_model]
    C = cons["c_gate"].to(device, compute_dtype)                               # [m, d_ff]
    # sum_i w[i] * (c_i ⊗ x_i^T) == C^T @ (diag(w) @ X) == C.T @ (w[:,None]*X)
    return C.T @ (w[:, None] * X)                                             # [d_ff, d_model]

def cg_ffn_gate(cons, task_dGate, lam=1e-4, maxit=100, tol=1e-5, device="cpu", compute_dtype=torch.float32):
    """CG solver for Gate"""
    task_dGate_compute = task_dGate.to(device, compute_dtype)
    rhs = A_times_delta_ffn_gate(task_dGate_compute, cons, device, compute_dtype)
    if rhs.numel() == 0:
        return task_dGate, {"rhs":rhs.cpu(),"z":torch.tensor([]),"residual_norm":0.0,"iterations":0}
    
    def Mv(z):
        dG = AT_times_y_ffn_gate(z, cons, task_dGate_compute.size(0), task_dGate_compute.size(1), device, compute_dtype)
        Az = A_times_delta_ffn_gate(dG, cons, device, compute_dtype)
        return Az + lam * z
    
    # CG
    x = torch.zeros_like(rhs); r = rhs.clone(); p = r.clone()
    rs = (r*r).sum()
    for it in range(maxit):
        Ap = Mv(p); alpha = rs / ((p*Ap).sum() + 1e-12)
        x += alpha * p; r -= alpha * Ap; rs_new = (r*r).sum()
        if torch.sqrt(rs_new) <= tol * torch.sqrt((rhs*rhs).sum()+1e-12): break
        p = r + (rs_new/(rs+1e-12))*p; rs = rs_new
    
    dG_w = AT_times_y_ffn_gate(x, cons, task_dGate_compute.size(0), task_dGate_compute.size(1), device, compute_dtype)
    dG_proj = task_dGate_compute - dG_w
    res = A_times_delta_ffn_gate(dG_proj, cons, device, compute_dtype)
    return dG_proj.to(task_dGate.dtype), {"rhs":rhs.cpu(),"z":x.cpu(),"residual_norm":res.norm().item(),"iterations":it+1}

# ===== FFN-Up: ΔW_up ∈ R[d_ff, d_model] (similar to Gate)
@torch.no_grad()
def A_times_delta_ffn_up(delta_dUp, cons, device="cpu", compute_dtype=torch.float32):
    """A(dW_up) for Up constraints"""
    y = []
    if cons["X_up"].numel() > 0:
        X = cons["X_up"].to(device, compute_dtype)        # [m, d_model]
        C = cons["c_up"].to(device, compute_dtype)        # [m, d_ff]
        sc = cons["sc_up"].to(device, compute_dtype).squeeze(-1)
        
        M = delta_dUp.to(device, compute_dtype) @ X.T     # [d_ff, m]
        yf = sc * (C * M.T).sum(dim=1)                    # [m]
        y.append(yf)
    return torch.cat(y, dim=0) if y else torch.zeros(0, device=device, dtype=compute_dtype)

@torch.no_grad()
def AT_times_y_ffn_up(y, cons, d_ff, d_model, device="cpu", compute_dtype=torch.float32):
    """Vectorized A^T·y for Up constraints"""
    if cons["X_up"].numel() == 0:
        return torch.zeros((d_ff, d_model), device=device, dtype=compute_dtype)
    
    w = (y * cons["sc_up"].squeeze(-1).to(device)).to(compute_dtype)          # [m]
    X = cons["X_up"].to(device, compute_dtype)                                 # [m, d_model]
    C = cons["c_up"].to(device, compute_dtype)                                 # [m, d_ff]
    return C.T @ (w[:, None] * X)                                             # [d_ff, d_model]

def cg_ffn_up(cons, task_dUp, lam=1e-4, maxit=100, tol=1e-5, device="cpu", compute_dtype=torch.float32):
    """CG solver for Up"""
    task_dUp_compute = task_dUp.to(device, compute_dtype)
    rhs = A_times_delta_ffn_up(task_dUp_compute, cons, device, compute_dtype)
    if rhs.numel() == 0:
        return task_dUp, {"rhs":rhs.cpu(),"z":torch.tensor([]),"residual_norm":0.0,"iterations":0}
    
    def Mv(z):
        dU = AT_times_y_ffn_up(z, cons, task_dUp_compute.size(0), task_dUp_compute.size(1), device, compute_dtype)
        Az = A_times_delta_ffn_up(dU, cons, device, compute_dtype)
        return Az + lam * z
    
    # CG
    x = torch.zeros_like(rhs); r = rhs.clone(); p = r.clone()
    rs = (r*r).sum()
    for it in range(maxit):
        Ap = Mv(p); alpha = rs / ((p*Ap).sum() + 1e-12)
        x += alpha * p; r -= alpha * Ap; rs_new = (r*r).sum()
        if torch.sqrt(rs_new) <= tol * torch.sqrt((rhs*rhs).sum()+1e-12): break
        p = r + (rs_new/(rs+1e-12))*p; rs = rs_new
    
    dU_w = AT_times_y_ffn_up(x, cons, task_dUp_compute.size(0), task_dUp_compute.size(1), device, compute_dtype)
    dU_proj = task_dUp_compute - dU_w
    res = A_times_delta_ffn_up(dU_proj, cons, device, compute_dtype)
    return dU_proj.to(task_dUp.dtype), {"rhs":rhs.cpu(),"z":x.cpu(),"residual_norm":res.norm().item(),"iterations":it+1}

# ===== FFN-Down: ΔW_down ∈ R[d_model, d_ff] (we use transposed ΔW_down^T for shape match)
@torch.no_grad()
def A_times_delta_ffn_down(delta_dDown_T, cons, device="cpu", compute_dtype=torch.float32):
    # delta_dDown_T: [d_ff, d_model]; H: [m,d_ff], c:[m,d_model]
    y = []
    if cons["H"].numel():
        H = cons["H"].to(device, compute_dtype)        # [m, d_ff]
        C = cons["c"].to(device, compute_dtype)        # [m, d_model]
        sc= cons["sc"].to(device, compute_dtype).squeeze(-1)
        M = H @ delta_dDown_T.to(device, compute_dtype) # [m, d_model]
        yf= sc * (M * C).sum(dim=1)
        y.append(yf)
    return torch.cat(y, dim=0) if y else torch.zeros(0, device=device, dtype=compute_dtype)

@torch.no_grad()
def AT_times_y_ffn_down(y, cons, d_ff, d_model, device="cpu", compute_dtype=torch.float32):
    dDown_T = torch.zeros((d_ff, d_model), device=device, dtype=compute_dtype)
    if cons["H"].numel():
        m = cons["H"].shape[0]
        w = (y[:m] * cons["sc"].squeeze(-1).to(device)).unsqueeze(1)
        H = cons["H"].to(device, compute_dtype)    # [m,d_ff]
        C = cons["c"].to(device, compute_dtype)    # [m,d_model]
        dDown_T += H.T @ (w * C)
    return dDown_T

# ===== FFN Dense/Cholesky efficient solvers (faster than CG when m is small) =====

@torch.no_grad()
def ffn_down_dense_project(cons, task_dDown_T, lam=1e-4, device="cpu", compute_dtype=torch.float32):
    """FFN Down: explicit Hadamard Gram + Cholesky solver (exact; faster for small m)"""
    # cons["H"]: [m, d_ff], cons["c"]: [m, d_model], cons["sc"]:[m,1]
    H = cons["H"].to(device, compute_dtype)               # [m, d_ff]
    C = cons["c"].to(device, compute_dtype)               # [m, d_model]
    s = cons["sc"].to(device, compute_dtype).squeeze(-1)  # [m]

    m = H.size(0)
    if m == 0:
        return task_dDown_T, {"residual_norm": 0.0, "solver": "dense_skip", "m": 0, "iterations": 0}

    # Gram: G = (s s^T) ⊙ (H H^T) ⊙ (C C^T) + λI
    HH = H @ H.T              # [m,m]
    CC = C @ C.T              # [m,m]
    G  = (HH * CC) * (s[:,None] * s[None,:])  # Hadamard
    G  = G + lam * torch.eye(m, device=device, dtype=compute_dtype)

    # rhs = s * diag( (H @ Δ) @ C^T )
    Δ = task_dDown_T.to(device, compute_dtype)            # [d_ff, d_model]
    M = (H @ Δ)                                           # [m, d_model]
    rhs = s * (M * C).sum(dim=1)                          # [m]

    # solve (G z = rhs)
    try:
        L = torch.linalg.cholesky(G)
        z = torch.cholesky_solve(rhs.unsqueeze(-1), L).squeeze(-1)  # [m]
    except RuntimeError as e:
        # Fallback to LU if Cholesky fails
        z = torch.linalg.solve(G, rhs)
    
    w = z * s                                                   # [m]

    # Δ_proj = Δ - A^T z ; A^T z = H^T @ (w[:,None] * C)
    dT_w = H.T @ (w[:, None] * C)                # [d_ff, d_model]
    dT_proj = Δ - dT_w
    
    # Residual ||A Δ_proj||
    M2   = (H @ dT_proj)                                     # [m, d_model]
    resid= (s * (M2 * C).sum(dim=1)).norm().item()

    return dT_proj.to(task_dDown_T.dtype), {
        "residual_norm": resid, "solver": "dense_cholesky", "m": m, "iterations": 1
    }

@torch.no_grad()
def ffn_gate_dense_project(cons, task_dGate, lam=1e-4, device="cpu", compute_dtype=torch.float32):
    """FFN Gate: explicit Hadamard Gram + Cholesky solver"""
    X = cons["X_gate"].to(device, compute_dtype)               # [m, d_model]
    C = cons["c_gate"].to(device, compute_dtype)               # [m, d_ff]
    s = cons["sc_gate"].to(device, compute_dtype).squeeze(-1)  # [m]

    m = X.size(0)
    if m == 0:
        return task_dGate, {"residual_norm": 0.0, "solver": "dense_skip", "m": 0, "iterations": 0}

    # Gram: G = (s s^T) ⊙ (C C^T) ⊙ (X X^T) + λI
    XX = X @ X.T              # [m,m]
    CC = C @ C.T              # [m,m]
    G  = (CC * XX) * (s[:,None] * s[None,:])  # Hadamard
    G  = G + lam * torch.eye(m, device=device, dtype=compute_dtype)

    # rhs = s * ((X @ Δ^T) ⊙ C).sum(-1)
    Δ = task_dGate.to(device, compute_dtype)            # [d_ff, d_model]
    M = X @ Δ.T                                         # [m, d_ff]
    rhs = s * (M * C).sum(dim=1)                        # [m]

    # solve (G z = rhs)
    try:
        L = torch.linalg.cholesky(G)
        z = torch.cholesky_solve(rhs.unsqueeze(-1), L).squeeze(-1)  # [m]
    except RuntimeError:
        z = torch.linalg.solve(G, rhs)
    
    w = z * s                                                   # [m]

    # Δ_proj = Δ - A^T z ; A^T z = C^T @ (w[:,None] * X)
    dG_w = C.T @ (w[:, None] * X)                # [d_ff, d_model]
    dG_proj = Δ - dG_w
    
    # Residual ||A Δ_proj||
    M2   = X @ dG_proj.T                                     # [m, d_ff]
    resid= (s * (M2 * C).sum(dim=1)).norm().item()

    return dG_proj.to(task_dGate.dtype), {
        "residual_norm": resid, "solver": "dense_cholesky", "m": m, "iterations": 1
    }


@torch.no_grad()
def ffn_up_dense_project(cons, task_dUp, lam=1e-4, device="cpu", compute_dtype=torch.float32):
    """FFN Up: explicit Hadamard Gram + Cholesky solver"""
    X = cons["X_up"].to(device, compute_dtype)               # [m, d_model]
    C = cons["c_up"].to(device, compute_dtype)               # [m, d_ff]
    s = cons["sc_up"].to(device, compute_dtype).squeeze(-1)  # [m]

    m = X.size(0)
    if m == 0:
        return task_dUp, {"residual_norm": 0.0, "solver": "dense_skip", "m": 0, "iterations": 0}

    # Gram: G = (s s^T) ⊙ (C C^T) ⊙ (X X^T) + λI
    XX = X @ X.T              # [m,m]
    CC = C @ C.T              # [m,m]
    G  = (CC * XX) * (s[:,None] * s[None,:])  # Hadamard
    G  = G + lam * torch.eye(m, device=device, dtype=compute_dtype)

    # rhs = s * ((X @ Δ^T) ⊙ C).sum(-1)
    Δ = task_dUp.to(device, compute_dtype)              # [d_ff, d_model]
    M = X @ Δ.T                                         # [m, d_ff]
    rhs = s * (M * C).sum(dim=1)                        # [m]

    # solve (G z = rhs)
    try:
        L = torch.linalg.cholesky(G)
        z = torch.cholesky_solve(rhs.unsqueeze(-1), L).squeeze(-1)  # [m]
    except RuntimeError:
        z = torch.linalg.solve(G, rhs)
    
    w = z * s                                                   # [m]

    # Δ_proj = Δ - A^T z ; A^T z = C^T @ (w[:,None] * X)
    dU_w = C.T @ (w[:, None] * X)                # [d_ff, d_model]
    dU_proj = Δ - dU_w
    
    # Residual ||A Δ_proj||
    M2   = X @ dU_proj.T                                     # [m, d_ff]
    resid= (s * (M2 * C).sum(dim=1)).norm().item()

    return dU_proj.to(task_dUp.dtype), {
        "residual_norm": resid, "solver": "dense_cholesky", "m": m, "iterations": 1
    }


# ===== Q/K/V/O Dense/Cholesky explicit solvers (same pattern as FFN) =====

@torch.no_grad()
def q_dense_project(cons_h, task_dQ, lam=1e-4, device="cpu", compute_dtype=torch.float32):
    """
    Dense projection for Q constraints:
    cons_h["Xi_q"]: [m, d_model], cons_h["kj"]: [m, hD], cons_h["sc_q"]: [m]
    Δ = task_dQ ∈ R[d_model, hD]
    Gram: G = (s s^T) ⊙ (X X^T) ⊙ (KJ KJ^T) + λI
    rhs_i = s_i * < (X_i Δ), kj_i >
    A^T z = X^T @ ( (z ⊙ s)[:,None] ⊙ kj )
    """
    if cons_h["Xi_q"].numel() == 0:
        return task_dQ, {"residual_norm": 0.0, "solver": "dense_skip", "m": 0, "iterations": 0}

    X  = cons_h["Xi_q"].to(device, compute_dtype)          # [m, d_model]
    KJ = cons_h["kj"].to(device, compute_dtype)            # [m, hD]
    s  = cons_h["sc_q"].to(device, compute_dtype).squeeze(-1)  # [m]
    m  = X.size(0)

    XX = X @ X.T                                           # [m, m]
    KK = KJ @ KJ.T                                         # [m, m]
    G  = (XX * KK) * (s[:, None] * s[None, :]) + lam * torch.eye(m, device=device, dtype=compute_dtype)

    Δ  = task_dQ.to(device, compute_dtype)                 # [d_model, hD]
    M  = X @ Δ                                             # [m, hD]
    rhs = s * (M * KJ).sum(dim=1)                          # [m]

    try:
        L = torch.linalg.cholesky(G)
        z = torch.cholesky_solve(rhs.unsqueeze(-1), L).squeeze(-1)
    except RuntimeError:
        z = torch.linalg.solve(G, rhs)

    w = z * s                                              # [m]
    dQ_w   = X.T @ (w[:, None] * KJ)                       # [d_model, hD]
    dQ_proj= Δ - dQ_w

    # Residual ||A dQ_proj||
    M2   = X @ dQ_proj                                     # [m, hD]
    resid= (s * (M2 * KJ).sum(dim=1)).norm().item()
    return dQ_proj.to(task_dQ.dtype), {"residual_norm": resid, "solver": "dense_cholesky", "m": m, "iterations": 1}


@torch.no_grad()
def k_dense_project(cons_h, task_dK, lam=1e-4, device="cpu", compute_dtype=torch.float32):
    """
    Dense projection for K constraints:
    cons_h["Xj_k"]: [m, d_model], cons_h["qi"]: [m, hD], cons_h["sc_k"]: [m]
    Δ = task_dK ∈ R[d_model, hD]
    Gram: G = (s s^T) ⊙ (X X^T) ⊙ (QI QI^T) + λI
    rhs_i = s_i * < (X_i Δ), qi_i >
    A^T z = X^T @ ( (z ⊙ s)[:,None] ⊙ qi )
    """
    if cons_h["Xj_k"].numel() == 0:
        return task_dK, {"residual_norm": 0.0, "solver": "dense_skip", "m": 0, "iterations": 0}

    X  = cons_h["Xj_k"].to(device, compute_dtype)          # [m, d_model]
    QI = cons_h["qi"].to(device, compute_dtype)            # [m, hD]
    s  = cons_h["sc_k"].to(device, compute_dtype).squeeze(-1)  # [m]
    m  = X.size(0)

    XX = X @ X.T
    QQ = QI @ QI.T
    G  = (XX * QQ) * (s[:, None] * s[None, :]) + lam * torch.eye(m, device=device, dtype=compute_dtype)

    Δ  = task_dK.to(device, compute_dtype)
    M  = X @ Δ                                             # [m, hD]
    rhs = s * (M * QI).sum(dim=1)

    try:
        L = torch.linalg.cholesky(G)
        z = torch.cholesky_solve(rhs.unsqueeze(-1), L).squeeze(-1)
    except RuntimeError:
        z = torch.linalg.solve(G, rhs)

    w = z * s
    dK_w   = X.T @ (w[:, None] * QI)
    dK_proj= Δ - dK_w

    M2   = X @ dK_proj
    resid= (s * (M2 * QI).sum(dim=1)).norm().item()
    return dK_proj.to(task_dK.dtype), {"residual_norm": resid, "solver": "dense_cholesky", "m": m, "iterations": 1}


@torch.no_grad()
def v_dense_project(cons_h, task_dV, lam=1e-4, device="cpu", compute_dtype=torch.float32):
    """
    Dense projection for V constraints:
    cons_h["Xi_v"]: [m, d_model], cons_h["rv"]: [m, hD], cons_h["sc_v"]: [m]
    Δ = task_dV ∈ R[d_model, hD]
    Gram: G = (s s^T) ⊙ (X X^T) ⊙ (RV RV^T) + λI
    rhs_i = s_i * < (X_i Δ), rv_i >
    A^T z = X^T @ ( (z ⊙ s)[:,None] ⊙ rv )
    """
    if cons_h["Xi_v"].numel() == 0:
        return task_dV, {"residual_norm": 0.0, "solver": "dense_skip", "m": 0, "iterations": 0}

    X  = cons_h["Xi_v"].to(device, compute_dtype)
    RV = cons_h["rv"].to(device, compute_dtype)
    s  = cons_h["sc_v"].to(device, compute_dtype).squeeze(-1)
    m  = X.size(0)

    XX = X @ X.T
    RR = RV @ RV.T
    G  = (XX * RR) * (s[:, None] * s[None, :]) + lam * torch.eye(m, device=device, dtype=compute_dtype)

    Δ  = task_dV.to(device, compute_dtype)
    M  = X @ Δ
    rhs = s * (M * RV).sum(dim=1)

    try:
        L = torch.linalg.cholesky(G)
        z = torch.cholesky_solve(rhs.unsqueeze(-1), L).squeeze(-1)
    except RuntimeError:
        z = torch.linalg.solve(G, rhs)

    w = z * s
    dV_w   = X.T @ (w[:, None] * RV)
    dV_proj= Δ - dV_w

    M2   = X @ dV_proj
    resid= (s * (M2 * RV).sum(dim=1)).norm().item()
    return dV_proj.to(task_dV.dtype), {"residual_norm": resid, "solver": "dense_cholesky", "m": m, "iterations": 1}


@torch.no_grad()
def o_dense_project(cons_h, task_dO, lam=1e-4, device="cpu", compute_dtype=torch.float32):
    """
    Dense projection for O constraints:
    cons_h["c_vec"]: [m, d_model], cons_h["z_h"]: [m, hD], cons_h["sc_o"]: [m]
    Δ = task_dO ∈ R[d_model, hD]
    Gram: G = (s s^T) ⊙ (C C^T) ⊙ (Z Z^T) + λI
    rhs_i = s_i * < (C_i Δ), z_i >
    A^T z = C^T @ ( (z ⊙ s)[:,None] ⊙ Z )
    """
    if cons_h["c_vec"].numel() == 0:
        return task_dO, {"residual_norm": 0.0, "solver": "dense_skip", "m": 0, "iterations": 0}

    C  = cons_h["c_vec"].to(device, compute_dtype)
    Z  = cons_h["z_h"].to(device, compute_dtype)
    s  = cons_h["sc_o"].to(device, compute_dtype).squeeze(-1)
    m  = C.size(0)

    CC = C @ C.T
    ZZ = Z @ Z.T
    G  = (CC * ZZ) * (s[:, None] * s[None, :]) + lam * torch.eye(m, device=device, dtype=compute_dtype)

    Δ  = task_dO.to(device, compute_dtype)
    M  = C @ Δ
    rhs = s * (M * Z).sum(dim=1)

    try:
        L = torch.linalg.cholesky(G)
        z = torch.cholesky_solve(rhs.unsqueeze(-1), L).squeeze(-1)
    except RuntimeError:
        z = torch.linalg.solve(G, rhs)

    w = z * s
    dO_w   = C.T @ (w[:, None] * Z)
    dO_proj= Δ - dO_w

    M2   = C @ dO_proj
    resid= (s * (M2 * Z).sum(dim=1)).norm().item()
    return dO_proj.to(task_dO.dtype), {"residual_norm": resid, "solver": "dense_cholesky", "m": m, "iterations": 1}

def cg_ffn_down(cons, task_dDown_T, lam=1e-4, maxit=100, tol=1e-5, device="cpu", compute_dtype=torch.float32):
    # Convert task_dDown_T to compute_dtype for CG
    task_dDown_T_compute = task_dDown_T.to(device, compute_dtype)
    rhs = A_times_delta_ffn_down(task_dDown_T_compute, cons, device, compute_dtype)
    if rhs.numel()==0:
        return task_dDown_T, {"rhs":rhs.cpu(),"z":torch.tensor([]),"residual_norm":0.0,"iterations":0}
    def Mv(z):
        dT = AT_times_y_ffn_down(z, cons, task_dDown_T_compute.size(0), task_dDown_T_compute.size(1), device, compute_dtype)
        Az = A_times_delta_ffn_down(dT, cons, device, compute_dtype)
        return Az + lam * z
    # CG
    x = torch.zeros_like(rhs); r=rhs.clone(); p=r.clone(); rs=(r*r).sum()
    it=0
    for it in range(maxit):
        Ap = Mv(p); alpha = rs / ((p*Ap).sum()+1e-12)
        x = x + alpha*p; r = r - alpha*Ap
        rs_new = (r*r).sum()
        if torch.sqrt(rs_new) <= tol * torch.sqrt((rhs*rhs).sum()+1e-12): break
        p = r + (rs_new/(rs+1e-12))*p; rs = rs_new
    dT_w = AT_times_y_ffn_down(x, cons, task_dDown_T_compute.size(0), task_dDown_T_compute.size(1), device, compute_dtype)
    dT_proj = task_dDown_T_compute - dT_w
    res = A_times_delta_ffn_down(dT_proj, cons, device, compute_dtype)
    # Back to original dtype
    return dT_proj.to(task_dDown_T.dtype), {"rhs":rhs.cpu(),"z":x.cpu(),"residual_norm":res.norm().item(),"iterations":it+1}


# ========== Vectorized CG solver (Q/K - original) ==========

def cg_single_head_batched(cons_h: Dict[str, torch.Tensor],
                          task_dQ: torch.Tensor, task_dK: torch.Tensor,
                          lambda_ridge: float = 1e-4, maxit: int = 100,
                          tol: float = 1e-5, device: str = "cpu",
                          compute_dtype: torch.dtype = torch.float32) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Vectorized single-head CG"""
    d_model, hD = task_dQ.shape
    
    # Convert task vectors to compute_dtype for CG
    task_dQ_compute = task_dQ.to(device, compute_dtype)
    task_dK_compute = task_dK.to(device, compute_dtype)
    
    # Right-hand side
    rhs = A_times_delta_qk_batched(task_dQ_compute, task_dK_compute, cons_h, device, compute_dtype)
    
    if rhs.numel() == 0:
        return task_dQ, task_dK, {
            "rhs": rhs.cpu(),
            "z": torch.tensor([]),
            "residual_norm": 0.0,
            "iterations": 0
        }
    
    def Mv(z):
        """Matrix-vector multiply: (AA^T + λI)z"""
        dQ_temp, dK_temp = AT_times_y_qk_batched(z, cons_h, (d_model, hD), device, compute_dtype)
        Az = A_times_delta_qk_batched(dQ_temp, dK_temp, cons_h, device, compute_dtype)
        return Az + lambda_ridge * z
    
    # Standard CG
    x = torch.zeros_like(rhs)
    r = rhs.clone()
    p = r.clone()
    rs = (r * r).sum()
    
    for it in range(maxit):
        Ap = Mv(p)
        alpha = rs / ((p * Ap).sum() + 1e-12)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = (r * r).sum()
        
        if torch.sqrt(rs_new) <= tol * torch.sqrt((rhs * rhs).sum() + 1e-12):
            break
        
        beta = rs_new / (rs + 1e-12)
        p = r + beta * p
        rs = rs_new
    
    # Projection
    dQ_w, dK_w = AT_times_y_qk_batched(x, cons_h, (d_model, hD), device, compute_dtype)
    dQ_proj_compute = task_dQ_compute - dQ_w
    dK_proj_compute = task_dK_compute - dK_w
    
    # Residual check
    residual = A_times_delta_qk_batched(dQ_proj_compute, dK_proj_compute, cons_h, device, compute_dtype)
    
    # Back to original dtype
    return dQ_proj_compute.to(task_dQ.dtype), dK_proj_compute.to(task_dK.dtype), {
        "rhs": rhs.cpu(),
        "z": x.cpu(),
        "residual_norm": residual.norm().item(),
        "iterations": it + 1
    }


# ========== Main optimized pipeline ==========

def optimized_layerwise_headwise_nullspace_projection(
    model_base, model_instruct, model_target,
    texts_R: List[str], tokenizer,
    selected_layers: List[int], selected_heads: List[int],
    neigh_radius: int, lambda_ridge: float, cg_maxit: int, cg_tol: float,
    scaling_factor: float = 1.0, compute_dtype: torch.dtype = torch.float32,
    # Unified parameter selection
    merge_types: str = "qkvof",  # e.g., "qk", "qkvo", "qkvof"
    # QK params
    q_rows_per_text: int = 8, k_rows_per_text: int = 8, w_q: float = 1.0, w_k: float = 1.0,
    # VO params
    v_rows_per_text: int = 4, o_rows_per_text: int = 4, w_v: float = 1.0, w_o: float = 1.0,
    # FFN params
    ffn_rows_per_text: int = 4, w_ffn: float = 1.0, readout_dirs: int = 2,
    seed: int = 42,
    # Multi-device config
    qk_device: str = "auto", vo_device: str = "auto", ffn_device: str = "auto",
    # Hook config
    use_hooks: bool = True
) -> Dict[str, Any]:
    """Optimized layer-wise head-wise null-space projection (supports Q/K/V/O/FFN)"""
    
    print("🚀 Starting optimized layer-wise head-wise null-space projection (Q/K/V/O/FFN)...")
    rng = random.Random(seed)
    
    d_model = model_target.config.hidden_size
    n_heads = model_target.config.num_attention_heads
    head_dim = d_model // n_heads
    kv_heads = getattr(model_target.config, 'num_key_value_heads', n_heads)
    
    print(f"📋 Config: d_model={d_model}, n_heads={n_heads}, kv_heads={kv_heads}")
    print(f"🔧 Task vector scaling factor: {scaling_factor}")
    print(f"Feature extraction: {'Hook-based (recommended)' if use_hooks else 'Original'}")
    
    # 1) Preprocess samples (unified)
    prepped_samples = prepare_samples_unified(
        texts_R, tokenizer, neigh_radius, merge_types,
        q_rows_per_text, k_rows_per_text, v_rows_per_text, o_rows_per_text, ffn_rows_per_text, rng
    )
    
    # Multi-device config
    if qk_device == "auto":
        qk_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if vo_device == "auto":
        vo_device = "cuda:1" if torch.cuda.device_count() > 1 else qk_device
    if ffn_device == "auto":
        ffn_device = "cuda:2" if torch.cuda.device_count() > 2 else vo_device
    
    # Parse merge types
    merge_q = 'q' in merge_types.lower()
    merge_k = 'k' in merge_types.lower()
    merge_v = 'v' in merge_types.lower()
    merge_o = 'o' in merge_types.lower()
    merge_f = 'f' in merge_types.lower()
    
    print(f"🔧 Temporarily load target model to GPU (multi-GPU mode)")
    print(f"🔧 Device assignment: QK={qk_device}, VO={vo_device}, FFN={ffn_device}")
    print(f"🎯 Merge types: {merge_types.upper()} (Q={merge_q}, K={merge_k}, V={merge_v}, O={merge_o}, F={merge_f})")
    
    model_R_temp = AutoModelForCausalLM.from_pretrained(
        model_target.config._name_or_path,
        torch_dtype=torch.float16,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True
    ).eval()
    
    total_stats = {
        "total_params_modified": 0,
        "total_norm_q": 0.0,
        "total_norm_k": 0.0,
        "total_norm_v": 0.0,
        "total_norm_o": 0.0,
        "total_norm_ffn": 0.0,
        "total_constraint_residual": 0.0,
        "total_cg_iterations": 0,
        "layer_stats": {}
    }
    
    # 2) Process by layer (key optimization: do each layer as a whole)
    for li_idx, li in enumerate(tqdm(selected_layers, desc="Optimized per-layer processing")):
        print(f"\n🔄 Processing layer {li} ({li_idx+1}/{len(selected_layers)})")
        
        # 2a) Unified constraint build (QK/VO/FFN)
        print(f"  📐 Building constraints for layer {li} ({merge_types.upper()})...")
        layer_constraints = build_constraints_single_layer_unified(
            model_R_temp, prepped_samples, li, selected_heads, merge_types,
            w_q, w_k, q_rows_per_text, k_rows_per_text,
            w_v, w_o, v_rows_per_text, o_rows_per_text,
            w_ffn, ffn_rows_per_text, readout_dirs,
            qk_device, vo_device, ffn_device, compute_dtype, use_hooks,
            max_seq_len=args.max_seq_len  # use CLI limit
        )
        
        # 2b) Unified task vector extraction (QK/VO/FFN)
        print(f"  🎯 Extracting task vectors for layer {li} ({merge_types.upper()})...")
        layer_task_vectors = task_vectors_single_layer_unified(
            model_base, model_instruct, li, selected_heads, merge_types, scaling_factor
            )
        
        layer_stats = {"heads": {}}
        
        # 2c) Unified solving & applying (QK/VO/FFN) — default to dense solvers; fallback to CG
        for h in tqdm(selected_heads, desc=f"Per-head solving for layer {li} ({merge_types.upper()})", leave=False):
            head_stat = {
                "constraints_qk": 0, "constraints_v": 0, "constraints_o": 0,
                "norm_q": 0.0, "norm_k": 0.0, "norm_v": 0.0, "norm_o": 0.0,
                "residual_norm_qk": 0.0, "residual_norm_v": 0.0, "residual_norm_o": 0.0,
                "cg_iterations": 0, "params_modified": 0
            }

            # —— Q / K (dense by default; fallback to CG) ——
            if (merge_q or merge_k) and "qk" in layer_constraints and h in layer_constraints["qk"]:
                cons_h_qk = layer_constraints["qk"][h]
                total_constraints_qk = cons_h_qk["Xi_q"].shape[0] + cons_h_qk["Xj_k"].shape[0]
                head_stat["constraints_qk"] = total_constraints_qk

                if total_constraints_qk > 0 and h in layer_task_vectors["qk"]:
                    task_qk = layer_task_vectors["qk"][h]
                    dQ_proj = None
                    dK_proj = None

                    # Q component
                    if merge_q and ("dQ" in task_qk) and (cons_h_qk["Xi_q"].numel() > 0):
                        try:
                            dQ_proj, info_q = q_dense_project(cons_h_qk, task_qk["dQ"], lambda_ridge, device="cpu", compute_dtype=compute_dtype)
                            head_stat["norm_q"] = dQ_proj.norm().item()
                            head_stat["residual_norm_qk"] += info_q["residual_norm"]
                            head_stat["cg_iterations"] += info_q.get("iterations", 1)
                        except RuntimeError:
                            # Fallback to CG
                            dQ_proj, _, info_qk = cg_single_head_batched(
                                {"Xi_q": cons_h_qk["Xi_q"], "kj": cons_h_qk["kj"], "sc_q": cons_h_qk["sc_q"],
                                 "Xj_k": torch.empty(0), "qi": torch.empty(0), "sc_k": torch.empty(0)},
                                task_qk["dQ"], torch.zeros_like(task_qk["dQ"]),
                                lambda_ridge, cg_maxit, cg_tol, device="cpu", compute_dtype=compute_dtype
                            )
                            head_stat["norm_q"] = dQ_proj.norm().item()
                            head_stat["residual_norm_qk"] += info_qk["residual_norm"]
                            head_stat["cg_iterations"] += info_qk["iterations"]

                    # K component
                    if merge_k and ("dK" in task_qk) and (cons_h_qk["Xj_k"].numel() > 0):
                        try:
                            dK_proj, info_k = k_dense_project(cons_h_qk, task_qk["dK"], lambda_ridge, device="cpu", compute_dtype=compute_dtype)
                            head_stat["norm_k"] = dK_proj.norm().item()
                            head_stat["residual_norm_qk"] += info_k["residual_norm"]
                            head_stat["cg_iterations"] += info_k.get("iterations", 1)
                        except RuntimeError:
                            _, dK_proj, info_qk = cg_single_head_batched(
                                {"Xi_q": torch.empty(0), "kj": torch.empty(0), "sc_q": torch.empty(0),
                                 "Xj_k": cons_h_qk["Xj_k"], "qi": cons_h_qk["qi"], "sc_k": cons_h_qk["sc_k"]},
                                torch.zeros_like(task_qk["dK"]), task_qk["dK"],
                                lambda_ridge, cg_maxit, cg_tol, device="cpu", compute_dtype=compute_dtype
                            )
                            head_stat["norm_k"] = dK_proj.norm().item()
                            head_stat["residual_norm_qk"] += info_qk["residual_norm"]
                            head_stat["cg_iterations"] += info_qk["iterations"]

                    # Apply to target model weights
                    layer_target = model_target.model.layers[li].self_attn
                    with torch.no_grad():
                        if merge_q and (dQ_proj is not None):
                            WQ_target = layer_target.q_proj.weight.data.to(compute_dtype)
                            q_start, q_end = h * head_dim, (h + 1) * head_dim
                            WQ_target[q_start:q_end, :] += dQ_proj.T.to(WQ_target.device)
                            layer_target.q_proj.weight.data = WQ_target.to(layer_target.q_proj.weight.dtype)
                            head_stat["params_modified"] += dQ_proj.numel()
                        if merge_k and (dK_proj is not None):
                            WK_target = layer_target.k_proj.weight.data.to(compute_dtype)
                            kvh = h % kv_heads
                            k_start, k_end = kvh * head_dim, (kvh + 1) * head_dim
                            WK_target[k_start:k_end, :] += dK_proj.T.to(WK_target.device)
                            layer_target.k_proj.weight.data = WK_target.to(layer_target.k_proj.weight.dtype)
                            head_stat["params_modified"] += dK_proj.numel()
            
            # —— V (dense by default; fallback to CG) ——
            if merge_v and "vo" in layer_constraints and h in layer_constraints["vo"]:
                cons_h_v = layer_constraints["vo"][h]
                if "Xi_v" in cons_h_v and cons_h_v["Xi_v"].numel() > 0:
                    head_stat["constraints_v"] = cons_h_v["Xi_v"].shape[0]
                    
                    if h in layer_task_vectors["vo"] and "dV" in layer_task_vectors["vo"][h]:
                        dV_task = layer_task_vectors["vo"][h]["dV"]
                        try:
                            dV_proj, info_v = v_dense_project(cons_h_v, dV_task, lambda_ridge, device="cpu", compute_dtype=compute_dtype)
                        except RuntimeError:
                            dV_proj, info_v = cg_v(cons_h_v, dV_task, lambda_ridge, cg_maxit, cg_tol, device="cpu", compute_dtype=compute_dtype)

                        with torch.no_grad():
                            layer_target = model_target.model.layers[li].self_attn
                            WV_t = layer_target.v_proj.weight.data.to(compute_dtype)
                            kvh = h % kv_heads
                            v_rows = slice(kvh*head_dim, (kvh+1)*head_dim)
                            WV_t[v_rows, :] += dV_proj.T.to(WV_t.device)
                            layer_target.v_proj.weight.data = WV_t.to(layer_target.v_proj.weight.dtype)

                        head_stat["norm_v"] = dV_proj.norm().item()
                        head_stat["residual_norm_v"] = info_v["residual_norm"]
                        head_stat["cg_iterations"] += info_v.get("iterations", 1)
                        head_stat["params_modified"] += dV_proj.numel()

            # —— O (dense by default; fallback to CG) ——
            if merge_o and "vo" in layer_constraints and h in layer_constraints["vo"]:
                cons_h_o = layer_constraints["vo"][h]
                if "c_vec" in cons_h_o and cons_h_o["c_vec"].numel() > 0:
                    head_stat["constraints_o"] = cons_h_o["c_vec"].shape[0]
                    
                    if h in layer_task_vectors["vo"] and "dO" in layer_task_vectors["vo"][h]:
                        dO_task = layer_task_vectors["vo"][h]["dO"]
                        try:
                            dO_proj, info_o = o_dense_project(cons_h_o, dO_task, lambda_ridge, device="cpu", compute_dtype=compute_dtype)
                        except RuntimeError:
                            dO_proj, info_o = cg_o(cons_h_o, dO_task, lambda_ridge, cg_maxit, cg_tol, device="cpu", compute_dtype=compute_dtype)

                        with torch.no_grad():
                            layer_target = model_target.model.layers[li].self_attn
                            WO_t = layer_target.o_proj.weight.data.to(compute_dtype)
                            o_cols = slice(h*head_dim, (h+1)*head_dim)
                            WO_t[:, o_cols] += dO_proj.to(WO_t.device)
                            layer_target.o_proj.weight.data = WO_t.to(layer_target.o_proj.weight.dtype)

                        head_stat["norm_o"] = dO_proj.norm().item()
                        head_stat["residual_norm_o"] = info_o["residual_norm"]
                        head_stat["cg_iterations"] += info_o.get("iterations", 1)
                        head_stat["params_modified"] += dO_proj.numel()
            
            layer_stats["heads"][h] = head_stat
            total_stats["total_params_modified"] += head_stat["params_modified"]
            total_stats["total_norm_q"] += head_stat["norm_q"]
            total_stats["total_norm_k"] += head_stat["norm_k"]
            total_stats["total_norm_v"] += head_stat["norm_v"]
            total_stats["total_norm_o"] += head_stat["norm_o"]
            total_stats["total_constraint_residual"] += (head_stat["residual_norm_qk"] + 
                                                        head_stat["residual_norm_v"] + 
                                                        head_stat["residual_norm_o"])
            total_stats["total_cg_iterations"] += head_stat["cg_iterations"]
            
            print(f"    Head {h}: QK constraints={head_stat['constraints_qk']}, V constraints={head_stat['constraints_v']}, "
                  f"O constraints={head_stat['constraints_o']}")
            print(f"Q norm={head_stat['norm_q']:.4f}, K norm={head_stat['norm_k']:.4f}, V norm={head_stat['norm_v']:.4f}, O norm={head_stat['norm_o']:.4f}")
            print(f"Q residual={head_stat['residual_norm_qk']:.6f}, V residual={head_stat['residual_norm_v']:.6f}, O residual={head_stat['residual_norm_o']:.6f}")
        
        # Handle FFN-Down once per layer
        if merge_f and "ffn" in layer_constraints and layer_constraints["ffn"].get("H", torch.empty(0)).numel() > 0:
            print(f"  🔧 Handling FFN-Down constraints for layer {li}...")
            ffn_cons = layer_constraints["ffn"]
            
            dDown_T_proj = None
            info_f = None
            
            if "ffn" in layer_task_vectors and "dDown_T" in layer_task_vectors["ffn"]:
                dDown_T_task = layer_task_vectors["ffn"]["dDown_T"]
                try:
                    # Default to dense solver
                    dDown_T_proj, info_f = ffn_down_dense_project(ffn_cons, dDown_T_task, lambda_ridge, device="cpu", compute_dtype=compute_dtype)
                except RuntimeError:
                    # Fallback to CG
                    dDown_T_proj, info_f = cg_ffn_down(ffn_cons, dDown_T_task, lambda_ridge, cg_maxit, cg_tol, device="cpu", compute_dtype=compute_dtype)
            
            if dDown_T_proj is not None and info_f is not None:
                with torch.no_grad():
                    Wd_t = model_target.model.layers[li].mlp.down_proj.weight.data.to(compute_dtype)  # [d_model, d_ff]
                    Wd_t += dDown_T_proj.T.to(Wd_t.device)  # transpose back
                    model_target.model.layers[li].mlp.down_proj.weight.data = Wd_t.to(model_target.model.layers[li].mlp.down_proj.weight.dtype)
                
                # FFN stats
                layer_stats["ffn"] = {
                    "constraints": ffn_cons["H"].shape[0],
                    "norm": dDown_T_proj.norm().item(),
                    "residual_norm": info_f["residual_norm"],
                    "cg_iterations": info_f.get("iterations", 1),
                    "params_modified": dDown_T_proj.numel()
                }
                
                total_stats["total_norm_ffn"] += layer_stats["ffn"]["norm"]
                total_stats["total_constraint_residual"] += layer_stats["ffn"]["residual_norm"]
                total_stats["total_cg_iterations"] += layer_stats["ffn"]["cg_iterations"]
                total_stats["total_params_modified"] += layer_stats["ffn"]["params_modified"]
                
                print(f"  FFN-Down: constraints={layer_stats['ffn']['constraints']}, "
                      f"norm={layer_stats['ffn']['norm']:.4f}, "
                      f"residual={layer_stats['ffn']['residual_norm']:.6f}")
        
        total_stats["layer_stats"][li] = layer_stats
    
    # Cleanup temp model
    del model_R_temp
    cleanup_memory()
    
    print(f"\n✅ Optimized layer-wise head-wise null-space projection done!")
    print(f"  📊 Totals:")
    print(f"     - Total params modified: {total_stats['total_params_modified']:,}")
    if merge_q:
        print(f"     - Total Q weight change norm: {total_stats['total_norm_q']:.6f}")
    if merge_k:
        print(f"     - Total K weight change norm: {total_stats['total_norm_k']:.6f}")
    if merge_v:
        print(f"     - Total V weight change norm: {total_stats['total_norm_v']:.6f}")
    if merge_o:
        print(f"     - Total O weight change norm: {total_stats['total_norm_o']:.6f}")
    if merge_f:
        print(f"     - Total FFN weight change norm: {total_stats['total_norm_ffn']:.6f}")
    print(f"     - Sum of constraint residuals: {total_stats['total_constraint_residual']:.6f}")
    print(f"     - Total CG iterations: {total_stats['total_cg_iterations']}")
    
    return total_stats


# ========== Entry point ==========

def main():
    parser = argparse.ArgumentParser(description="Efficient layer-wise head-wise null-space projection merging — supports complete Q/K/V/O/FFN constraints")
    
    # Basic paths
    parser.add_argument("--base", type=str,
                       default="/opt/data/private/hzhcode/huggingface/models/Qwen/Qwen2.5-7B")
    parser.add_argument("--instruct", type=str,
                       default="/opt/data/private/hzhcode/huggingface/models/Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--target", type=str,
                       default="/opt/data/private/hzhcode/huggingface/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    
    # Data & constraint params
    parser.add_argument("--texts_r", type=str, required=True, help="Path to JSON samples")
    parser.add_argument("--max_samples_r", type=int, default=10, help="Max number of samples")
    parser.add_argument("--neigh_radius", type=int, default=2, help="Neighborhood radius around boundary tokens")
    parser.add_argument("--q_rows_per_text", type=int, default=8, help="Rows per text for Q constraints")
    parser.add_argument("--k_rows_per_text", type=int, default=8, help="Rows per text for K constraints")
    
    # Layer & head config
    parser.add_argument("--layers_tail", type=int, default=2, help="Operate on last N layers")
    parser.add_argument("--heads", type=str, default="all", help="Heads to operate on ('all' or comma-separated indices)")
    
    # Weights & solvers
    parser.add_argument("--w_q", type=float, default=1.0, help="Weight for Q constraints")
    parser.add_argument("--w_k", type=float, default=1.0, help="Weight for K constraints")
    parser.add_argument("--scaling_factor", type=float, default=1.0, help="Task vector scaling factor")
    parser.add_argument("--lambda_ridge", type=float, default=1e-4, help="Ridge parameter")
    parser.add_argument("--cg_maxit", type=int, default=100, help="Max CG iterations")
    parser.add_argument("--cg_tol", type=float, default=1e-5, help="CG convergence tolerance")
    
    # Compute config
    parser.add_argument("--compute_precision", type=str, choices=["fp32", "fp64"], default="fp32",
                       help="Compute precision")
    # Multi-device config
    parser.add_argument("--qk_device", type=str, default="auto",
                       help="Device for QK constraints ('auto', 'cpu', 'cuda:0', 'cuda:1', etc.)")
    parser.add_argument("--vo_device", type=str, default="auto",
                       help="Device for VO constraints ('auto', 'cpu', 'cuda:0', 'cuda:1', etc.)")
    parser.add_argument("--ffn_device", type=str, default="auto",
                       help="Device for FFN constraints ('auto', 'cpu', 'cuda:0', 'cuda:1', etc.)")
    
    # Hook config
    parser.add_argument("--use_hooks", action="store_true", default=True,
                       help="Use hooks to capture precise internal layer features (default: True)")
    parser.add_argument("--no_hooks", action="store_true",
                       help="Disable hooks and use the original feature extraction")
    parser.add_argument("--max_seq_len", type=int, default=5120,
                       help="Max allowed sequence length; samples longer than this are skipped (default: 5120)")
    
    # Unified parameter selection (e.g., from an ultimate merge script)
    parser.add_argument("--merge_types", type=str, default="qk",
                       help="Merge types: any combination of q/k/v/o/f (e.g., 'qk', 'qkvo', 'qkvof', 'f'; default: qk)")
    
    parser.add_argument("--v_rows_per_text", type=int, default=4, help="Rows per text for V constraints")
    parser.add_argument("--o_rows_per_text", type=int, default=4, help="Rows per text for O constraints")
    parser.add_argument("--ffn_rows_per_text", type=int, default=4, help="Rows per text for FFN-Down constraints")
    
    parser.add_argument("--readout_dirs", type=int, default=2, help="Number of readout directions c per head/layer")
    parser.add_argument("--w_v", type=float, default=1.0, help="Weight for V constraints")
    parser.add_argument("--w_o", type=float, default=1.0, help="Weight for O constraints")
    parser.add_argument("--w_ffn", type=float, default=1.0, help="Weight for FFN-Down constraints")
    
    # Output config
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--save_merged_model", action="store_true", help="Save the merged model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Compute precision
    compute_dtype = torch.float64 if args.compute_precision == "fp64" else torch.float32
    
    print("🚀 Efficient layer-wise head-wise null-space projection merging — supports complete Q/K/V/O/FFN constraints")
    print("=" * 70)
    print(f"Base: {args.base}")
    print(f"Instruct: {args.instruct}")
    print(f"Target: {args.target}")
    print(f"Task vector scaling factor: {args.scaling_factor}")
    print(f"Compute precision: {args.compute_precision.upper()}")
    print(f"Devices: QK={args.qk_device}, VO={args.vo_device}, FFN={args.ffn_device}")
    
    # Hook mode
    use_hooks = args.use_hooks and not args.no_hooks
    print(f"Feature extraction: {'Hook-based (recommended)' if use_hooks else 'Original'}")

    start_time = time.time()

    # Load models (on CPU)
    print("\n📥 Loading models onto CPU...")
    model_base = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
    ).eval()
    
    model_instruct = AutoModelForCausalLM.from_pretrained(
        args.instruct, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
    ).eval()
    
    model_target = AutoModelForCausalLM.from_pretrained(
        args.target, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
    ).eval()
    
    tokenizer = AutoTokenizer.from_pretrained(args.target, use_fast=True, trust_remote_code=True)

    # Config
    num_layers = model_target.config.num_hidden_layers
    n_heads = model_target.config.num_attention_heads

    selected_layers = list(range(num_layers - args.layers_tail, num_layers))
    if args.heads == "all":
        selected_heads = list(range(n_heads))
    else:
        selected_heads = [int(x) for x in args.heads.split(",")]

    print(f"📋 Selection:")
    print(f"  Layers: {selected_layers}")
    print(f"  Heads: {len(selected_heads)}/{n_heads}")

    # Read data
    texts_R = read_json_samples(args.texts_r, tokenizer, args.max_samples_r)
    print(f"📊 Number of JSON samples: {len(texts_R)}")

    # Run optimized null-space projection
    print("\n🔬 Running optimized layer-wise head-wise null-space projection...")
    stats = optimized_layerwise_headwise_nullspace_projection(
        model_base, model_instruct, model_target,
        texts_R, tokenizer,
        selected_layers, selected_heads,
        args.neigh_radius, args.lambda_ridge, args.cg_maxit, args.cg_tol,
        args.scaling_factor, compute_dtype,
        # Merge types
        args.merge_types,
        # QK
        args.q_rows_per_text, args.k_rows_per_text, args.w_q, args.w_k,
        # VO
        args.v_rows_per_text, args.o_rows_per_text, args.w_v, args.w_o,
        # FFN
        args.ffn_rows_per_text, args.w_ffn, args.readout_dirs, args.seed,
        # Devices
        args.qk_device, args.vo_device, args.ffn_device,
        # Hooks
        use_hooks
    )

    # Save config & stats
    end_time = time.time()
    config_data = {
        "base": args.base, "instruct": args.instruct, "target": args.target,
        "layers": selected_layers, "heads": selected_heads,
        "compute_precision": args.compute_precision,
        "qk_device": args.qk_device,
        "vo_device": args.vo_device,
        "ffn_device": args.ffn_device,
        "use_hooks": use_hooks,
        "neigh_radius": args.neigh_radius,
        "merge_types": args.merge_types,
        "q_rows_per_text": args.q_rows_per_text, "k_rows_per_text": args.k_rows_per_text,
        "w_q": args.w_q, "w_k": args.w_k,
        "v_rows_per_text": args.v_rows_per_text, "o_rows_per_text": args.o_rows_per_text,
        "w_v": args.w_v, "w_o": args.w_o,
        "ffn_rows_per_text": args.ffn_rows_per_text, "w_ffn": args.w_ffn,
        "readout_dirs": args.readout_dirs,
        "scaling_factor": args.scaling_factor,
        "lambda_ridge": args.lambda_ridge,
        "cg_maxit": args.cg_maxit, "cg_tol": args.cg_tol,
        "runtime_seconds": end_time - start_time,
        "optimization": "layerwise_batched_vectorized_qkvo_ffn",
        "stats": stats
    }

    with open(os.path.join(args.out_dir, "optimized_qkvo_ffn_config.json"), "w", encoding="utf-8") as f:
        json.dump(config_data, f, ensure_ascii=False, indent=2)

    # Save merged model
    if args.save_merged_model:
        out_model = os.path.join(args.out_dir, "merged_qkvo_ffn")
        print(f"💾 Saving merged model to: {out_model}")
        model_target.save_pretrained(out_model)
        tokenizer.save_pretrained(out_model)

    print(f"\n✅ Finished! Elapsed: {end_time - start_time:.1f}s")
    print(f"📁 Output directory: {args.out_dir}")
    print(f"🚀 Improvements: supports complete Q/K/V/O/FFN constraints; constraint building reduces from O(N_text×H_head) to O(N_text); vectorized A/AT computations")


if __name__ == "__main__":
    main()
