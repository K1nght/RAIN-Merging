#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fast True-Forward Instruction-Attention QP (Q/K only) with Alignment & Leak

Two-pass pipeline (still O(S) forwards):
1) Pass-1 (Anchor): for every sample, forward ORIGINAL model -> collect per-(layer,head)
   alignment a0 (I->R) and leak u0 (I->~R).
2) Apply the WHOLE projected task-vector ONCE using alpha_prior (coupled QK by default).
3) Pass-2 (Post): forward again for every sample -> collect a1, u1; build
   Δa = a1 - a0 (alignment gain), Δu = u1 - u0 (leak change).
4) Diagonal design → build per-head diagonal H and linear term b:
     b_i = Σ_s ( Δa_i^{(s)} - ρ * Δu_i^{(s)} )
     H_i = S * λ_H + μ_H * Σ_s u1_i^{(s)} + ε
   (ε>0 small jitter)
5) Solve box QP with L2 prior (+ optional L1). Diagonal shortcut yields per-dim closed-form-like step.
6) Save α*, scale task-vectors, optional save model with applied α*.

Notes
- This script only touches Q/K (routing). V/O/FFN are unchanged.
- By default Q&K are COUPLED (a single α per (layer,head)). Use --decouple_qk to split α_Q and α_K.
- "Alignment" a = sum attention mass from I to R. "Leak" u = sum attention mass from I to NOT R (computed by row-sum minus a).
"""

import os, re, json, math, time, argparse, pickle
from typing import Dict, Any, List, Tuple, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# ===========================
# Utilities: math & masking
# ===========================
def stable_softmax_with_masks(scores: torch.Tensor, causal=True, attn_mask=None):
    """
    scores: [H, T, T] (float32 recommended)
    returns attention [H, T, T] with causal & padding masks applied before softmax.
    """
    H, T, _ = scores.shape
    dev = scores.device
    valid = torch.tril(torch.ones(T, T, device=dev, dtype=torch.bool)) if causal \
            else torch.ones(T, T, device=dev, dtype=torch.bool)
    if attn_mask is not None:
        key_keep = attn_mask.to(torch.bool)  # [T]
        valid = valid & key_keep.unsqueeze(0).expand(T, T)

    sm = scores.clone()
    valid3 = valid.unsqueeze(0).expand(H, T, T)
    sm[~valid3] = torch.finfo(scores.dtype).min

    row_valid = valid.any(dim=-1).unsqueeze(0).expand(H, T)
    row_max = sm.max(dim=-1, keepdim=True).values
    row_max = torch.where(row_valid.unsqueeze(-1), row_max, torch.zeros_like(row_max))

    z = torch.exp(sm - row_max) * valid3.to(sm.dtype)
    denom = z.sum(dim=-1, keepdim=True)

    empty = denom == 0
    if empty.any():
        eye = torch.eye(T, device=dev, dtype=z.dtype)
        z = torch.where(empty, eye.unsqueeze(0).expand_as(z), z)
        denom = torch.where(empty, torch.ones_like(denom), denom)
    return z / denom


# ===========================
# JSONL and data loading helpers
# ===========================
def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load JSONL file (one JSON object per line)"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"⚠️  Line {line_no} JSON decode error: {e}")
                continue
    return data

def build_full_text_from_sample(sample: Dict[str, Any], tokenizer) -> str:
    """Build full text following DeepSeek-R1 format"""
    prompt = sample.get("prompt", "")
    reason = sample.get("reason", "")
    response = sample.get("response", "")
    
    # Try to apply chat template
    try:
        pre = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, 
            add_generation_prompt=True
        )
    except Exception:
        pre = prompt
    
    # Build complete text
    full_text = f"{pre}{reason}\n</think>\n\n{response}"
    return full_text

def extract_spans_from_related_list(related_prompt_list: List[Any]) -> List[str]:
    """Extract span texts from related_prompt_list structure (Legacy function, kept for compatibility)"""
    spans = []
    if not related_prompt_list:
        return spans
    
    for rel_item in related_prompt_list:
        if isinstance(rel_item, dict):
            # New format: dict with prompt_spans and response_spans
            for span_text in rel_item.get("prompt_spans", []):
                if isinstance(span_text, str) and span_text.strip():
                    spans.append(span_text)
            for span_text in rel_item.get("response_spans", []):
                if isinstance(span_text, str) and span_text.strip():
                    spans.append(span_text)
        elif isinstance(rel_item, list):
            # Legacy format: nested lists
            for inner_item in rel_item:
                if isinstance(inner_item, str) and inner_item.strip():
                    spans.append(inner_item)
        elif isinstance(rel_item, str) and rel_item.strip():
            # Direct string
            spans.append(rel_item)
    
    return spans


# The old function has been removed; please use extract_instruction_wise_spans instead.


def extract_instruction_wise_spans(sample: Dict[str, Any], tokenizer) -> Dict[str, Any]:
    """
    Properly handle multiple-instruction cases: each instruction has its own related and unrelated spans.
    
    Returns:
        {
            'instructions': List[Dict],  # detailed info for each instruction
            'global_unrelated_spans': List[int],  # global unrelated spans
            'global_unrelated_count': int,        # number of global unrelated spans
        }
        
        Each instruction contains:
        {
            'instruction_id': str,
            'instruction_spans': List[int],  # I_k
            'related_spans': List[int],      # R_k 
            'unrelated_spans': List[int],    # U_k
            'related_spans_count': int,      # |R_k|
            'unrelated_spans_count': int,    # |U_k|
        }
    """
    
    # Build full text
    full_text = build_full_text_from_sample(sample, tokenizer)
    
    instructions = []
    related_prompt_list = sample.get("related_prompt_list", [])
    
    # Process each instruction
    for item in related_prompt_list:
        if not isinstance(item, dict):
            continue
            
        instruction_id = item.get("instruction_id", "")
        
        # Extract various spans
        prompt_spans = item.get("prompt_spans", [])
        response_spans = item.get("response_spans", [])
        unrelated_response_spans = item.get("unrelated_response_spans", [])
        
        # instruction spans = prompt_spans
        instruction_token_spans = []
        for span_text in prompt_spans:
            if isinstance(span_text, str) and span_text.strip():
                spans = _find_token_spans(full_text, span_text, tokenizer)
                instruction_token_spans.extend(spans)
        
        # related spans = prompt_spans + response_spans (for computing a_k)
        related_token_spans = []
        for span_text in prompt_spans + response_spans:
            if isinstance(span_text, str) and span_text.strip():
                spans = _find_token_spans(full_text, span_text, tokenizer)
                related_token_spans.extend(spans)
        
        # unrelated spans = unrelated_response_spans (for computing u_k)
        unrelated_token_spans = []
        for span_text in unrelated_response_spans:
            if isinstance(span_text, str) and span_text.strip():
                spans = _find_token_spans(full_text, span_text, tokenizer)
                unrelated_token_spans.extend(spans)
        
        # Convert to unique token indices
        instruction_indices = _union_indices(instruction_token_spans)
        related_indices = _union_indices(related_token_spans)
        unrelated_indices = _union_indices(unrelated_token_spans)
        
        instructions.append({
            'instruction_id': instruction_id,
            'instruction_spans': instruction_indices,  # I_k
            'related_spans': related_indices,          # R_k
            'unrelated_spans': unrelated_indices,      # U_k (instruction-level)
            'related_spans_count': len(related_token_spans),      # for normalization
            'unrelated_spans_count': len(unrelated_token_spans),  # for normalization
        })
    
    # Process global unrelated_response_spans (shared by all instructions)
    global_unrelated_spans = sample.get("unrelated_response_spans", [])
    global_unrelated_token_spans = []
    
    for span_text in global_unrelated_spans:
        if isinstance(span_text, str) and span_text.strip():
            spans = _find_token_spans(full_text, span_text, tokenizer)
            global_unrelated_token_spans.extend(spans)
    
    global_unrelated_indices = _union_indices(global_unrelated_token_spans)
    
    return {
        'instructions': instructions,
        'global_unrelated_spans': global_unrelated_indices,
        'global_unrelated_count': len(global_unrelated_token_spans),
    }


# ===========================
# Text span helpers
# ===========================
def _find_token_spans(text: str, substr: str, tokenizer) -> List[Tuple[int, int]]:
    """Find all occurrences of substr as token spans [start,end), compatible with offset mapping"""
    if not substr or not substr.strip():
        return []
    
    spans = []
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=True)
    offs = enc["offset_mapping"]

    def c2t(char_pos):
        """Convert character position to token index"""
        for i, (s, e) in enumerate(offs):
            if s <= char_pos < e:
                return i
        return None

    # Find all character-level matches
    for m in re.finditer(re.escape(substr), text):
        cs, ce = m.span()
        ts = c2t(cs)
        te = c2t(ce - 1)
        if ts is not None and te is not None:
            spans.append((ts, te + 1))
    
    return spans


def _union_indices(spans: List[Tuple[int, int]]) -> List[int]:
    bag = set()
    for s, e in spans:
        for t in range(s, e):
            bag.add(t)
    return sorted(bag)


# ===========================
# Forward: compute A per layer
# ===========================

@torch.no_grad()
def forward_attn_per_layer_optimized(
    model, tokenizer, full_text: str, layers: List[int], device=None, verbose=False
):
    """
    🚀 Optimized attention extraction using hooks to grab attention directly during the forward pass.
    
    Based on tests, this method is >50× faster than the legacy approach and more memory efficient.
    
    Args:
        model: transformer model
        tokenizer: tokenizer
        full_text: input text
        layers: list of layer indices to extract
        device: compute device
        verbose: whether to print debug info
    
    Returns:
        attn_by_layer: {layer: [H,T,T] float32}
        input_ids: List[int], attn_mask: [T] (cpu)
    """
    if device is None:
        device = next(model.parameters()).device
    
    if verbose:
        print(f"      🚀 Using hooks for optimized attention extraction")
    
    # Tokenize input
    encoding = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding.get("attention_mask", torch.ones_like(input_ids)).to(device)
    
    T = input_ids.shape[1]
    if verbose:
        print(f"      🔤 Sequence length: {T}")
    
    # Set up attention extractors
    attention_cache = {}
    hooks = []
    
    def create_attention_hook(layer_idx: int):
        """Create an attention hook for a specific layer"""
        def attention_hook(module, input, output):
            try:
                # Retrieve attention weights from output
                if isinstance(output, tuple) and len(output) >= 2:
                    attention_weights = output[1]
                    if attention_weights is not None and isinstance(attention_weights, torch.Tensor) and attention_weights.dim() == 4:
                        # Capture to CPU immediately
                        attention_cache[layer_idx] = attention_weights[0].detach().cpu()

                        if verbose:
                            print(f"        ✅ Layer {layer_idx}: {attention_cache[layer_idx].shape}")

                        # Replace the GPU tensor with None in the returned output so the
                        # model's all_self_attns accumulator stores None instead of holding
                        # a live GPU tensor until the full forward completes.
                        return (output[0], None) + output[2:]

            except Exception as e:
                if verbose:
                    print(f"        ⚠️  Layer {layer_idx} hook error: {e}")

        return attention_hook
    
    # Register hooks for selected layers
    for layer_idx in layers:
        if layer_idx < len(model.model.layers):
            layer = model.model.layers[layer_idx]
            if hasattr(layer, 'self_attn'):
                hook = layer.self_attn.register_forward_hook(create_attention_hook(layer_idx))
                hooks.append(hook)
    
    try:
        # Run forward pass; hooks will capture attention
        with torch.no_grad():
            _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,  # ensure attention is computed
                use_cache=False
            )
        
        # Clean up hooks
        for hook in hooks:
            hook.remove()
        
        if verbose:
            print(f"      ✅ Successfully extracted attention for {len(attention_cache)} layers")
        
        # Return format consistent with original function
        input_ids_list = input_ids[0].cpu().tolist()
        attention_mask_list = attention_mask[0].cpu().tolist()
        
        return attention_cache, input_ids_list, attention_mask_list
        
    except Exception as e:
        # Clean up hooks
        for hook in hooks:
            hook.remove()
        
        if verbose:
            print(f"      ❌ Hook method failed: {e}, falling back to original method")
        
        # Fallback to original method
        return forward_attn_per_layer_original(model, tokenizer, full_text, layers, device, verbose)


@torch.no_grad()  
def forward_attn_per_layer_original(model, tokenizer, full_text: str, layers: List[int], device=None, verbose=False):
    """Original attention extraction function, used as a fallback."""
    # Choose the device for attention computation; prefer the provided device
    if device is not None:
        attention_device = torch.device(device)
    else:
        # If no device specified, use the first available CUDA device
        attention_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if verbose:
        print(f"      🔤 Tokenizing text (length: {len(full_text)} chars)...")
        print(f"      🎯 Attention computation device: {attention_device}")

    # Tokenize
    enc = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
    model_device = next(model.parameters()).device
    input_ids = enc["input_ids"].to(model_device)
    attn_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(model_device)

    T = input_ids.shape[1]
    if verbose:
        print(f"      📊 Token length: {T}")

    attn_by_layer = {}

    # Cache key model info
    try:
        d_model = model.config.hidden_size
        H = model.config.num_attention_heads
        H_kv = getattr(model.config, "num_key_value_heads", H)
        d_head = d_model // H
        if verbose:
            print(f"      🧠 Model config: {H} heads, {d_head} d_head, {H_kv} kv_heads")
    except Exception as e:
        if verbose:
            print(f"      ❌ Failed to get model config: {e}")
        # Provide defaults or return empty result
        return {}, input_ids[0].cpu().tolist(), attn_mask[0].cpu().tolist()

    with torch.no_grad():
        if verbose:
            print(f"      🚀 Computing attention for layers: {layers}")

        # Forward once and cache hidden states
        try:
            hidden_states_cache = {}
            
            x = model.model.embed_tokens(input_ids[0])  # [T, d]
            hidden_states_cache[-1] = x  # embedding layer
            
            max_layer_needed = max(layers) if layers else len(model.model.layers) - 1
            
            for l in range(max_layer_needed + 1):
                layer_mod = model.model.layers[l]
                x = layer_mod(x.unsqueeze(0), attention_mask=attn_mask)[0].squeeze(0)
                hidden_states_cache[l] = x
            
            # Now compute attention for requested layers
            for l in layers:
                if l >= len(model.model.layers):
                    if verbose:
                        print(f"        ⚠️  Layer {l} >= model layers ({len(model.model.layers)}), skipping")
                    continue
                
                try:
                    if verbose:
                        print(f"        📊 Processing layer {l}...")
                    
                    layer_mod = model.model.layers[l]
                    
                    # Get input to this layer
                    if l == 0:
                        x_input = model.model.embed_tokens(input_ids[0])  # [T, d]
                    else:
                        x_input = hidden_states_cache.get(l-1)
                        if x_input is None:
                            if verbose:
                                print(f"          ⚠️  No cached hidden state for layer {l-1}")
                            continue
                    
                    # Layer norm
                    x_norm = layer_mod.input_layernorm(x_input.unsqueeze(0))[0]  # [T, d]
                    
                    # Self-attention
                    sa = layer_mod.self_attn
                    x_norm = x_norm.to(sa.q_proj.weight.dtype)
                    
                    Q_lin = sa.q_proj(x_norm)
                    K_lin = sa.k_proj(x_norm)
                    
                    # Move to attention device for computation
                    Q_lin = Q_lin.to(attention_device, torch.float32)
                    K_lin = K_lin.to(attention_device, torch.float32)
                    attn_mask_device = attn_mask[0].to(attention_device)
                    
                    # Reshape
                    Q = Q_lin.view(T, H, d_head).permute(1, 0, 2).contiguous()  # [H,T,d]
                    K = K_lin.view(T, H_kv, d_head).permute(1, 0, 2).contiguous()  # [H_kv,T,d]
                    
                    if H_kv < H:
                        rep = H // H_kv
                        K = K.repeat_interleave(rep, dim=0)  # [H,T,d]
                    
                    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_head)  # [H,T,T]
                    
                    # Apply masks and softmax
                    A = stable_softmax_with_masks(scores, causal=True, attn_mask=attn_mask_device)
                    
                    # Move back to CPU and store
                    attn_by_layer[l] = A.detach().cpu()
                    
                    # Clean up intermediate tensors
                    del Q_lin, K_lin, Q, K, scores, A
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    if verbose:
                        print(f"          ✅ Layer {l}: {attn_by_layer[l].shape}")
                        
                except Exception as layer_error:
                    if verbose:
                        print(f"          ❌ Layer {l} failed: {layer_error}")
                    continue
            
            # Clean up cached hidden states
            del hidden_states_cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                        
        except Exception as e:
            if verbose:
                print(f"      ❌ Forward pass failed: {e}")
            return {}, input_ids[0].cpu().tolist(), attn_mask[0].cpu().tolist()

    input_ids_list = input_ids[0].cpu().tolist()
    attn_mask_list = attn_mask[0].cpu().tolist()
    
    if verbose:
        print(f"      ✅ Computed attention for {len(attn_by_layer)} layers")
    
    return attn_by_layer, input_ids_list, attn_mask_list


@torch.no_grad()
def forward_attn_per_layer(model, tokenizer, full_text: str, layers: List[int], device=None, verbose=False):
    """
    Returns:
      attn_by_layer: {layer: [H,T,T] float32}
      input_ids: List[int], attn_mask: [T] (cpu)
    """
    # Choose the device for attention computation; prefer the provided device
    if device is not None:
        attention_device = torch.device(device)
    else:
        # If no device specified, use the first available CUDA device
        attention_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if verbose:
        print(f"      🔤 Tokenizing text (length: {len(full_text)} chars)...")
        print(f"      🎯 Attention computation device: {attention_device}")

    enc = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
    # Run forward on the model's first-layer device
    model_device = next(model.parameters()).device
    input_ids = enc["input_ids"].to(model_device)
    attn_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(model_device)

    T = input_ids.shape[1]
    if verbose:
        print(f"      📏 Sequence length: {T} tokens")
        if torch.cuda.is_available():
            print(f"      💾 GPU before forward: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

    if verbose:
        print(f"      🧠 Forward pass through model...")
    with torch.no_grad():
        out = model.model(input_ids=input_ids, output_hidden_states=True)
        hidden_states = out.hidden_states  # tuple len = n_layers+1
    
    if verbose and torch.cuda.is_available():
        print(f"      💾 GPU after forward: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

    d_model = model.config.hidden_size
    H = model.config.num_attention_heads
    H_kv = getattr(model.config, "num_key_value_heads", H)
    d_head = d_model // H

    if verbose:
        print(f"      🧮 Computing attention for {len(layers)} layers...")

    attn_by_layer = {}
    for li, l in enumerate(layers):
        if verbose and len(layers) > 5 and li % max(1, len(layers)//5) == 0:
            print(f"        🔄 Layer {l} ({li+1}/{len(layers)})")
            if torch.cuda.is_available():
                print(f"          💾 GPU: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
        
        layer_mod = model.model.layers[l]
        sa = layer_mod.self_attn
        
        # Compute hidden states on the model device, then move to the attention device
        x = layer_mod.input_layernorm(hidden_states[l])       # [B,T,d]
        x = x[0].to(sa.q_proj.weight.dtype)                   # [T,d]
        
        # Linear transforms on the original device
        Q_lin = sa.q_proj(x)                                  # [T, H*d]
        K_lin = sa.k_proj(x)                                  # [T, H_kv*d]
        
        # Move to chosen attention device
        Q_lin = Q_lin.to(attention_device, torch.float32)
        K_lin = K_lin.to(attention_device, torch.float32)
        attn_mask_device = attn_mask[0].to(attention_device)
        
        Q = Q_lin.view(T, H, d_head).permute(1, 0, 2).contiguous()         # [H,T,d]
        K = K_lin.view(T, H_kv, d_head).permute(1, 0, 2).contiguous()      # [H_kv,T,d]
        if H_kv < H:
            rep = H // H_kv
            K = K.repeat_interleave(rep, dim=0)               # [H,T,d]

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_head)  # [H,T,T]
        A = stable_softmax_with_masks(scores, causal=True, attn_mask=attn_mask_device)
        attn_by_layer[l] = A.detach().cpu()
    
    if verbose:
        print(f"      ✅ Attention computation complete")
        if torch.cuda.is_available():
            print(f"      💾 GPU final: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

    return attn_by_layer, enc["input_ids"][0].tolist(), attn_mask[0].cpu()


def compute_align_leak_vectors_instruction_wise(
    model, tok, full_text, selected_layers, selected_heads, 
    instructions_data, global_unrelated_spans, global_unrelated_count, device=None, verbose=False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Correctly compute attention for multiple instructions: compute per-instruction, then aggregate.
    Memory-optimized version: get attention for all layers in a single forward, then free immediately.
    
    Args:
        instructions_data: List[Dict] span info for each instruction
        global_unrelated_spans: global unrelated spans
        global_unrelated_count: number of global unrelated spans
    
    Returns:
        (a_total, u_total): aggregated alignment and leak vectors
    """
    # 🚀 Get all layer attention matrices in one shot (try hooks method)
    try:
        A_by_layer, _, _ = forward_attn_per_layer_optimized(model, tok, full_text, selected_layers, device=device, verbose=verbose)
        if verbose:
            print(f"      ✅ Successfully extracted attention using hooks method")
    except Exception as e:
        if verbose:
            print(f"      ⚠️  Hooks method failed, falling back to original: {e}")
        A_by_layer, _, _ = forward_attn_per_layer(model, tok, full_text, selected_layers, device=device, verbose=verbose)
    
    m_heads = len(selected_layers) * len(selected_heads)
    dev = next(model.parameters()).device
    
    a_total = torch.zeros(m_heads, dtype=torch.float32, device=dev)
    u_total = torch.zeros(m_heads, dtype=torch.float32, device=dev)
    
    # Pre-extract and validate span indices to avoid recompute
    processed_instructions = []
    T = None
    for layer_key in A_by_layer:
        T = A_by_layer[layer_key].shape[1]  # sequence length
        break
    
    if T is None:
        return a_total, u_total
    
    for instr in instructions_data:
        # Validate all indices
        I_k_valid = [i for i in instr['instruction_spans'] if 0 <= i < T]
        R_k_valid = [i for i in instr['related_spans'] if 0 <= i < T]  
        U_k_valid = [i for i in instr['unrelated_spans'] if 0 <= i < T]
        
        processed_instructions.append({
            'I_k': I_k_valid,
            'R_k': R_k_valid,
            'U_k': U_k_valid,
            'related_count_k': instr['related_spans_count'],
            'unrelated_count_k': instr['unrelated_spans_count']
        })
    
    # Validate global unrelated spans
    U_global_valid = [i for i in global_unrelated_spans if 0 <= i < T]
    
    # 🧠 Efficiently process all layer/head attention
    head_idx = 0
    for layer_idx, l in enumerate(selected_layers):
        A_layer = A_by_layer[l]  # [H, T, T]
        
        for head_idx_in_layer, h in enumerate(selected_heads):
            A_h = A_layer[h]  # [T, T]
            
            # Compute alignment & leak for this head across all instructions
            a_head_total = 0.0
            u_head_total = 0.0
            
            for instr in processed_instructions:
                I_k = instr['I_k']
                R_k = instr['R_k']
                U_k = instr['U_k']
                related_count_k = instr['related_count_k']
                unrelated_count_k = instr['unrelated_count_k']

                # Compute I_rows once and reuse for all three span queries
                I_rows = A_h[I_k] if len(I_k) > 0 else None  # [|I_k|, T]

                # a_k: I_k → R_k
                a_k = 0.0
                if I_rows is not None and len(R_k) > 0:
                    a_k = I_rows[:, R_k].sum().item() / max(1, related_count_k)

                # u_k (instruction-level): I_k → U_k
                u_k_instr = 0.0
                if I_rows is not None and len(U_k) > 0:
                    u_k_instr = I_rows[:, U_k].sum().item() / max(1, unrelated_count_k)

                # u_k_global: I_k → global unrelated
                u_k_global = 0.0
                if I_rows is not None and len(U_global_valid) > 0:
                    u_k_global = I_rows[:, U_global_valid].sum().item() / max(1, global_unrelated_count)

                a_head_total += a_k
                u_head_total += u_k_instr + u_k_global
            
            # Store results for this head
            a_total[head_idx] = a_head_total
            u_total[head_idx] = u_head_total
            head_idx += 1
    
    # 💾 After processing, free attention matrices immediately to release memory
    del A_by_layer, processed_instructions
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return a_total, u_total


# ===========================
# Inplace add/rollback QK (supports couple/decouple)
# ===========================
@torch.no_grad()
def add_qk_alpha_inplace(model, projected_tv, axes, alpha_vec,
                         selected_layers, selected_heads,
                         couple_qk: bool = True) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Apply α to Q/K slices in-place; return deltas for rollback.
    If couple_qk=True, axes types={'QK'} and each α controls both Q & K of that head.
    If couple_qk=False, axes types={'Q','K'} and α per type.
    """
    H_all = model.config.num_attention_heads
    d_model = model.config.hidden_size
    hD = d_model // H_all
    H_kv = getattr(model.config, "num_key_value_heads", H_all)

    deltas = []
    col_map = { (l,h,t): i for i,(l,h,t) in enumerate(axes["flat_index"]) }
    types = set(axes["types"])

    for l in selected_layers:
        sa = model.model.layers[l].self_attn
        WQ = sa.q_proj.weight
        WK = sa.k_proj.weight
        for h in selected_heads:
            # Q
            dQ = projected_tv.get("qk", {}).get(l, {}).get(h, {}).get("dQ_proj", None)
            if dQ is not None and (("Q" in types) or ("QK" in types)):
                idx_key = (l,h,"QK") if ("QK" in types) else (l,h,"Q")
                i = col_map.get(idx_key)
                if i is not None:
                    a = float(alpha_vec[i].item())
                    if a != 0.0:
                        ds = (dQ.T.to(WQ.device, WQ.dtype)) * a    # [hD, d_model]
                        q_start, q_end = h*hD, (h+1)*hD
                        delta = torch.zeros_like(WQ)
                        delta[q_start:q_end, :] = ds
                        WQ.data.add_(delta)
                        deltas.append((WQ, delta))
            # K
            dK = projected_tv.get("qk", {}).get(l, {}).get(h, {}).get("dK_proj", None)
            if dK is not None and (("K" in types) or ("QK" in types)):
                idx_key = (l,h,"QK") if ("QK" in types) else (l,h,"K")
                i = col_map.get(idx_key)
                if i is not None:
                    a = float(alpha_vec[i].item())
                    if a != 0.0:
                        ds = (dK.T.to(WK.device, WK.dtype)) * a
                        kvh = h % H_kv
                        k_start, k_end = kvh*hD, (kvh+1)*hD
                        delta = torch.zeros_like(WK)
                        delta[k_start:k_end, :] = ds
                        WK.data.add_(delta)
                        deltas.append((WK, delta))
    return deltas


@torch.no_grad()
def rollback(deltas: List[Tuple[torch.Tensor, torch.Tensor]]):
    """Rollback parameter modifications; supports CPU/GPU deltas."""
    for p, d in deltas:
        # Ensure delta is on the correct device/dtype
        if d.device != p.device or d.dtype != p.dtype:
            d = d.to(p.device, p.dtype)
        p.data.sub_(d)


@torch.no_grad()
def add_alpha_inplace_with_vo_ffn_cpu_optimized(model,
                                               projected_tv: Dict[str, Any],
                                               axes: Dict[str, Any],
                                               alpha_vec: torch.Tensor,
                                               selected_layers: List[int],
                                               selected_heads: List[int],
                                               couple_qk: bool = True,
                                               verbose: bool = False) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Fully-CPU merge version: merge all task-vector contributions on CPU, then assign once to the GPU model to avoid OOM.
    Strategy:
    1) QK params: still apply directly (smaller memory footprint)
    2) VO/FFN params: merge entirely on CPU, then replace GPU parameters once
    Returns all deltas for rollback.
    """
    deltas = []

    if verbose:
        print("  🚀 Applying Q/K parameters...")
    # 1) Apply Q/K directly (relatively small memory)
    deltas += add_qk_alpha_inplace(model, projected_tv, axes, alpha_vec,
                                   selected_layers, selected_heads, couple_qk=couple_qk)

    # 2) VO/FFN params: CPU-merged approach
    has_vo = "vo" in projected_tv and len(projected_tv["vo"]) > 0
    has_ffn = "ffn" in projected_tv and len(projected_tv["ffn"]) > 0
    if not (has_vo or has_ffn):
        return deltas

    head_alpha = _derive_head_alpha_from_qk(alpha_vec, axes, selected_layers, selected_heads, couple_qk)
    layer_alpha = _derive_layer_alpha_from_heads(head_alpha, selected_layers, selected_heads)

    # --- VO CPU full-merge ---
    if has_vo:
        if verbose:
            print("  💾 Applying V/O parameters (full CPU merge)...")
        
        d_model = model.config.hidden_size
        H_all = model.config.num_attention_heads
        hD = d_model // H_all
        H_kv = getattr(model.config, "num_key_value_heads", H_all)
        
        # Process VO per layer
        for l, per_h in projected_tv["vo"].items():
            layer = model.model.layers[l]
            sa = layer.self_attn
            WV = sa.v_proj.weight
            WO = sa.o_proj.weight
            
            # 1. Copy original params to CPU for merging
            WV_cpu = WV.data.cpu().float()  # work in float32 on CPU
            WO_cpu = WO.data.cpu().float()
            
            # 2. Accumulate all deltas on CPU
            for h, tv in per_h.items():
                a = head_alpha.get((l,h), None)
                if a is None or a == 0.0: 
                    continue
                    
                # V
                dV = tv.get("dV_proj", None)
                if dV is not None:
                    kvh = h % H_kv
                    v_start, v_end = kvh*hD, (kvh+1)*hD
                    WV_cpu[v_start:v_end, :] += (dV.T.cpu().float()) * a
                
                # O  
                dO = tv.get("dO_proj", None)
                if dO is not None:
                    o_start, o_end = h*hD, (h+1)*hD
                    WO_cpu[:, o_start:o_end] += (dO.cpu().float()) * a
            
            # 3. Compute actual CPU deltas (for rollback)
            v_delta_cpu = WV_cpu - WV.data.cpu().float()
            o_delta_cpu = WO_cpu - WO.data.cpu().float()
            
            # 4. Assign back to GPU once (avoid extra allocations)
            if v_delta_cpu.abs().sum() > 1e-12:  # any real change?
                WV.data.copy_(WV_cpu.to(WV.dtype).to(WV.device))
                deltas.append((WV, v_delta_cpu.to(WV.dtype)))  # keep CPU delta
                
            if o_delta_cpu.abs().sum() > 1e-12:
                WO.data.copy_(WO_cpu.to(WO.dtype).to(WO.device))
                deltas.append((WO, o_delta_cpu.to(WO.dtype)))  # keep CPU delta
            
            # 5. Explicitly free CPU temporaries
            del WV_cpu, WO_cpu, v_delta_cpu, o_delta_cpu

    # --- FFN CPU full-merge ---
    if has_ffn:
        if verbose:
            print("  🔧 Applying FFN parameters (full CPU merge)...")
            
        # Process FFN per layer
        for l, tv in projected_tv["ffn"].items():
            a = layer_alpha.get(l, None)
            if a is None or a == 0.0:
                continue
                
            layer = model.model.layers[l]
            
            # Handle each FFN matrix
            for k_name, mod_name in [
                ("dGate_proj","gate_proj"),
                ("dUp_proj","up_proj"),
                ("dDown_proj","down_proj"),
            ]:
                dW = tv.get(k_name, None)
                if dW is None:
                    continue
                    
                W = getattr(layer.mlp, mod_name).weight
                
                # 1. Copy original to CPU for merging
                W_cpu = W.data.cpu().float()
                
                # 2. Apply delta on CPU
                delta_cpu = (dW.cpu().float()) * a
                W_merged_cpu = W_cpu + delta_cpu
                
                # 3. Copy merged param back to GPU once (copy_ avoids extra allocation)
                W.data.copy_(W_merged_cpu.to(W.dtype).to(W.device))
                
                # 4. Save delta for rollback (keep on CPU to save VRAM)
                deltas.append((W, delta_cpu.to(W.dtype)))
                
                # 5. Explicitly free CPU temporaries
                del W_cpu, W_merged_cpu, delta_cpu

    if verbose:
        print(f"  ✅ Applied {len(deltas)} parameter updates in total")
                
    return deltas


@torch.no_grad()
def add_alpha_inplace_with_vo_ffn(model,
                                  projected_tv: Dict[str, Any],
                                  axes: Dict[str, Any],
                                  alpha_vec: torch.Tensor,
                                  selected_layers: List[int],
                                  selected_heads: List[int],
                                  couple_qk: bool = True) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    First apply the original Q/K in-place function, then apply:
      - VO uses the same head alpha
      - FFN uses the mean alpha of heads in that layer
    Returns all deltas for rollback.
    
    Note: This is the original version and may run into VRAM issues.
          Prefer add_alpha_inplace_with_vo_ffn_cpu_optimized.
    """
    deltas = []

    # 1) Use the original function for Q/K
    deltas += add_qk_alpha_inplace(model, projected_tv, axes, alpha_vec,
                                   selected_layers, selected_heads, couple_qk=couple_qk)

    # 2) If VO/FFN exist, derive coefficients and apply
    has_vo = "vo" in projected_tv and len(projected_tv["vo"]) > 0
    has_ffn = "ffn" in projected_tv and len(projected_tv["ffn"]) > 0
    if not (has_vo or has_ffn):
        return deltas

    head_alpha = _derive_head_alpha_from_qk(alpha_vec, axes, selected_layers, selected_heads, couple_qk)
    layer_alpha = _derive_layer_alpha_from_heads(head_alpha, selected_layers, selected_heads)

    # --- VO ---
    if has_vo:
        for l, per_h in projected_tv["vo"].items():
            layer = model.model.layers[l]; sa = layer.self_attn
            WV = sa.v_proj.weight; WO = sa.o_proj.weight
            d_model = model.config.hidden_size
            H_all = model.config.num_attention_heads
            hD = d_model // H_all
            H_kv = getattr(model.config, "num_key_value_heads", H_all)

            for h, tv in per_h.items():
                a = head_alpha.get((l,h), None)
                if a is None: 
                    continue
                # V
                dV = tv.get("dV_proj", None)
                if dV is not None and a != 0.0:
                    kvh = h % H_kv
                    v_start, v_end = kvh*hD, (kvh+1)*hD
                    ds = (dV.T.to(WV.device, WV.dtype)) * a
                    delta = torch.zeros_like(WV); delta[v_start:v_end,:] = ds
                    WV.data.add_(delta); deltas.append((WV, delta))
                # O
                dO = tv.get("dO_proj", None)
                if dO is not None and a != 0.0:
                    o_start, o_end = h*hD, (h+1)*hD
                    ds = (dO.to(WO.device, WO.dtype)) * a
                    delta = torch.zeros_like(WO); delta[:, o_start:o_end] = ds
                    WO.data.add_(delta); deltas.append((WO, delta))

    # --- FFN ---
    if has_ffn:
        for l, tv in projected_tv["ffn"].items():
            a = layer_alpha.get(l, None)
            if a is None or a == 0.0:
                continue
            layer = model.model.layers[l]
            for k_name, mod_name in [
                ("dGate_proj","gate_proj"),
                ("dUp_proj","up_proj"),
                ("dDown_proj","down_proj"),
            ]:
                dW = tv.get(k_name, None)
                if dW is None:
                    continue
                W = getattr(layer.mlp, mod_name).weight
                ds = (dW.to(W.device, W.dtype)) * a
                W.data.add_(ds); deltas.append((W, ds))

    return deltas


# ===========================
# Memory monitoring utilities
# ===========================
def log_memory_usage(tag: str, verbose: bool = True):
    """Record current RAM and VRAM usage."""
    if not verbose:
        return
        
    import psutil
    import gc
    
    # CPU memory
    process = psutil.Process()
    cpu_memory_mb = process.memory_info().rss / 1024 / 1024
    
    # GPU memory
    gpu_memory_str = ""
    if torch.cuda.is_available():
        gpu_allocated_gb = torch.cuda.memory_allocated() / 1024**3
        gpu_reserved_gb = torch.cuda.memory_reserved() / 1024**3
        gpu_memory_str = f" | GPU: {gpu_allocated_gb:.2f}GB allocated, {gpu_reserved_gb:.2f}GB reserved"
    
    # Python object count
    gc.collect()
    object_count = len(gc.get_objects())
    
    print(f"    💾 [{tag}] CPU: {cpu_memory_mb:.1f}MB{gpu_memory_str} | Objects: {object_count}")


def aggressive_memory_cleanup():
    """Aggressive memory cleanup."""
    import gc
    
    # Force multiple garbage collections
    for _ in range(3):
        gc.collect()
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ===========================
# QP solver (box + L2 prior + L1)
# ===========================
def _sym_psd(H: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    Hs = 0.5 * (H + H.T)
    min_diag = torch.diag(Hs).min().item()
    if min_diag < eps:
        Hs = Hs + (eps - min_diag + eps) * torch.eye(Hs.size(0), device=Hs.device, dtype=Hs.dtype)
    return Hs

def _diag_like(H: torch.Tensor, thr: float = 1e-3) -> bool:
    diag = torch.diag(torch.diag(H))
    off = H - diag
    num = torch.linalg.norm(off).item()
    den = torch.linalg.norm(H).item() + 1e-12
    return (num / den) < thr

def _soft(x: torch.Tensor, t: float) -> torch.Tensor:
    if t <= 0: return x
    return torch.sign(x) * torch.clamp(x.abs() - t, min=0.0)

def _proj_box(x: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    return torch.min(torch.max(x, lo), hi)

@torch.no_grad()
def solve_box_qp_with_prior(
    H: torch.Tensor, b: torch.Tensor,
    alpha_prior: torch.Tensor,
    l2_prior: float = 0.0,
    l1: float = 0.0,
    box_lo: float = 0.0, box_hi: float = 1.5,
    per_dim_lo: torch.Tensor = None, per_dim_hi: torch.Tensor = None,
    use_diagonal_shortcut: bool = True,
    max_iter: int = 800, tol: float = 1e-6,
    strong_convex_jitter: float = 1e-6,
    power_iter: int = 20,
    verbose: bool = True
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    min_{lo<=α<=hi} 0.5 α^T (H + λI) α - (b + λ α_prior)^T α + μ ||α||_1
    """
    device = H.device
    m = b.numel()
    H_eff = _sym_psd(H + l2_prior * torch.eye(m, device=device, dtype=H.dtype), eps=strong_convex_jitter)
    b_eff = b + l2_prior * alpha_prior

    lo = torch.full((m,), box_lo, device=device, dtype=b.dtype) if per_dim_lo is None else per_dim_lo.to(device, b.dtype)
    hi = torch.full((m,), box_hi, device=device, dtype=b.dtype) if per_dim_hi is None else per_dim_hi.to(device, b.dtype)

    # diagonal shortcut
    if use_diagonal_shortcut and _diag_like(H_eff):
        if verbose:
            print("[QP] diagonal shortcut")
        d = torch.diag(H_eff) + strong_convex_jitter
        a = _soft(b_eff, l1) / d
        a = _proj_box(a, lo, hi)
        return a, {"status": "diag", "iters": 1}

    # power iteration for Lipschitz
    def spec_norm(A, iters=10):
        v = torch.randn(m, device=device, dtype=A.dtype)
        v = v / (v.norm() + 1e-12)
        for _ in range(iters):
            v = A @ v
            v = v / (v.norm() + 1e-12)
        return torch.dot(v, A @ v).item()
    try:
        L = spec_norm(H_eff, iters=power_iter)
    except Exception:
        L = float(torch.diag(H_eff).abs().max().item()) + 1e-6
    step = 1.0 / (L + 1e-6)
    if verbose:
        print(f"[QP] L≈{L:.3e}, step≈{step:.3e}")

    a = torch.clamp(alpha_prior.clone(), min=lo.min().item(), max=hi.max().item())
    y = a.clone()
    t = 1.0

    def grad(x): return (H_eff @ x) - b_eff

    last_obj = None
    for k in range(1, max_iter + 1):
        g = grad(y)
        x_new = _soft(y - step * g, l1 * step)
        x_new = _proj_box(x_new, lo, hi)
        t_new = 0.5 * (1 + math.sqrt(1 + 4 * t * t))
        y = x_new + ((t - 1) / t_new) * (x_new - a)
        a, t = x_new, t_new

        obj = 0.5 * (a @ (H_eff @ a)) - (b_eff @ a) + l1 * a.abs().sum()
        if last_obj is not None and abs((obj - last_obj).item()) <= tol * (abs(last_obj.item()) + 1e-12):
            if verbose: print(f"[QP] converged at it={k}")
            break
        last_obj = obj
        if verbose and (k % 50 == 0):
            gn = grad(a).norm().item()
            print(f"[QP] it={k:4d} obj={obj.item():.6e} ||grad||={gn:.3e}")

    return a, {"status": "pgd-nesterov", "iters": k, "final_obj": float(obj.item())}


# ===========================
# Priors & apply/save helpers
# ===========================
def build_alpha_prior(axes: Dict[str, Any], prior_scalar: float = 1.0,
                      per_type_priors: Optional[Dict[str, float]] = None) -> torch.Tensor:
    m = axes["dimensions"]["m"]
    prior = torch.full((m,), float(prior_scalar), dtype=torch.float32)
    if per_type_priors:
        for i, (_, _, t) in enumerate(axes["flat_index"]):
            if t in per_type_priors:
                prior[i] = float(per_type_priors[t])
    return prior


def _derive_head_alpha_from_qk(alpha_vec: torch.Tensor,
                               axes: Dict[str, Any],
                               selected_layers: List[int],
                               selected_heads: List[int],
                               couple_qk: bool) -> Dict[Tuple[int,int], float]:
    """
    Return per-(layer, head) alpha (to reuse for V/O).
    - Coupled: take α at (l,h,'QK')
    - Decoupled: average α for (l,h,'Q') and (l,h,'K') (use whichever exists)
    """
    idx = { (l,h,t): i for i,(l,h,t) in enumerate(axes["flat_index"]) }
    head_alpha: Dict[Tuple[int,int], List[float]] = {}
    for l in selected_layers:
        for h in selected_heads:
            key = (l,h)
            if couple_qk:
                i = idx.get((l,h,"QK"))
                if i is not None:
                    head_alpha[key] = [float(alpha_vec[i].item())]
            else:
                v = []
                iq = idx.get((l,h,"Q"))
                ik = idx.get((l,h,"K"))
                if iq is not None: v.append(float(alpha_vec[iq].item()))
                if ik is not None: v.append(float(alpha_vec[ik].item()))
                if v: head_alpha[key] = v
    # Average
    return { k: (sum(v)/len(v)) for k,v in head_alpha.items() }


def _derive_layer_alpha_from_heads(head_alpha: Dict[Tuple[int,int], float],
                                   selected_layers: List[int],
                                   selected_heads: List[int]) -> Dict[int, float]:
    """
    Return per-layer alpha (for FFN reuse): the mean of all head alphas in the layer.
    """
    by_layer: Dict[int, List[float]] = { l: [] for l in selected_layers }
    for l in selected_layers:
        for h in selected_heads:
            if (l,h) in head_alpha:
                by_layer[l].append(head_alpha[(l,h)])
    layer_alpha = {}
    for l, vv in by_layer.items():
        if len(vv) > 0:
            layer_alpha[l] = float(sum(vv) / len(vv))
    return layer_alpha


def apply_alpha_to_projected_task_vectors(projected_task_vectors: Dict[str, Any],
                                          alpha_star: torch.Tensor,
                                          axes: Dict[str, Any],
                                          couple_qk: bool = True,
                                          verbose: bool = True) -> Dict[str, Any]:
    """
    Scale task-vectors by α*:
      - Q/K: unchanged logic
      - V/O: reuse same (layer, head) α (if decoupled, use average of Q/K)
      - FFN: reuse per-layer mean α across heads
    Only scale keys present in projected_task_vectors (skip others).
    """
    scaled = {"qk": {}}
    counts = {"Q": 0, "K": 0, "V": 0, "O": 0, "FFN": 0}
    types = set(axes["types"])

    # --- Q/K: same as before ---
    for i, (l, h, t) in enumerate(axes["flat_index"]):
        a = float(alpha_star[i].item())
        if l not in scaled["qk"]: scaled["qk"][l] = {}
        if h not in scaled["qk"][l]: scaled["qk"][l][h] = {}
        src = projected_task_vectors.get("qk", {}).get(l, {}).get(h, {})
        if "QK" in types and t == "QK":
            if "dQ_proj" in src:
                scaled["qk"][l][h]["dQ_proj"] = src["dQ_proj"] * a; counts["Q"] += 1
            if "dK_proj" in src:
                scaled["qk"][l][h]["dK_proj"] = src["dK_proj"] * a; counts["K"] += 1
        elif t in ("Q","K"):
            if t == "Q" and "dQ_proj" in src:
                scaled["qk"][l][h]["dQ_proj"] = src["dQ_proj"] * a; counts["Q"] += 1
            if t == "K" and "dK_proj" in src:
                scaled["qk"][l][h]["dK_proj"] = src["dK_proj"] * a; counts["K"] += 1

    # --- NEW: derive VO / FFN α ---
    # If task-vector actually contains VO or FFN, build corresponding scaling
    has_vo = "vo" in projected_task_vectors and len(projected_task_vectors["vo"]) > 0
    has_ffn = "ffn" in projected_task_vectors and len(projected_task_vectors["ffn"]) > 0

    if has_vo or has_ffn:
        # Need selected_layers/heads; read them from axes
        layers = axes["layers"]; heads = axes["heads"]
        head_alpha = _derive_head_alpha_from_qk(alpha_star, axes, layers, heads, couple_qk)
        layer_alpha = _derive_layer_alpha_from_heads(head_alpha, layers, heads)

    # --- V/O ---
    if has_vo:
        scaled["vo"] = {}
        for l, per_h in projected_task_vectors["vo"].items():
            if l not in scaled["vo"]: scaled["vo"][l] = {}
            for h, src in per_h.items():
                if h not in scaled["vo"][l]: scaled["vo"][l][h] = {}
                a = head_alpha.get((l,h), None)
                if a is None: 
                    continue
                if "dV_proj" in src:
                    scaled["vo"][l][h]["dV_proj"] = src["dV_proj"] * a; counts["V"] += 1
                if "dO_proj" in src:
                    scaled["vo"][l][h]["dO_proj"] = src["dO_proj"] * a; counts["O"] += 1

    # --- FFN ---
    if has_ffn:
        scaled["ffn"] = {}
        for l, tv in projected_task_vectors["ffn"].items():
            a = layer_alpha.get(l, None)
            if a is None:
                continue
            scaled["ffn"][l] = {}
            for k_name, dW in tv.items():
                scaled["ffn"][l][k_name] = dW * a
                counts["FFN"] += 1

    if verbose:
        msg = f"[Apply] scaled Q={counts['Q']}, K={counts['K']}"
        if has_vo:  msg += f", V={counts['V']}, O={counts['O']}"
        if has_ffn: msg += f", FFN={counts['FFN']}"
        print(msg)
    return scaled


def save_alpha_coefficients(alpha_star: torch.Tensor, axes: Dict[str, Any],
                            output_path: str, merge_types: str = "qk",
                            extra: Dict[str, Any] = None):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data = {
        "merge_types": merge_types,
        "axes_info": {
            "layers": axes["layers"],
            "heads": axes["heads"],
            "types": axes["types"],
            "dimensions": axes["dimensions"],
            "flat_index": axes["flat_index"],
        },
        "statistics": {
            "total": len(alpha_star),
            "min": float(alpha_star.min().item()),
            "max": float(alpha_star.max().item()),
            "mean": float(alpha_star.mean().item()),
            "std": float(alpha_star.std().item())
        },
        "alpha": [float(x) for x in alpha_star.tolist()],
    }
    if extra: data["extra"] = extra
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    torch.save({"alpha_star": alpha_star.cpu(), "axes": axes, "merge_types": merge_types},
               output_path.replace(".json", ".pt"))
    print(f"[Save] α -> {output_path}")


def visualize_alpha_coefficients(alpha_star: torch.Tensor, axes: Dict[str, Any], 
                                output_dir: str, couple_qk: bool = True, 
                                projected_task_vectors: Dict[str, Any] = None,
                                verbose: bool = True):
    """
    Create a unified heatmap containing all alpha coefficients for task-vector parameters.
    Q/K/V/O are combined in one subplot, FFN in a separate subplot.
    The figure size and fonts are auto-tuned to avoid overlaps.
    """
    if verbose:
        print("🎨 Creating unified alpha coefficient heatmap...")
    
    # Create output directory
    vis_dir = os.path.join(output_dir, "alpha_visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    layers = axes["layers"]
    heads = axes["heads"]
    n_layers = len(layers)
    n_heads = len(heads)
    
    # Auto-adjust sizes
    if n_layers > 20 or n_heads > 20:
        # Large heatmap (e.g., 28×28)
        base_font_size = max(8, min(12, 150 // max(n_layers, n_heads)))  # dynamic font size
        annot_font_size = max(6, base_font_size - 2)  # slightly smaller annotations
        cell_size = max(0.4, min(1.0, 25 / max(n_layers, n_heads)))  # cell size
        show_numbers = True  # always show numbers per request
    else:
        # Small/medium heatmap
        base_font_size = 12
        annot_font_size = 10
        cell_size = 0.8
        show_numbers = True
    
    # Matplotlib parameters
    plt.rcParams['figure.dpi'] = 200
    plt.rcParams['savefig.dpi'] = 200
    plt.rcParams['font.size'] = base_font_size
    
    if verbose:
        print(f"  📐 Heatmap dimensions: {n_layers}x{n_heads}")
        print(f"  🔤 Font sizes: base={base_font_size}, annot={annot_font_size}")
        print(f"  📊 Show numbers: {show_numbers}")
    
    # Check if FFN data exists
    has_ffn = projected_task_vectors is not None and "ffn" in projected_task_vectors and len(projected_task_vectors["ffn"]) > 0
    
    # Compute subplot sizes
    qk_width = max(10, n_heads * cell_size + 4)  # base width + margin
    qk_height = max(8, n_layers * cell_size + 3)  # base height + margin
    
    # Up to 2 subplots: Q/K/V/O + FFN
    if has_ffn:
        # Two subplots
        ffn_width = 4  # FFN plot is narrower
        total_width = qk_width + ffn_width + 3  # add spacing
        fig, axes_list = plt.subplots(1, 2, figsize=(total_width, qk_height))
    else:
        # Single subplot: Q/K/V/O
        fig, axes_list = plt.subplots(1, 1, figsize=(qk_width, qk_height))
        axes_list = [axes_list]
    
    subplot_idx = 0
    
    # ==================== 1) Q/K/V/O unified heatmap ====================
    ax = axes_list[subplot_idx]
    
    # Labels
    x_labels = [f'{h}' for h in heads]
    y_labels = [f'{l}' for l in layers]
    
    if couple_qk:
        # Coupled mode: one alpha per (layer, head), used for Q/K/V/O
        alpha_matrix = np.zeros((n_layers, n_heads))
        
        for i, (l, h, t) in enumerate(axes["flat_index"]):
            if t == "QK":
                li = layers.index(l)
                hi = heads.index(h)
                alpha_matrix[li, hi] = float(alpha_star[i].item())
        
    else:
        # Decoupled: display average of Q and K
        alpha_Q = np.zeros((n_layers, n_heads))
        alpha_K = np.zeros((n_layers, n_heads))
        
        for i, (l, h, t) in enumerate(axes["flat_index"]):
            li = layers.index(l)
            hi = heads.index(h)
            if t == "Q":
                alpha_Q[li, hi] = float(alpha_star[i].item())
            elif t == "K":
                alpha_K[li, hi] = float(alpha_star[i].item())
        
        # Average Q/K; V/O use the same values
        alpha_matrix = (alpha_Q + alpha_K) / 2
    
    # Number formatting
    if alpha_matrix.max() > 100:
        fmt = '.1f'
    elif alpha_matrix.max() > 10:
        fmt = '.2f'
    else:
        fmt = '.3f'
        
    heatmap_kws = {
        'annot': show_numbers, 
        'fmt': fmt, 
        'cmap': 'RdYlBu_r',
        'xticklabels': x_labels,
        'yticklabels': y_labels,
        'ax': ax, 
        'cbar_kws': {'label': 'Alpha Coefficient', 'shrink': 0.8},
        'square': False,
        'linewidths': 0.3,
        'cbar': True
    }
    
    if show_numbers:
        heatmap_kws['annot_kws'] = {'size': annot_font_size, 'weight': 'normal'}
        
    sns.heatmap(alpha_matrix, **heatmap_kws)
    ax.set_title('Q/K/V/O Alpha Coefficients', fontweight='bold', fontsize=base_font_size+2, pad=20)
    ax.set_xlabel('Attention Heads', fontweight='bold', fontsize=base_font_size+1)
    ax.set_ylabel('Layers', fontweight='bold', fontsize=base_font_size+1)
    
    # Ticks
    ax.tick_params(axis='x', rotation=0 if n_heads <= 10 else 45, labelsize=max(7, base_font_size-2))
    ax.tick_params(axis='y', rotation=0, labelsize=max(7, base_font_size-2))
        
    subplot_idx += 1
    
    # ==================== 2) FFN heatmap ====================
    if has_ffn:
        ax = axes_list[subplot_idx]
        
        # Derive FFN alphas (layer-wise average from Q/K/V/O)
        head_alpha = _derive_head_alpha_from_qk(alpha_star, axes, layers, heads, couple_qk)
        layer_alpha = _derive_layer_alpha_from_heads(head_alpha, layers, heads)
        
        # FFN heatmap data (one column: per-layer value)
        alpha_ffn_matrix = np.zeros((n_layers, 1))
        for li, l in enumerate(layers):
            alpha_ffn_matrix[li, 0] = layer_alpha.get(l, 0.0)
        
        # Number formatting
        if alpha_ffn_matrix.max() > 100:
            ffn_fmt = '.1f' 
        elif alpha_ffn_matrix.max() > 10:
            ffn_fmt = '.2f'
        else:
            ffn_fmt = '.3f'
        
        # FFN heatmap
        heatmap_kws = {
            'annot': True,  # always show numbers
            'fmt': ffn_fmt, 
            'cmap': 'Oranges',
            'xticklabels': ['FFN'],
            'yticklabels': y_labels,
            'ax': ax, 
            'cbar_kws': {'label': 'Alpha Coefficient', 'shrink': 0.8},
            'square': False,
            'linewidths': 0.5,
            'cbar': True
        }
        
        heatmap_kws['annot_kws'] = {'size': max(8, annot_font_size), 'weight': 'normal'}
            
        sns.heatmap(alpha_ffn_matrix, **heatmap_kws)
        ax.set_title('FFN Alpha Coefficients', fontweight='bold', fontsize=base_font_size+2, pad=20)
        ax.set_xlabel('Component', fontweight='bold', fontsize=base_font_size+1)
        ax.set_ylabel('Layers', fontweight='bold', fontsize=base_font_size+1)
        
        ax.tick_params(axis='x', rotation=0, labelsize=base_font_size)
        ax.tick_params(axis='y', rotation=0, labelsize=max(7, base_font_size-2))
    
    # Layout
    title_fontsize = max(14, base_font_size + 4)
    plt.suptitle('Task-Vector Alpha Coefficients', 
                fontsize=title_fontsize, fontweight='bold', y=0.98)
    
    if n_layers > 20 or n_heads > 20:
        plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    else:
        plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    
    # Save figures (PNG + PDF)
    plt.savefig(os.path.join(vis_dir, "alpha_unified_heatmap.png"), 
               bbox_inches='tight', facecolor='white', dpi=200)
    plt.savefig(os.path.join(vis_dir, "alpha_unified_heatmap.pdf"), 
                bbox_inches='tight', facecolor='white')
    plt.close()
    
    # ==================== Save summary ====================
    alpha_values = alpha_matrix.flatten()
    ffn_values = []
    if has_ffn:
        head_alpha = _derive_head_alpha_from_qk(alpha_star, axes, layers, heads, couple_qk)
        layer_alpha = _derive_layer_alpha_from_heads(head_alpha, layers, heads)
        ffn_values = [layer_alpha.get(l, 0.0) for l in layers]
    
    summary_text = f"""Task-Vector Alpha Coefficients Visualization Summary
====================================================

Model Configuration:
- Layers: {layers}
- Heads: {heads}
- Coupling Mode: {'Q/K Coupled' if couple_qk else 'Q/K Decoupled'}
- Components: {'Q/K/V/O + FFN' if has_ffn else 'Q/K/V/O only'}

QP Optimized Alpha (Q/K/V/O):
- Total Parameters: {len(alpha_star)}
- Min: {np.min(alpha_values):.6f}
- Max: {np.max(alpha_values):.6f}
- Mean: {np.mean(alpha_values):.6f}
- Std: {np.std(alpha_values):.6f}"""
    
    if has_ffn and ffn_values:
        summary_text += f"""

FFN Alpha (Layer-wise Average):
- Min: {np.min(ffn_values):.6f}
- Max: {np.max(ffn_values):.6f}
- Mean: {np.mean(ffn_values):.6f}
- Std: {np.std(ffn_values):.6f}"""
    
    summary_text += f"""

Visualization Layout:
- Left subplot: Q/K/V/O Alpha coefficients (all use the same values)
- Right subplot: FFN Alpha coefficients (layer-wise averages)

Notes:
- Q/K/V/O values are identical since V/O derive from Q/K
- Each cell shows the alpha coefficient for that layer-head combination
- FFN values are per-layer averages of the corresponding head alphas

Files saved in: {vis_dir}
"""
    
    with open(os.path.join(vis_dir, "unified_visualization_summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary_text)
    
    if verbose:
        print(f"  ✅ Unified visualization saved to: {vis_dir}")
        print(f"  📊 Generated 1 combined heatmap with 2 subplots")
        print(f"  📈 Q/K/V/O Alpha range: [{np.min(alpha_values):.4f}, {np.max(alpha_values):.4f}]")
        if has_ffn and ffn_values:
            print(f"  🔧 FFN Alpha range: [{np.min(ffn_values):.4f}, {np.max(ffn_values):.4f}]")


# ===========================
# Core: QP optimization variants
# ===========================
@torch.no_grad()
def optimize_alpha_true_forward_fast_align_leak(
    projected_pkl: str,
    base_model_path: str,
    json_data_file: str,
    selected_layers: List[int],
    selected_heads: List[int],
    couple_qk: bool = True,         # True: one α per (layer,head) applying to both Q&K
    prior_scalar: float = 1.0,
    per_type_priors: Optional[Dict[str, float]] = None,   # if couple, allow 'QK' key
    l2_prior: float = 1e-1,
    l1: float = 0.0,
    box_lo: float = 0.0, box_hi: float = 1.5,
    # NEW hyper-params for H and b construction
    H_lambda: float = 1.0,          # constant on H diagonal
    H_mu: float = 1.0,              # weight on post-leak u1
    rho_du: float = 0.5,            # b uses (Δa - rho_du * Δu)
    device: str = "cuda:0",
    verbose: bool = True
) -> Dict[str, Any]:

    # --- load projected task-vectors ---
    with open(projected_pkl, "rb") as f:
        proj = pickle.load(f)
    projected_tv = proj["projected_task_vectors"]
    cfg = proj["config"]
    if selected_layers == "all" or selected_layers is None:
        selected_layers = cfg["selected_layers"]
    if selected_heads == "all" or selected_heads is None:
        selected_heads = cfg["selected_heads"]

    # --- axes ---
    if couple_qk:
        types = ["QK"]
        flat_index = [(l,h,"QK") for l in selected_layers for h in selected_heads]
    else:
        types = ["Q","K"]
        flat_index = [(l,h,t) for l in selected_layers for h in selected_heads for t in types]
    axes = {
        "layers": selected_layers, "heads": selected_heads, "types": types,
        "flat_index": flat_index,
        "dimensions": {"m": len(flat_index), "n_layers": len(selected_layers),
                       "n_heads": len(selected_heads), "n_types": len(types)}
    }
    m = axes["dimensions"]["m"]

    # --- model & tokenizer ---
    if verbose:
        print("🔧 Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(base_model_path, use_fast=True, trust_remote_code=True)
    
    if verbose:
        print("🔧 Loading model...")
        if torch.cuda.is_available():
            print(f"  💾 Initial GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
    
    # Use device_map="auto" to shard across GPUs; attention computation device is set separately
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True
    ).eval()
    
    if verbose:
        print(f"✅ Model loaded with device_map=auto")
        if torch.cuda.is_available():
            print(f"  💾 GPU Memory after loading: {torch.cuda.memory_allocated()/1024**3:.2f}GB / {torch.cuda.max_memory_allocated()/1024**3:.2f}GB")

    # --- load JSONL samples ---
    if verbose:
        print("📂 Loading JSONL training data...")
    
    # Detect file format and load accordingly
    if json_data_file.endswith('.jsonl'):
        samples = load_jsonl(json_data_file)
    else:
        # Legacy JSON format
        with open(json_data_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "samples" in data:
            samples = data["samples"]
        else:
            samples = data if isinstance(data, list) else [data]
    
    S = len(samples)
    if verbose:
        print(f"  📊 Loaded {S} training samples from {'JSONL' if json_data_file.endswith('.jsonl') else 'JSON'} format")

    # --- alpha prior ---
    if verbose:
        print("🎯 Building alpha prior vector...")
    # per-type priors: if couple, allow 'QK' key; else 'Q','K'
    alpha_prior = build_alpha_prior(axes, prior_scalar=prior_scalar, per_type_priors=per_type_priors).to(next(model.parameters()).device)
    if verbose:
        print(f"  ✅ Alpha prior: shape={tuple(alpha_prior.shape)}, range=[{alpha_prior.min():.3f}, {alpha_prior.max():.3f}]")

    # --- PASS 1: anchor a0/u0 for all samples ---
    if verbose:
        print("\n⏩ Pass-1 (Anchor): computing (a0,u0) for all samples...")
        log_memory_usage("Before Pass-1", verbose)
    a0_list: List[torch.Tensor] = []
    u0_list: List[torch.Tensor] = []

    # head slots
    head_slots = [(l,h) for l in selected_layers for h in selected_heads]
    m_heads = len(head_slots)

    for si, sample in enumerate(samples):
        # Use new instruction-wise processing logic
        span_data = extract_instruction_wise_spans(sample, tok)
        instructions_data = span_data['instructions']
        global_unrelated_spans = span_data['global_unrelated_spans']
        global_unrelated_count = span_data['global_unrelated_count']

        # Build full text
        full_text = build_full_text_from_sample(sample, tok)

        if not instructions_data:
            a0 = torch.zeros(m_heads, dtype=torch.float32, device=alpha_prior.device)
            u0 = torch.zeros(m_heads, dtype=torch.float32, device=alpha_prior.device)
        else:
            # Debug: check token ranges
            if verbose and si == 0:  # Only for first sample
                tokenizer_output = tok(full_text, return_offsets_mapping=True, add_special_tokens=True)
                tokens = tokenizer_output['input_ids']
                seq_len = len(tokens)
                print(f"    🔍 Debug info for sample {si}: seq_len={seq_len}, instructions={len(instructions_data)}")
                for idx, instr in enumerate(instructions_data):
                    I_k = instr['instruction_spans']
                    R_k = instr['related_spans']
                    U_k = instr['unrelated_spans']
                    print(f"      Instr {idx+1} ({instr['instruction_id']}): I={len(I_k)}, R={len(R_k)}, U={len(U_k)}")
                print(f"    📊 Global unrelated: {len(global_unrelated_spans)} tokens, {global_unrelated_count} spans")
            
            # Instruction-wise computation
            a0, u0 = compute_align_leak_vectors_instruction_wise(
                model, tok, full_text, selected_layers, selected_heads, 
                instructions_data, global_unrelated_spans, global_unrelated_count,
                device=device, verbose=verbose
            )

        a0_list.append(a0); u0_list.append(u0)
        
        # 💾 Explicitly free per-sample temporaries
        del span_data, instructions_data, global_unrelated_spans, full_text
        if 'a0' in locals():
            del a0, u0
        
        if verbose and (si+1) % max(1,S//10 if S>=10 else 1) == 0:
            print(f"  • Anchored {si+1}/{S}")
            if torch.cuda.is_available():
                print(f"    💾 GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

        # 🔄 Force-clear GPU cache to avoid VRAM accumulation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 📊 Do an aggressive cleanup every 10 samples
        if (si + 1) % 10 == 0:
            aggressive_memory_cleanup()

    # Phase cleanup
    if verbose:
        log_memory_usage("After Pass-1", verbose)
    aggressive_memory_cleanup()

    # --- Apply whole task-vector once, with alpha_prior (coupled or decoupled) ---
    if verbose:
        print("\n🔧 Applying WHOLE task-vector once (according to alpha_prior)...")
    deltas_all = add_alpha_inplace_with_vo_ffn_cpu_optimized(
        model, projected_tv, axes, alpha_prior,
        selected_layers, selected_heads, couple_qk=couple_qk, verbose=verbose
    )

    # --- PASS 2: a1/u1 for all, accumulate Δa, Δu and post u1 ---
    if verbose:
        print("⏩ Pass-2 (Post): computing (a1,u1), accumulating Δa, Δu, and u1...")
        log_memory_usage("Before Pass-2", verbose)
    sum_dA = torch.zeros(m_heads, dtype=torch.float32, device=alpha_prior.device)
    sum_dU = torch.zeros(m_heads, dtype=torch.float32, device=alpha_prior.device)
    sum_u1 = torch.zeros(m_heads, dtype=torch.float32, device=alpha_prior.device)

    for si, sample in enumerate(samples):
        # Same instruction-wise processing
        span_data = extract_instruction_wise_spans(sample, tok)
        instructions_data = span_data['instructions']
        global_unrelated_spans = span_data['global_unrelated_spans']
        global_unrelated_count = span_data['global_unrelated_count']

        full_text = build_full_text_from_sample(sample, tok)

        if not instructions_data:
            a1 = torch.zeros(m_heads, dtype=torch.float32, device=alpha_prior.device)
            u1 = torch.zeros(m_heads, dtype=torch.float32, device=alpha_prior.device)
        else:
            a1, u1 = compute_align_leak_vectors_instruction_wise(
                model, tok, full_text, selected_layers, selected_heads, 
                instructions_data, global_unrelated_spans, global_unrelated_count,
                device=device, verbose=verbose
            )

        dA = a1 - a0_list[si]
        dU = u1 - u0_list[si]

        sum_dA += dA
        sum_dU += dU
        sum_u1 += u1

        # 💾 Explicit cleanup per sample
        del span_data, instructions_data, global_unrelated_spans, full_text
        if 'a1' in locals():
            del a1, u1, dA, dU

        if verbose and (si+1) % max(1,S//10 if S>=10 else 1) == 0:
            print(f"  • Posted {si+1}/{S} samples")
            if torch.cuda.is_available():
                print(f"    💾 GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

        # 🔄 Force-clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 📊 Aggressive cleanup every 10 samples
        if (si + 1) % 10 == 0:
            aggressive_memory_cleanup()

    # rollback
    rollback(deltas_all)
    if verbose:
        print("  🔄 Rolled back task-vector changes")
        log_memory_usage("After Pass-2", verbose)
    
    # Final memory cleanup
    aggressive_memory_cleanup()

    # --- Build diagonal H and b using alignment & leak ---
    if verbose:
        print("\n🔬 Building diagonal QP (alignment + leak) and solving...")
        print("\n🧪 [QP Diagnostics] Intermediate Statistics:")
        print(f"  📈 Samples: S = {S}")
        print(f"  📈 sum_dA (alignment changes):")
        print(f"      • Range: [{sum_dA.min().item():.6f}, {sum_dA.max().item():.6f}]")
        print(f"      • Mean: {sum_dA.mean().item():.6f}, Std: {sum_dA.std().item():.6f}")
        print(f"      • Non-zero count: {(sum_dA.abs() > 1e-8).sum().item()}/{len(sum_dA)}")
        print(f"  📈 sum_dU (leak changes):")
        print(f"      • Range: [{sum_dU.min().item():.6f}, {sum_dU.max().item():.6f}]")
        print(f"      • Mean: {sum_dU.mean().item():.6f}, Std: {sum_dU.std().item():.6f}")
        print(f"      • Non-zero count: {(sum_dU.abs() > 1e-8).sum().item()}/{len(sum_dU)}")
        print(f"  📈 sum_u1 (post-leak):")
        print(f"      • Range: [{sum_u1.min().item():.6f}, {sum_u1.max().item():.6f}]")
        print(f"      • Mean: {sum_u1.mean().item():.6f}, Std: {sum_u1.std().item():.6f}")

    # Head-level stats → α-dimension
    # Coupled: α-dim = m_heads.  Decoupled: α-dim = 2*m_heads, split evenly to Q and K.
    if couple_qk:
        b_head = (sum_dA - rho_du * sum_dU)            # larger is better
        H_head = (S * H_lambda + H_mu * sum_u1 + 1e-6) # leak increases curvature → shrink α
        H = torch.diag(H_head)
        b = b_head
        alpha_prior_match = alpha_prior
    else:
        # split evenly to Q/K
        b = torch.zeros(m, dtype=torch.float32, device=alpha_prior.device)
        H_diag = torch.zeros(m, dtype=torch.float32, device=alpha_prior.device)
        # precompute index lists for Q/K
        idx_Q, idx_K = [], []
        for i,(l,h,t) in enumerate(axes["flat_index"]):
            if t == "Q": idx_Q.append(i)
            else: idx_K.append(i)
        idx_Q = torch.tensor(idx_Q, device=b.device); idx_K = torch.tensor(idx_K, device=b.device)

        # expand head vectors to Q/K with 1/2 split
        b.scatter_(0, idx_Q, 0.5 * (sum_dA - rho_du * sum_dU))
        b.scatter_(0, idx_K, 0.5 * (sum_dA - rho_du * sum_dU))
        H_diag.scatter_(0, idx_Q, 0.5 * (S * H_lambda + H_mu * sum_u1) + 1e-6)
        H_diag.scatter_(0, idx_K, 0.5 * (S * H_lambda + H_mu * sum_u1) + 1e-6)

        H = torch.diag(H_diag)
        alpha_prior_match = alpha_prior  # same shape (2*m_heads) when decoupled

    if verbose:
        print(f"\n🧪 [QP Diagnostics] Final QP Problem:")
        H_diag_values = torch.diag(H)
        print(f"  📊 H diagonal values:")
        print(f"      • Range: [{H_diag_values.min().item():.6f}, {H_diag_values.max().item():.6f}]")
        print(f"      • Mean: {H_diag_values.mean().item():.6f}, Std: {H_diag_values.std().item():.6f}")
        print(f"  📊 b vector values:")
        print(f"      • Range: [{b.min().item():.6f}, {b.max().item():.6f}]")
        print(f"      • Mean: {b.mean().item():.6f}, Std: {b.std().item():.6f}")
        print(f"      • Norm: {b.norm().item():.6f}")
        print(f"      • Non-zero count: {(b.abs() > 1e-8).sum().item()}/{len(b)}")
        print(f"  📊 Alpha prior:")
        print(f"      • Range: [{alpha_prior_match.min().item():.6f}, {alpha_prior_match.max().item():.6f}]")
        print(f"      • Mean: {alpha_prior_match.mean().item():.6f}")
        print(f"  📊 Effective gradient ratio (b/H):")
        b_over_h = b / (H_diag_values + 1e-12)
        print(f"      • Range: [{b_over_h.min().item():.6f}, {b_over_h.max().item():.6f}]")
        print(f"      • Mean: {b_over_h.mean().item():.6f}")
        print(f"  🎛️  QP parameters: H_lambda={H_lambda}, H_mu={H_mu}, rho_du={rho_du}")
        print(f"  🎛️  l2_prior={l2_prior}, l1={l1}, box=[{box_lo}, {box_hi}]")
        
        # Estimate alpha without constraints
        unconstrained_alpha = b_over_h
        print(f"  🔮 Unconstrained alpha estimate (b/H):")
        print(f"      • Range: [{unconstrained_alpha.min().item():.6f}, {unconstrained_alpha.max().item():.6f}]")
        in_bounds = ((unconstrained_alpha >= box_lo) & (unconstrained_alpha <= box_hi)).sum().item()
        print(f"      • Would be in bounds [{box_lo}, {box_hi}]: {in_bounds}/{len(unconstrained_alpha)}")
        
        # Impact of L2 regularization
        H_eff_diag = H_diag_values + l2_prior
        b_eff = b + l2_prior * alpha_prior_match
        alpha_with_l2 = b_eff / H_eff_diag
        print(f"  🔮 With L2 regularization alpha estimate:")
        print(f"      • Range: [{alpha_with_l2.min().item():.6f}, {alpha_with_l2.max().item():.6f}]")
        in_bounds_l2 = ((alpha_with_l2 >= box_lo) & (alpha_with_l2 <= box_hi)).sum().item()
        print(f"      • Would be in bounds [{box_lo}, {box_hi}]: {in_bounds_l2}/{len(alpha_with_l2)}")

    alpha_star, info = solve_box_qp_with_prior(
        H, b,
        alpha_prior=alpha_prior_match,
        l2_prior=l2_prior, l1=l1,
        box_lo=box_lo, box_hi=box_hi,
        use_diagonal_shortcut=True,
        max_iter=200, tol=1e-7, verbose=verbose
    )

    if verbose:
        print(f"  ✅ QP solved: {info['status']}, iters={info['iters']}")
        print(f"  📊 Alpha*: range=[{alpha_star.min().item():.3f}, {alpha_star.max().item():.3f}], mean={alpha_star.mean().item():.3f}")

    return {
        "alpha_star": alpha_star.detach().cpu(),
        "axes": axes,
        "H": H.detach().cpu(), "b": b.detach().cpu(),
        "qp_info": info,
        "alpha_prior": alpha_prior_match.detach().cpu(),
        "projected_task_vectors": projected_tv,
        "samples": S,
        "couple_qk": couple_qk,
        "stats": {
            "sum_dA": sum_dA.detach().cpu(),
            "sum_dU": sum_dU.detach().cpu(),
            "sum_u1": sum_u1.detach().cpu()
        }
    }


@torch.no_grad()
def optimize_alpha_anchor_only(
    projected_pkl: str,
    base_model_path: str,
    json_data_file: str,
    selected_layers: List[int],
    selected_heads: List[int],
    couple_qk: bool = True,
    prior_scalar: float = 1.0,              # still allow user α_prior (as mean for L2 prior)
    per_type_priors: Optional[Dict[str, float]] = None,
    l2_prior: float = 1e-1,
    l1: float = 0.0,
    box_lo: float = 0.0, box_hi: float = 1.5,
    H_lambda: float = 1.0, H_mu: float = 1.0,
    rho_du: float = 0.5,
    kappa_a: float = 1.0, kappa_u: float = 1.0,   # NEW: scaling for anchor scores
    device: str = "cuda:0",
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Anchor-only QP optimization: compute (a0,u0) ONLY on the original model,
    encode "more alignment is better, more leak is worse" directly into b and H.
    """

    # --- load projected task-vectors & axes ---
    with open(projected_pkl, "rb") as f:
        proj = pickle.load(f)
    projected_tv = proj["projected_task_vectors"]
    cfg = proj["config"]
    if selected_layers == "all" or selected_layers is None:
        selected_layers = cfg["selected_layers"]
    if selected_heads == "all" or selected_heads is None:
        selected_heads = cfg["selected_heads"]

    if couple_qk:
        types = ["QK"]
        flat_index = [(l,h,"QK") for l in selected_layers for h in selected_heads]
    else:
        types = ["Q","K"]
        flat_index = [(l,h,t) for l in selected_layers for h in selected_heads for t in types]
    axes = {
        "layers": selected_layers, "heads": selected_heads, "types": types,
        "flat_index": flat_index,
        "dimensions": {"m": len(flat_index), "n_layers": len(selected_layers),
                       "n_heads": len(selected_heads), "n_types": len(types)}
    }
    m = axes["dimensions"]["m"]

    # --- model & tokenizer ---
    if verbose: 
        print("🔧 Loading tokenizer...")
        print("🔧 Loading model...")
    tok = AutoTokenizer.from_pretrained(base_model_path, use_fast=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.bfloat16,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True
    ).eval()
    
    if verbose:
        print("✅ Model loaded with device_map=auto")

    # --- load samples ---
    if verbose:
        print("📂 Loading training data...")
    if json_data_file.endswith(".jsonl"):
        samples = load_jsonl(json_data_file)
    else:
        with open(json_data_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        samples = data["samples"] if isinstance(data, dict) and "samples" in data else (data if isinstance(data, list) else [data])
    S = len(samples)
    
    if verbose:
        print(f"  📊 Loaded {S} training samples")

    # --- prior vector for L2 ---
    alpha_prior = build_alpha_prior(axes, prior_scalar=prior_scalar, per_type_priors=per_type_priors).to(next(model.parameters()).device)

    # --- Single pass: anchor a0/u0 ---
    if verbose: 
        print("\n⛳ Anchor-only: computing (a0,u0) ...")
    head_slots = [(l,h) for l in selected_layers for h in selected_heads]
    m_heads = len(head_slots)
    sum_a0 = torch.zeros(m_heads, dtype=torch.float32, device=alpha_prior.device)
    sum_u0 = torch.zeros(m_heads, dtype=torch.float32, device=alpha_prior.device)

    for si, sample in enumerate(samples):
        # Use instruction-wise processing
        span_data = extract_instruction_wise_spans(sample, tok)
        instructions_data = span_data['instructions']
        global_unrelated_spans = span_data['global_unrelated_spans']
        global_unrelated_count = span_data['global_unrelated_count']

        # Full text
        full_text = build_full_text_from_sample(sample, tok)

        if not instructions_data:
            a0 = torch.zeros(m_heads, dtype=torch.float32, device=alpha_prior.device)
            u0 = torch.zeros(m_heads, dtype=torch.float32, device=alpha_prior.device)
        else:
            a0, u0 = compute_align_leak_vectors_instruction_wise(
                model, tok, full_text, selected_layers, selected_heads, 
                instructions_data, global_unrelated_spans, global_unrelated_count,
                device=device, verbose=verbose
            )

        sum_a0 += a0
        sum_u0 += u0
        
        # 💾 Explicit cleanup
        del span_data, instructions_data, global_unrelated_spans, full_text
        if 'a0' in locals():
            del a0, u0
        
        if verbose and (si+1) % max(1, S//10 if S>=10 else 1) == 0:
            print(f"  • Anchored {si+1}/{S}")
            if torch.cuda.is_available():
                print(f"    💾 GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

        if torch.cuda.is_available(): 
            torch.cuda.empty_cache()
        
        if (si + 1) % 10 == 0:
            aggressive_memory_cleanup()

    # --- Build QP ---
    if verbose:
        print("\n🔬 Building anchor-only QP and solving...")
        print("\n🧪 [QP Diagnostics] Intermediate Statistics:")
        print(f"  📈 Samples: S = {S}")
        print(f"  📈 sum_a0 (anchor alignment):")
        print(f"      • Range: [{sum_a0.min().item():.6f}, {sum_a0.max().item():.6f}]")
        print(f"      • Mean: {sum_a0.mean().item():.6f}, Std: {sum_a0.std().item():.6f}")
        print(f"      • Non-zero count: {(sum_a0.abs() > 1e-8).sum().item()}/{len(sum_a0)}")
        print(f"  📈 sum_u0 (anchor leak):")
        print(f"      • Range: [{sum_u0.min().item():.6f}, {sum_u0.max().item():.6f}]")
        print(f"      • Mean: {sum_u0.mean().item():.6f}, Std: {sum_u0.std().item():.6f}")
        print(f"      • Non-zero count: {(sum_u0.abs() > 1e-8).sum().item()}/{len(sum_u0)}")
        
    if couple_qk:
        b_head = kappa_a * sum_a0 - rho_du * kappa_u * sum_u0
        H_head = (S * H_lambda + H_mu * sum_u0 + 1e-6)
        H = torch.diag(H_head)
        b = b_head
        alpha_prior_match = alpha_prior
    else:
        b = torch.zeros(m, dtype=torch.float32, device=alpha_prior.device)
        H_diag = torch.zeros(m, dtype=torch.float32, device=alpha_prior.device)
        idx_Q = torch.tensor([i for i,(_,_,t) in enumerate(axes["flat_index"]) if t=="Q"], device=b.device)
        idx_K = torch.tensor([i for i,(_,_,t) in enumerate(axes["flat_index"]) if t=="K"], device=b.device)
        reward = kappa_a * sum_a0 - rho_du * kappa_u * sum_u0
        curvature = (S * H_lambda + H_mu * sum_u0 + 1e-6)
        b.scatter_(0, idx_Q, 0.5 * reward)
        b.scatter_(0, idx_K, 0.5 * reward)
        H_diag.scatter_(0, idx_Q, 0.5 * curvature)
        H_diag.scatter_(0, idx_K, 0.5 * curvature)
        H = torch.diag(H_diag)
        alpha_prior_match = alpha_prior

    if verbose:
        print(f"\n🧪 [QP Diagnostics] Final QP Problem:")
        H_diag_values = torch.diag(H)
        print(f"  📊 H diagonal values:")
        print(f"      • Range: [{H_diag_values.min().item():.6f}, {H_diag_values.max().item():.6f}]")
        print(f"      • Mean: {H_diag_values.mean().item():.6f}, Std: {H_diag_values.std().item():.6f}")
        print(f"  📊 b vector values:")
        print(f"      • Range: [{b.min().item():.6f}, {b.max().item():.6f}]")
        print(f"      • Mean: {b.mean().item():.6f}, Std: {b.std().item():.6f}")
        print(f"      • Norm: {b.norm().item():.6f}")
        print(f"      • Non-zero count: {(b.abs() > 1e-8).sum().item()}/{len(b)}")
        print(f"  📊 Alpha prior:")
        print(f"      • Range: [{alpha_prior_match.min().item():.6f}, {alpha_prior_match.max().item():.6f}]")
        print(f"      • Mean: {alpha_prior_match.mean().item():.6f}")
        print(f"  📊 Effective gradient ratio (b/H):")
        b_over_h = b / (H_diag_values + 1e-12)
        print(f"      • Range: [{b_over_h.min().item():.6f}, {b_over_h.max().item():.6f}]")
        print(f"      • Mean: {b_over_h.mean().item():.6f}")
        print(f"  🎛️  QP parameters: H_lambda={H_lambda}, H_mu={H_mu}, rho_du={rho_du}")
        print(f"  🎛️  kappa_a={kappa_a}, kappa_u={kappa_u}")
        
        unconstrained_alpha = b_over_h
        print(f"  🔮 Unconstrained alpha estimate (b/H):")
        print(f"      • Range: [{unconstrained_alpha.min().item():.6f}, {unconstrained_alpha.max().item():.6f}]")
        in_bounds = ((unconstrained_alpha >= box_lo) & (unconstrained_alpha <= box_hi)).sum().item()
        print(f"      • Would be in bounds [{box_lo}, {box_hi}]: {in_bounds}/{len(unconstrained_alpha)}")
        
        H_eff_diag = H_diag_values + l2_prior
        b_eff = b + l2_prior * alpha_prior_match
        alpha_with_l2 = b_eff / H_eff_diag
        print(f"  🔮 With L2 regularization alpha estimate:")
        print(f"      • Range: [{alpha_with_l2.min().item():.6f}, {alpha_with_l2.max().item():.6f}]")
        in_bounds_l2 = ((alpha_with_l2 >= box_lo) & (alpha_with_l2 <= box_hi)).sum().item()
        print(f"      • Would be in bounds [{box_lo}, {box_hi}]: {in_bounds_l2}/{len(alpha_with_l2)}")

    alpha_star, info = solve_box_qp_with_prior(
        H, b, alpha_prior=alpha_prior_match,
        l2_prior=l2_prior, l1=l1,
        box_lo=box_lo, box_hi=box_hi,
        use_diagonal_shortcut=True,
        max_iter=200, tol=1e-7, verbose=verbose
    )

    if verbose:
        print(f"  ✅ QP solved: {info['status']}, iters={info['iters']}")
        print(f"  📊 Alpha*: range=[{alpha_star.min().item():.3f}, {alpha_star.max().item():.3f}], mean={alpha_star.mean().item():.3f}")

    return {
        "alpha_star": alpha_star.detach().cpu(),
        "axes": axes,
        "H": H.detach().cpu(), "b": b.detach().cpu(),
        "qp_info": info,
        "alpha_prior": alpha_prior_match.detach().cpu(),
        "projected_task_vectors": projected_tv,
        "samples": S,
        "couple_qk": couple_qk,
        "stats": { "sum_a0": sum_a0.detach().cpu(), "sum_u0": sum_u0.detach().cpu() }
    }


@torch.no_grad()
def optimize_alpha_post_only(
    projected_pkl: str,
    base_model_path: str,
    json_data_file: str,
    selected_layers: List[int],
    selected_heads: List[int],
    couple_qk: bool = True,
    l2_prior: float = 1e-1,
    l1: float = 0.0,
    box_lo: float = 0.0, box_hi: float = 1.5,
    H_lambda: float = 1.0, H_mu: float = 1.0,
    rho_du: float = 0.5,
    kappa_a: float = 1.0, kappa_u: float = 1.0,   # NEW: scaling for post scores
    device: str = "cuda:0",
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Post-only QP optimization: apply the task-vector once with unit prior α=1,
    then compute (a1,u1) ONLY on the post state to build the QP.
    """

    # --- load projected task-vectors & axes ---
    with open(projected_pkl, "rb") as f:
        proj = pickle.load(f)
    projected_tv = proj["projected_task_vectors"]
    cfg = proj["config"]
    if selected_layers == "all" or selected_layers is None:
        selected_layers = cfg["selected_layers"]
    if selected_heads == "all" or selected_heads is None:
        selected_heads = cfg["selected_heads"]

    if couple_qk:
        types = ["QK"]
        flat_index = [(l,h,"QK") for l in selected_layers for h in selected_heads]
    else:
        types = ["Q","K"]
        flat_index = [(l,h,t) for l in selected_layers for h in selected_heads for t in types]
    axes = {
        "layers": selected_layers, "heads": selected_heads, "types": types,
        "flat_index": flat_index,
        "dimensions": {"m": len(flat_index), "n_layers": len(selected_layers),
                       "n_heads": len(selected_heads), "n_types": len(types)}
    }
    m = axes["dimensions"]["m"]

    # --- model & tokenizer ---
    if verbose: 
        print("🔧 Loading tokenizer...")
        print("🔧 Loading model...")
    tok = AutoTokenizer.from_pretrained(base_model_path, use_fast=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.bfloat16,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True
    ).eval()
    
    if verbose:
        print("✅ Model loaded with device_map=auto")

    # --- load samples ---
    if verbose:
        print("📂 Loading training data...")
    if json_data_file.endswith(".jsonl"):
        samples = load_jsonl(json_data_file)
    else:
        with open(json_data_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        samples = data["samples"] if isinstance(data, dict) and "samples" in data else (data if isinstance(data, list) else [data])
    S = len(samples)
    
    if verbose:
        print(f"  📊 Loaded {S} training samples")

    # --- set α_probe = 1 (same shape whether coupled/decoupled) ---
    alpha_probe = torch.ones(m, dtype=torch.float32, device=next(model.parameters()).device)

    # --- Apply task-vector once with α_probe ---
    if verbose:
        print("\n🔧 Applying task-vector ONCE with α_probe=1...")
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"  💾 [Before apply] GPU: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    
    torch.cuda.empty_cache()
    
    deltas_all = add_alpha_inplace_with_vo_ffn_cpu_optimized(
        model, projected_tv, axes, alpha_probe,
        selected_layers, selected_heads, couple_qk=couple_qk, verbose=verbose
    )
    
    if verbose and torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"  💾 [After apply] GPU: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

    # --- Single pass: post a1/u1 ---
    if verbose: 
        print("⏭️ Post-only: computing (a1,u1) ...")
    head_slots = [(l,h) for l in selected_layers for h in selected_heads]
    m_heads = len(head_slots)
    sum_a1 = torch.zeros(m_heads, dtype=torch.float32, device=alpha_probe.device)
    sum_u1 = torch.zeros(m_heads, dtype=torch.float32, device=alpha_probe.device)

    for si, sample in enumerate(samples):
        # Instruction-wise logic
        span_data = extract_instruction_wise_spans(sample, tok)
        instructions_data = span_data['instructions']
        global_unrelated_spans = span_data['global_unrelated_spans']
        global_unrelated_count = span_data['global_unrelated_count']

        full_text = build_full_text_from_sample(sample, tok)

        if not instructions_data:
            a1 = torch.zeros(m_heads, dtype=torch.float32, device=alpha_probe.device)
            u1 = torch.zeros(m_heads, dtype=torch.float32, device=alpha_probe.device)
        else:
            a1, u1 = compute_align_leak_vectors_instruction_wise(
                model, tok, full_text, selected_layers, selected_heads, 
                instructions_data, global_unrelated_spans, global_unrelated_count,
                device=device, verbose=verbose
            )

        sum_a1 += a1
        sum_u1 += u1
        
        # 💾 Explicit cleanup
        del span_data, instructions_data, global_unrelated_spans, full_text
        if 'a1' in locals():
            del a1, u1
        
        if verbose and (si+1) % max(1, S//10 if S>=10 else 1) == 0:
            print(f"  • Posted {si+1}/{S}")
            if torch.cuda.is_available():
                print(f"    💾 GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
                
        if torch.cuda.is_available(): 
            torch.cuda.empty_cache()
        
        if (si + 1) % 10 == 0:
            aggressive_memory_cleanup()

    # --- rollback probe ---
    if verbose:
        print("  🔄 Rolling back probe task-vector changes")
    rollback(deltas_all)

    # --- Build QP ---
    if verbose:
        print("\n🔬 Building post-only QP and solving...")
        print("\n🧪 [QP Diagnostics] Intermediate Statistics:")
        print(f"  📈 Samples: S = {S}")
        print(f"  📈 sum_a1 (post alignment):")
        print(f"      • Range: [{sum_a1.min().item():.6f}, {sum_a1.max().item():.6f}]")
        print(f"      • Mean: {sum_a1.mean().item():.6f}, Std: {sum_a1.std().item():.6f}")
        print(f"      • Non-zero count: {(sum_a1.abs() > 1e-8).sum().item()}/{len(sum_a1)}")
        print(f"  📈 sum_u1 (post leak):")
        print(f"      • Range: [{sum_u1.min().item():.6f}, {sum_u1.max().item():.6f}]")
        print(f"      • Mean: {sum_u1.mean().item():.6f}, Std: {sum_u1.std().item():.6f}")
        print(f"      • Non-zero count: {(sum_u1.abs() > 1e-8).sum().item()}/{len(sum_u1)}")
        
    if couple_qk:
        b_head = kappa_a * sum_a1 - rho_du * kappa_u * sum_u1
        H_head = (S * H_lambda + H_mu * sum_u1 + 1e-6)
        H = torch.diag(H_head)
        b = b_head
        # L2 prior can still use α_prior=1 or 0; we follow two-pass: use α_prior=1 as the "default magnitude" center
        alpha_prior_match = torch.ones_like(alpha_probe)
    else:
        b = torch.zeros(m, dtype=torch.float32, device=alpha_probe.device)
        H_diag = torch.zeros(m, dtype=torch.float32, device=alpha_probe.device)
        idx_Q = torch.tensor([i for i,(_,_,t) in enumerate(axes["flat_index"]) if t=="Q"], device=b.device)
        idx_K = torch.tensor([i for i,(_,_,t) in enumerate(axes["flat_index"]) if t=="K"], device=b.device)
        reward = kappa_a * sum_a1 - rho_du * kappa_u * sum_u1
        curvature = (S * H_lambda + H_mu * sum_u1 + 1e-6)
        b.scatter_(0, idx_Q, 0.5 * reward)
        b.scatter_(0, idx_K, 0.5 * reward)
        H_diag.scatter_(0, idx_Q, 0.5 * curvature)
        H_diag.scatter_(0, idx_K, 0.5 * curvature)
        H = torch.diag(H_diag)
        alpha_prior_match = torch.ones_like(alpha_probe)

    if verbose:
        print(f"\n🧪 [QP Diagnostics] Final QP Problem:")
        H_diag_values = torch.diag(H)
        print(f"  📊 H diagonal values:")
        print(f"      • Range: [{H_diag_values.min().item():.6f}, {H_diag_values.max().item():.6f}]")
        print(f"      • Mean: {H_diag_values.mean().item():.6f}, Std: {H_diag_values.std().item():.6f}")
        print(f"  📊 b vector values:")
        print(f"      • Range: [{b.min().item():.6f}, {b.max().item():.6f}]")
        print(f"      • Mean: {b.mean().item():.6f}, Std: {b.std().item():.6f}")
        print(f"      • Norm: {b.norm().item():.6f}")
        print(f"      • Non-zero count: {(b.abs() > 1e-8).sum().item()}/{len(b)}")
        print(f"  📊 Alpha prior:")
        print(f"      • Range: [{alpha_prior_match.min().item():.6f}, {alpha_prior_match.max().item():.6f}]")
        print(f"      • Mean: {alpha_prior_match.mean().item():.6f}")
        print(f"  📊 Effective gradient ratio (b/H):")
        b_over_h = b / (H_diag_values + 1e-12)
        print(f"      • Range: [{b_over_h.min().item():.6f}, {b_over_h.max().item():.6f}]")
        print(f"      • Mean: {b_over_h.mean().item():.6f}")
        print(f"  🎛️  QP parameters: H_lambda={H_lambda}, H_mu={H_mu}, rho_du={rho_du}")
        print(f"  🎛️  kappa_a={kappa_a}, kappa_u={kappa_u}")
        
        unconstrained_alpha = b_over_h
        print(f"  🔮 Unconstrained alpha estimate (b/H):")
        print(f"      • Range: [{unconstrained_alpha.min().item():.6f}, {unconstrained_alpha.max().item():.6f}]")
        in_bounds = ((unconstrained_alpha >= box_lo) & (unconstrained_alpha <= box_hi)).sum().item()
        print(f"      • Would be in bounds [{box_lo}, {box_hi}]: {in_bounds}/{len(unconstrained_alpha)}")
        
        H_eff_diag = H_diag_values + l2_prior
        b_eff = b + l2_prior * alpha_prior_match
        alpha_with_l2 = b_eff / H_eff_diag
        print(f"  🔮 With L2 regularization alpha estimate:")
        print(f"      • Range: [{alpha_with_l2.min().item():.6f}, {alpha_with_l2.max().item():.6f}]")
        in_bounds_l2 = ((alpha_with_l2 >= box_lo) & (alpha_with_l2 <= box_hi)).sum().item()
        print(f"      • Would be in bounds [{box_lo}, {box_hi}]: {in_bounds_l2}/{len(alpha_with_l2)}")

    alpha_star, info = solve_box_qp_with_prior(
        H, b, alpha_prior=alpha_prior_match,
        l2_prior=l2_prior, l1=l1,
        box_lo=box_lo, box_hi=box_hi,
        use_diagonal_shortcut=True,
        max_iter=200, tol=1e-7, verbose=verbose
    )

    if verbose:
        print(f"  ✅ QP solved: {info['status']}, iters={info['iters']}")
        print(f"  📊 Alpha*: range=[{alpha_star.min().item():.3f}, {alpha_star.max().item():.3f}], mean={alpha_star.mean().item():.3f}")

    return {
        "alpha_star": alpha_star.detach().cpu(),
        "axes": axes,
        "H": H.detach().cpu(), "b": b.detach().cpu(),
        "qp_info": info,
        "alpha_prior": alpha_prior_match.detach().cpu(),
        "projected_task_vectors": projected_tv,
        "samples": S,
        "couple_qk": couple_qk,
        "stats": { "sum_a1": sum_a1.detach().cpu(), "sum_u1": sum_u1.detach().cpu() }
    }


# ===========================
# CLI
# ===========================
def main():
    ap = argparse.ArgumentParser(description="Fast True-Forward QP (Q/K, Alignment+Leak, 2-pass)")
    ap.add_argument("--projected_file", type=str, required=True, help="*.pkl from null-space projection stage")
    ap.add_argument("--base_model", type=str, required=True, help="HF id or local path of base model")
    ap.add_argument("--json_data", type=str, required=True, help="training JSON with instruction/related spans")
    ap.add_argument("--layers", type=str, default="all", help="'all' or comma-separated indices")
    ap.add_argument("--heads", type=str, default="all", help="'all' or comma-separated indices")
    ap.add_argument("--prior_scalar", type=float, default=1.0)
    ap.add_argument("--l2_prior", type=float, default=1e-1)
    ap.add_argument("--l1", type=float, default=0.0)
    ap.add_argument("--box_lo", type=float, default=0.0)
    ap.add_argument("--box_hi", type=float, default=1.5)
    ap.add_argument("--device", type=str, default="cuda:0", help="device for attention computation")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--decouple_qk", action="store_true", help="if set, learn separate α for Q and K (still 2 passes)")
    ap.add_argument("--save_model", action="store_true", help="apply α* to base model Q/K and save")
    # NEW hyper-params
    ap.add_argument("--H_lambda", type=float, default=1.0, help="constant term for H diagonal (scaled internally by S)")
    ap.add_argument("--H_mu", type=float, default=1.0, help="weight on post-leak u1 in H")
    ap.add_argument("--rho_du", type=float, default=0.5, help="penalty weight on leak change in b")
    # NEW QP variant parameters
    ap.add_argument("--qp_variant", choices=["two_pass","anchor_only","post_only"],
                    default="two_pass", help="Choose QP construction method")
    ap.add_argument("--kappa_a", type=float, default=1.0, help="scaling for alignment score (single-state variants)")
    ap.add_argument("--kappa_u", type=float, default=1.0, help="scaling for leak score (single-state variants)")
    ap.add_argument("--verbose", action="store_true", help="verbose mode, show hook-method debug info")
    args = ap.parse_args()

    # read projected config to resolve "all"
    print(f"Loading projected task vectors...")
    with open(args.projected_file, "rb") as f:
        proj = pickle.load(f)
    cfg = proj["config"]
    use_layers = cfg["selected_layers"] if args.layers == "all" else [int(x) for x in args.layers.split(",")]
    use_heads = cfg["selected_heads"] if args.heads == "all" else [int(x) for x in args.heads.split(",")]

    couple_qk = not args.decouple_qk

    os.makedirs(args.out, exist_ok=True)
    variant = args.qp_variant
    print(f"🚀 Fast True-Forward QP ({variant}) Starting")
    print("=" * 60)
    print(f"📁 Input: {args.projected_file}")
    print(f"🤖 Model: {args.base_model} (device_map=auto)")
    print(f"📊 Data: {args.json_data}")
    print(f"🎯 Layers: {use_layers}")
    print(f"🎯 Heads: {use_heads}")
    print(f"🔧 Attention Device: {args.device}")
    print(f"🔗 QK coupling: {couple_qk}")
    print(f"🧭 QP Variant: {variant}")
    if variant in ["anchor_only", "post_only"]:
        print(f"⚙️  H_lambda={args.H_lambda}, H_mu={args.H_mu}, rho_du={args.rho_du}")
        print(f"⚙️  kappa_a={args.kappa_a}, kappa_u={args.kappa_u}")
    else:
        print(f"⚙️  H_lambda={args.H_lambda}, H_mu={args.H_mu}, rho_du={args.rho_du}")
    print("=" * 60)

    if variant == "two_pass":
        res = optimize_alpha_true_forward_fast_align_leak(
            projected_pkl=args.projected_file,
            base_model_path=args.base_model,
            json_data_file=args.json_data,
            selected_layers=use_layers,
            selected_heads=use_heads,
            couple_qk=couple_qk,
            prior_scalar=args.prior_scalar,
            per_type_priors={"QK": args.prior_scalar} if couple_qk else None,
            l2_prior=args.l2_prior,
            l1=args.l1,
            box_lo=args.box_lo, box_hi=args.box_hi,
            H_lambda=args.H_lambda, H_mu=args.H_mu, rho_du=args.rho_du,
            device=args.device,
            verbose=args.verbose
        )
        tag = "align_leak"

    elif variant == "anchor_only":
        res = optimize_alpha_anchor_only(
            projected_pkl=args.projected_file,
            base_model_path=args.base_model,
            json_data_file=args.json_data,
            selected_layers=use_layers,
            selected_heads=use_heads,
            couple_qk=couple_qk,
            prior_scalar=args.prior_scalar,
            per_type_priors={"QK": args.prior_scalar} if couple_qk else None,
            l2_prior=args.l2_prior,
            l1=args.l1,
            box_lo=args.box_lo, box_hi=args.box_hi,
            H_lambda=args.H_lambda, H_mu=args.H_mu, rho_du=args.rho_du,
            kappa_a=args.kappa_a, kappa_u=args.kappa_u,
            device=args.device,
            verbose=args.verbose
        )
        tag = "anchor_only"

    else:  # variant == "post_only"
        res = optimize_alpha_post_only(
            projected_pkl=args.projected_file,
            base_model_path=args.base_model,
            json_data_file=args.json_data,
            selected_layers=use_layers,
            selected_heads=use_heads,
            couple_qk=couple_qk,
            l2_prior=args.l2_prior,
            l1=args.l1,
            box_lo=args.box_lo, box_hi=args.box_hi,
            H_lambda=args.H_lambda, H_mu=args.H_mu, rho_du=args.rho_du,
            kappa_a=args.kappa_a, kappa_u=args.kappa_u,
            device=args.device,
            verbose=args.verbose
        )
        tag = "post_only"

    # save α & matrices
    print("\n💾 Saving optimization results...")
    out_dir = args.out
    
    print("  📊 Saving QP result tensor...")
    torch.save(res, os.path.join(out_dir, f"true_forward_qp_result_{tag}.pt"))

    merge_types_tag = "qk_coupled" if couple_qk else "qk_decoupled"
    print("  📄 Saving alpha coefficients...")
    extra_data = {"qp_info": res["qp_info"], "samples": res["samples"], "couple_qk": couple_qk,
                  "H_lambda": args.H_lambda, "H_mu": args.H_mu, "rho_du": args.rho_du,
                  "variant": variant}
    if variant in ["anchor_only", "post_only"]:
        extra_data.update({"kappa_a": args.kappa_a, "kappa_u": args.kappa_u})
    
    save_alpha_coefficients(
        alpha_star=res["alpha_star"], axes=res["axes"],
        output_path=os.path.join(out_dir, f"alpha_true_forward_{tag}.json"),
        merge_types=merge_types_tag,
        extra=extra_data
    )

    # # Optionally scale and save task-vectors (commented out)
    # print("\n🔧 Applying alpha* to projected task-vectors...")
    # scaled_tv = apply_alpha_to_projected_task_vectors(
    #     proj["projected_task_vectors"], res["alpha_star"], res["axes"], couple_qk=couple_qk, verbose=args.verbose
    # )
    # print("  💾 Saving scaled task-vectors...")
    # with open(os.path.join(out_dir, "scaled_task_vectors.pkl"), "wb") as f:
    #     pickle.dump(scaled_tv, f)
    # print(f"  ✅ Scaled task-vectors -> {os.path.join(out_dir, 'scaled_task_vectors.pkl')}")

    # Visualize alpha coefficients
    print("\n🎨 Creating alpha coefficient visualizations...")
    try:
        visualize_alpha_coefficients(
            res["alpha_star"], res["axes"], out_dir, couple_qk=couple_qk, 
            projected_task_vectors=proj["projected_task_vectors"], verbose=args.verbose
        )
    except Exception as e:
        print(f"  ⚠️  Visualization failed: {e}")
        print("  💡 Make sure matplotlib and seaborn are installed: pip install matplotlib seaborn")

    if args.save_model:
        print("\n🤖 Applying alpha* to base model and saving...")
        print("  🔧 Reloading model for final application...")
        # reload model with auto device mapping
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model, torch_dtype=torch.bfloat16,
            device_map="auto" if torch.cuda.is_available() else "cpu",
            trust_remote_code=True
        ).eval()
        
        print("  🔧 Applying alpha* to Q/K/V/O/FFN weights...")
        deltas = add_alpha_inplace_with_vo_ffn_cpu_optimized(
            model, proj["projected_task_vectors"], res["axes"], res["alpha_star"].to(next(model.parameters()).device),
            selected_layers=use_layers, selected_heads=use_heads, couple_qk=couple_qk, verbose=args.verbose
        )
        
        print("  💾 Saving optimized model...")
        # keep the deltas (do not rollback), save model
        out_model_dir = os.path.join(out_dir, f"qp_optimized_model_{tag}")
        os.makedirs(out_model_dir, exist_ok=True)
        model.save_pretrained(out_model_dir)
        
        print("  💾 Saving tokenizer...")
        tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True, trust_remote_code=True)
        tok.save_pretrained(out_model_dir)
        print(f"  ✅ Model saved: {out_model_dir}")

    print("\n" + "=" * 60)
    print(f"✅ QP Optimization Complete ({variant})!")
    print("=" * 60)
    print(f"📊 Alpha* Statistics:")
    print(f"  • Range: [{res['alpha_star'].min().item():.3f}, {res['alpha_star'].max().item():.3f}]")
    print(f"  • Mean: {res['alpha_star'].mean().item():.3f}")
    print(f"  • Std: {res['alpha_star'].std().item():.3f}")
    print(f"🧮 Matrix Shapes:")
    print(f"  • H: {tuple(res['H'].shape)}")
    print(f"  • b: {tuple(res['b'].shape)}")
    print(f"📁 Output Directory: {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
