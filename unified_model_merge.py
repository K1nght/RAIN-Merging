#!/usr/bin/env python3
"""
Unified Model Merge Script - Stage 3
Function: Supports two task-vector merge modes — Alpha weighting and Scaling Factor — usable alone or in combination.
Input: Projected task vectors file + alpha coefficients file (optional) + scaling factor (optional)
Output: The merged model
"""

import os
import json
import argparse
import pickle
import time
from typing import List, Dict, Any, Optional, Union, Tuple
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_alpha_coefficients(alpha_file: str) -> Dict[str, Any]:
    """Load alpha coefficient file"""
    print(f"📥 Loading Alpha coefficients: {alpha_file}")
    
    if not os.path.exists(alpha_file):
        raise FileNotFoundError(f"Alpha coefficient file not found: {alpha_file}")
    
    # Support both .pt and .json formats
    if alpha_file.endswith('.pt'):
        alpha_data = torch.load(alpha_file, map_location='cpu', weights_only=False)
        if isinstance(alpha_data, dict) and 'alpha_star' in alpha_data:
            alpha_star = alpha_data['alpha_star']
            axes = alpha_data['axes']
            merge_types = alpha_data.get('merge_types', 'qk')
        else:
            raise ValueError("Invalid alpha coefficients .pt file format")
    elif alpha_file.endswith('.json'):
        with open(alpha_file, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        alpha_star = torch.tensor(json_data['alpha'], dtype=torch.float32)
        axes_info = json_data['axes_info']
        merge_types = json_data.get('merge_types', 'qk')
        
        # Reconstruct axes structure
        axes = {
            'layers': axes_info['layers'],
            'heads': axes_info['heads'],
            'types': axes_info['types'],
            'dimensions': axes_info['dimensions'],
            'flat_index': axes_info['flat_index']
        }
    else:
        raise ValueError(f"Unsupported alpha coefficient file format: {alpha_file}")
    
    # Show alpha statistics
    print(f"📊 Alpha stats:")
    print(f"  Count: {len(alpha_star)}")
    print(f"  Merge Types: {merge_types.upper()}")
    print(f"  Mean: {alpha_star.mean():.4f}")
    print(f"  Std: {alpha_star.std():.4f}")
    print(f"  Range: [{alpha_star.min():.4f}, {alpha_star.max():.4f}]")
    print(f"  Non-zero ratio: {(alpha_star.abs() > 0.01).float().mean():.2%}")
    
    return {
        'alpha_star': alpha_star,
        'axes': axes,
        'merge_types': merge_types
    }


def load_projected_task_vectors(file_path: str) -> Dict[str, Any]:
    """Load projected task vectors"""
    print(f"📥 Loading projection result: {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Projection file not found: {file_path}")
    
    with open(file_path, 'rb') as f:
        projected_data = pickle.load(f)
    
    # Display loaded config
    config = projected_data["config"]
    stats = projected_data["projection_stats"]
    
    print(f"📋 Projection config:")
    print(f"  Merge Types: {config['merge_types'].upper()}")
    print(f"  Layers: {config['selected_layers']}")
    print(f"  Heads: {len(config['selected_heads'])}/{config['n_heads']}")
    print(f"  Compute dtype: {config['compute_dtype']}")
    print(f"  Total CG iterations: {stats['total_cg_iterations']}")
    print(f"  Total constraint residual: {stats['total_constraint_residual']:.6f}")
    
    file_size = os.path.getsize(file_path) / 1024 / 1024
    print(f"✅ Loaded: {file_size:.1f} MB")
    
    return projected_data


def _derive_head_alpha_from_qk(alpha_vec: torch.Tensor,
                               axes: Dict[str, Any],
                               selected_layers: List[int],
                               selected_heads: List[int],
                               couple_qk: bool) -> Dict[Tuple[int,int], float]:
    """
    Return per-(layer, head) alpha (to reuse for V/O).
    - Coupled: take α of (l,h,'QK')
    - Decoupled: average α of (l,h,'Q') and (l,h,'K') (use whichever exists)
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
    Return per-layer alpha (to reuse for FFN): average α of all known heads in the layer
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


def apply_weights_to_projected_task_vectors(
    projected_task_vectors: Dict[str, Any],
    alpha_data: Optional[Dict[str, Any]] = None,
    scaling_factor: Optional[float] = None,
    merge_types: str = "",
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Apply weights (alpha coefficients or scaling factor) to the projected task vectors.
    Supports weighting across Q/K/V/O/FFN.
    Intelligently handles incomplete alpha coverage, using the scaling factor for missing params.
    
    NEW: Integrates V/O/FFN alpha derivation logic from qp_true_forward_fast.py
    - V/O: reuse the same (layer, head) α (average Q/K if decoupled)
    - FFN: reuse the per-layer average of head αs
    """
    
    if alpha_data is None and scaling_factor is None:
        raise ValueError("You must provide at least one of alpha_data or scaling_factor.")
    
    # Determine actual parameter groups available in the projection file
    proj_merge_types = merge_types
    available_groups = set(projected_task_vectors.keys())
    
    # Determine which parameter types are covered by alpha
    alpha_covered_types = set()
    if alpha_data is not None:
        alpha_types = alpha_data["axes"]["types"]
        alpha_star = alpha_data["alpha_star"]
        axes = alpha_data["axes"]
        
        # Analyze alpha coverage
        for param_type in alpha_types:
            if param_type == "QK":
                alpha_covered_types.update(["Q", "K"])
            else:
                alpha_covered_types.add(param_type)
    
    # Default scaling used for params not covered by alpha
    default_scaling = scaling_factor if scaling_factor is not None else 1.0
    
    scaled = {"qk": {}, "vo": {}, "ffn": {}}
    counts = {"Q": 0, "K": 0, "V": 0, "O": 0, "Gate": 0, "Up": 0, "Down": 0}
    
    # Parse projection merge types (determine which params to process)
    merge_q = 'q' in proj_merge_types.lower()
    merge_k = 'k' in proj_merge_types.lower()
    merge_v = 'v' in proj_merge_types.lower()
    merge_o = 'o' in proj_merge_types.lower()
    merge_f = 'f' in proj_merge_types.lower()
    
    # Parameter type mapping
    type_mapping = {
        "Q": ("qk", "dQ_proj"),
        "K": ("qk", "dK_proj"),
        "V": ("vo", "dV_proj"),
        "O": ("vo", "dO_proj"),
        "Gate": ("ffn", "dGate_proj"),
        "Up": ("ffn", "dUp_proj"),
        "Down": ("ffn", "dDown_T_proj")
    }
    
    # Step 1: Process parameters covered by alpha (Q/K)
    if alpha_data is not None:
        print(f"🔧 Applying Alpha weighting...")
        
        # Apply by flat_index of alpha coefficients
        for i, (layer, head, param_type) in enumerate(axes["flat_index"]):
            alpha_val = float(alpha_star[i].item())
            
            # Handle QK_COUPLED mode: one alpha applied to both Q and K
            if param_type == "QK" and (merge_q or merge_k):
                # Initialize structure
                if layer not in scaled["qk"]:
                    scaled["qk"][layer] = {}
                if head not in scaled["qk"][layer]:
                    scaled["qk"][layer][head] = {}
                
                # Source
                src_data = projected_task_vectors.get("qk", {}).get(layer, {}).get(head, {})
                
                # Apply alpha to Q weights
                if merge_q and "dQ_proj" in src_data:
                    final_weight = alpha_val
                    if scaling_factor is not None:
                        final_weight *= scaling_factor
                    scaled["qk"][layer][head]["dQ_proj"] = src_data["dQ_proj"] * final_weight
                    counts["Q"] += 1
                
                # Apply alpha to K weights
                if merge_k and "dK_proj" in src_data:
                    final_weight = alpha_val
                    if scaling_factor is not None:
                        final_weight *= scaling_factor
                    scaled["qk"][layer][head]["dK_proj"] = src_data["dK_proj"] * final_weight
                    counts["K"] += 1
                
                continue
            
            # Handle decoupled Q/K
            should_process = False
            if param_type == "Q" and merge_q:
                should_process = True
            elif param_type == "K" and merge_k:
                should_process = True
            
            if not should_process:
                continue
            
            # Get group/key mapping
            if param_type not in type_mapping:
                continue
            
            group, key = type_mapping[param_type]
            
            # Initialize structure
            if layer not in scaled[group]:
                scaled[group][layer] = {}
            if head not in scaled[group][layer]:
                scaled[group][layer][head] = {}
            
            # Source
            src_data = projected_task_vectors.get(group, {}).get(layer, {}).get(head, {})
            if key in src_data:
                final_weight = alpha_val
                if scaling_factor is not None:
                    final_weight *= scaling_factor
                scaled[group][layer][head][key] = src_data[key] * final_weight
                counts[param_type] += 1
        
        # Step 1.5: Derive alpha for V/O/FFN using qp_true_forward_fast.py logic
        has_vo = "vo" in projected_task_vectors and len(projected_task_vectors["vo"]) > 0
        has_ffn = "ffn" in projected_task_vectors and len(projected_task_vectors["ffn"]) > 0
        
        if has_vo or has_ffn:
            # Determine coupling mode
            couple_qk = "QK" in alpha_types
            
            # Derive head/layer alphas
            layers = axes["layers"]
            heads = axes["heads"]
            head_alpha = _derive_head_alpha_from_qk(alpha_star, axes, layers, heads, couple_qk)
            layer_alpha = _derive_layer_alpha_from_heads(head_alpha, layers, heads)
            
            if verbose:
                print(f"🔧 Deriving Alpha for V/O/FFN:")
                print(f"  Head Alpha count: {len(head_alpha)}")
                print(f"  Layer Alpha count: {len(layer_alpha)}")
            
            # Apply derived alpha to V/O
            if has_vo and (merge_v or merge_o):
                for l, per_h in projected_task_vectors["vo"].items():
                    if l not in scaled["vo"]:
                        scaled["vo"][l] = {}
                    for h, src in per_h.items():
                        if h not in scaled["vo"][l]:
                            scaled["vo"][l][h] = {}
                        a = head_alpha.get((l,h), None)
                        if a is None:
                            continue
                        
                        # Apply derived alpha
                        final_alpha = a
                        if scaling_factor is not None:
                            final_alpha *= scaling_factor
                        
                        if merge_v and "dV_proj" in src:
                            scaled["vo"][l][h]["dV_proj"] = src["dV_proj"] * final_alpha
                            counts["V"] += 1
                        if merge_o and "dO_proj" in src:
                            scaled["vo"][l][h]["dO_proj"] = src["dO_proj"] * final_alpha
                            counts["O"] += 1
            
            # Apply derived alpha to FFN
            if has_ffn and merge_f:
                for l, tv in projected_task_vectors["ffn"].items():
                    a = layer_alpha.get(l, None)
                    if a is None:
                        continue
                    
                    if l not in scaled["ffn"]:
                        scaled["ffn"][l] = {}
                    
                    # Apply derived alpha
                    final_alpha = a
                    if scaling_factor is not None:
                        final_alpha *= scaling_factor
                    
                    for k_name, dW in tv.items():
                        if dW is not None:
                            scaled["ffn"][l][k_name] = dW * final_alpha
                            if k_name == "dGate_proj":
                                counts["Gate"] += 1
                            elif k_name == "dUp_proj":
                                counts["Up"] += 1
                            elif k_name == "dDown_T_proj":
                                counts["Down"] += 1
            
            # Update alpha-covered types
            if has_vo:
                alpha_covered_types.update(["V", "O"])
            if has_ffn:
                alpha_covered_types.update(["Gate", "Up", "Down"])
    
    # Step 2: Params not covered by Alpha (use scaling factor)
    uncovered_params = []
    if merge_v and "V" not in alpha_covered_types:
        uncovered_params.append("V")
    if merge_o and "O" not in alpha_covered_types:
        uncovered_params.append("O")
    if merge_f and not any(t in alpha_covered_types for t in ["Gate", "Up", "Down"]):
        uncovered_params.extend(["Gate", "Up", "Down"])
    
    if uncovered_params and default_scaling != 0:
        print(f"🔧 Applying Scaling Factor ({default_scaling}) to params not covered by Alpha: {uncovered_params}")
        
        # V/O
        if ("V" in uncovered_params or "O" in uncovered_params) and "vo" in projected_task_vectors:
            for layer, layer_data in projected_task_vectors["vo"].items():
                if layer not in scaled["vo"]:
                    scaled["vo"][layer] = {}
                
                for head, head_data in layer_data.items():
                    if head not in scaled["vo"][layer]:
                        scaled["vo"][layer][head] = {}
                    
                    # V
                    if "V" in uncovered_params and merge_v and "dV_proj" in head_data:
                        scaled["vo"][layer][head]["dV_proj"] = head_data["dV_proj"] * default_scaling
                        counts["V"] += 1
                    
                    # O
                    if "O" in uncovered_params and merge_o and "dO_proj" in head_data:
                        scaled["vo"][layer][head]["dO_proj"] = head_data["dO_proj"] * default_scaling
                        counts["O"] += 1
        
        # FFN
        ffn_uncovered = [p for p in uncovered_params if p in ["Gate", "Up", "Down"]]
        if ffn_uncovered and merge_f and "ffn" in projected_task_vectors:
            for layer, layer_data in projected_task_vectors["ffn"].items():
                if layer not in scaled["ffn"]:
                    scaled["ffn"][layer] = {}
                
                # Gate
                if "Gate" in ffn_uncovered and "dGate_proj" in layer_data:
                    scaled["ffn"][layer]["dGate_proj"] = layer_data["dGate_proj"] * default_scaling
                    counts["Gate"] += 1
                
                # Up
                if "Up" in ffn_uncovered and "dUp_proj" in layer_data:
                    scaled["ffn"][layer]["dUp_proj"] = layer_data["dUp_proj"] * default_scaling
                    counts["Up"] += 1
                
                # Down
                if "Down" in ffn_uncovered and "dDown_T_proj" in layer_data:
                    scaled["ffn"][layer]["dDown_T_proj"] = layer_data["dDown_T_proj"] * default_scaling
                    counts["Down"] += 1
    
    # Step 3: Scaling-Factor-only mode (no alpha)
    if alpha_data is None:
        print(f"🔧 Applying Scaling Factor: {scaling_factor}")
        
        # Process all available task vectors
        for group in ["qk", "vo", "ffn"]:
            if group not in projected_task_vectors:
                continue
                
            scaled[group] = {}
            
            for layer, layer_data in projected_task_vectors[group].items():
                scaled[group][layer] = {}
                
                if group == "ffn":
                    # FFN params have no head dimension
                    for key, param in layer_data.items():
                        if param is not None:
                            scaled[group][layer][key] = param * scaling_factor
                            
                            # Stats
                            if key == "dGate_proj":
                                counts["Gate"] += 1
                            elif key == "dUp_proj":
                                counts["Up"] += 1
                            elif key == "dDown_T_proj":
                                counts["Down"] += 1
                else:
                    # QK/VO params have head dimension
                    for head, head_data in layer_data.items():
                        scaled[group][layer][head] = {}
                        
                        for key, param in head_data.items():
                            if param is not None:
                                scaled[group][layer][head][key] = param * scaling_factor
                                
                                # Stats
                                if key == "dQ_proj":
                                    counts["Q"] += 1
                                elif key == "dK_proj":
                                    counts["K"] += 1
                                elif key == "dV_proj":
                                    counts["V"] += 1
                                elif key == "dO_proj":
                                    counts["O"] += 1
    
    if verbose:
        mode_str = ""
        if alpha_data is not None and uncovered_params:
            # Hybrid mode: part Alpha, part Scaling
            alpha_params = [t for t in ["Q", "K", "V", "O", "Gate", "Up", "Down"] if t in alpha_covered_types]
            mode_str = f"Hybrid: Alpha weighting ({alpha_params}) + Scaling ({default_scaling}) ({uncovered_params})"
        elif alpha_data is not None and scaling_factor is not None:
            mode_str = f"Alpha weighting × Scaling ({scaling_factor})"
        elif alpha_data is not None:
            mode_str = "Alpha weighting"
        elif scaling_factor is not None:
            mode_str = f"Scaling Factor ({scaling_factor})"
            
        print(f"🔧 {mode_str} application stats:")
        for param_type, count in counts.items():
            if count > 0:
                # Mark the weight source
                weight_source = ""
                if alpha_data is not None:
                    if param_type in alpha_covered_types:
                        weight_source = "(Alpha)"
                    elif param_type in uncovered_params:
                        weight_source = f"(Scaling={default_scaling})"
                else:
                    weight_source = f"(Scaling={scaling_factor})"
                
                print(f"  {param_type}: {count} parameters {weight_source}")
    
    return scaled


def apply_weighted_merge_to_model(
    model, 
    weighted_task_vectors: Dict[str, Any],
    config: Dict[str, Any],
    merge_info: Dict[str, Any]
) -> Dict[str, Any]:
    """Apply the weighted task vectors to the model"""
    
    mode_str = merge_info.get("mode", "Weighted Merge")
    print(f"🔧 Applying {mode_str}")
    
    # Extract info from config
    merge_types = config["merge_types"]
    selected_layers = config["selected_layers"]
    selected_heads = config["selected_heads"]
    d_model = config["d_model"]
    head_dim = config["head_dim"]
    kv_heads = config["kv_heads"]
    
    # Parse merge types
    merge_q = 'q' in merge_types.lower()
    merge_k = 'k' in merge_types.lower()
    merge_v = 'v' in merge_types.lower()
    merge_o = 'o' in merge_types.lower()
    merge_f = 'f' in merge_types.lower()
    
    stats = {
        "total_params_modified": 0,
        "total_norm_q": 0.0,
        "total_norm_k": 0.0,
        "total_norm_v": 0.0,
        "total_norm_o": 0.0,
        "total_norm_ffn": 0.0,
        "layer_stats": {},
        "merge_info": merge_info
    }
    
    with torch.no_grad():
        for li in tqdm(selected_layers, desc=f"Applying {mode_str}"):
            layer_stats = {"heads": {}}
            
            # Q/K updates
            if (merge_q or merge_k) and li in weighted_task_vectors["qk"]:
                layer_target = model.model.layers[li].self_attn
                
                for h in selected_heads:
                    if h in weighted_task_vectors["qk"][li]:
                        head_stat = {"params_modified": 0, "norm_q": 0.0, "norm_k": 0.0}
                        weighted_data = weighted_task_vectors["qk"][li][h]
                        
                        # Q
                        if merge_q and "dQ_proj" in weighted_data:
                            dQ_weighted = weighted_data["dQ_proj"]
                            WQ_target = layer_target.q_proj.weight.data
                            q_start, q_end = h * head_dim, (h + 1) * head_dim
                            WQ_target[q_start:q_end, :] += dQ_weighted.T.to(WQ_target.device)
                            
                            head_stat["norm_q"] = dQ_weighted.norm().item()
                            head_stat["params_modified"] += dQ_weighted.numel()
                        
                        # K
                        if merge_k and "dK_proj" in weighted_data:
                            dK_weighted = weighted_data["dK_proj"]
                            WK_target = layer_target.k_proj.weight.data
                            kvh = h % kv_heads
                            k_start, k_end = kvh * head_dim, (kvh + 1) * head_dim
                            WK_target[k_start:k_end, :] += dK_weighted.T.to(WK_target.device)
                            
                            head_stat["norm_k"] = dK_weighted.norm().item()
                            head_stat["params_modified"] += dK_weighted.numel()
                        
                        layer_stats["heads"][h] = head_stat
                        stats["total_params_modified"] += head_stat["params_modified"]
                        stats["total_norm_q"] += head_stat["norm_q"]
                        stats["total_norm_k"] += head_stat["norm_k"]
            
            # V/O updates
            if (merge_v or merge_o) and li in weighted_task_vectors["vo"]:
                layer_target = model.model.layers[li].self_attn
                
                for h in selected_heads:
                    if h in weighted_task_vectors["vo"][li]:
                        if h not in layer_stats["heads"]:
                            layer_stats["heads"][h] = {"params_modified": 0, "norm_v": 0.0, "norm_o": 0.0}
                        head_stat = layer_stats["heads"][h]
                        weighted_data = weighted_task_vectors["vo"][li][h]
                        
                        # V
                        if merge_v and "dV_proj" in weighted_data:
                            dV_weighted = weighted_data["dV_proj"]
                            WV_target = layer_target.v_proj.weight.data
                            kvh = h % kv_heads
                            v_rows = slice(kvh*head_dim, (kvh+1)*head_dim)
                            WV_target[v_rows, :] += dV_weighted.T.to(WV_target.device)
                            
                            head_stat["norm_v"] = dV_weighted.norm().item()
                            head_stat["params_modified"] += dV_weighted.numel()
                        
                        # O
                        if merge_o and "dO_proj" in weighted_data:
                            dO_weighted = weighted_data["dO_proj"]
                            WO_target = layer_target.o_proj.weight.data
                            o_cols = slice(h*head_dim, (h+1)*head_dim)
                            WO_target[:, o_cols] += dO_weighted.to(WO_target.device)
                            
                            head_stat["norm_o"] = dO_weighted.norm().item()
                            head_stat["params_modified"] += dO_weighted.numel()
                        
                        stats["total_params_modified"] += head_stat.get("params_modified", 0)
                        stats["total_norm_v"] += head_stat.get("norm_v", 0.0)
                        stats["total_norm_o"] += head_stat.get("norm_o", 0.0)
            
            # FFN updates
            if merge_f and li in weighted_task_vectors["ffn"]:
                weighted_data = weighted_task_vectors["ffn"][li]
                
                # Initialize FFN stats
                layer_stats["ffn"] = {
                    "norm_gate": 0.0, "norm_up": 0.0, "norm_down": 0.0,
                    "params_modified": 0
                }
                
                # Gate
                if "dGate_proj" in weighted_data and weighted_data["dGate_proj"] is not None:
                    dGate_weighted = weighted_data["dGate_proj"]
                    Wg_target = model.model.layers[li].mlp.gate_proj.weight.data
                    Wg_target += dGate_weighted.to(Wg_target.device)
                    
                    layer_stats["ffn"]["norm_gate"] = dGate_weighted.norm().item()
                    layer_stats["ffn"]["params_modified"] += dGate_weighted.numel()
                
                # Up
                if "dUp_proj" in weighted_data and weighted_data["dUp_proj"] is not None:
                    dUp_weighted = weighted_data["dUp_proj"]
                    Wu_target = model.model.layers[li].mlp.up_proj.weight.data
                    Wu_target += dUp_weighted.to(Wu_target.device)
                    
                    layer_stats["ffn"]["norm_up"] = dUp_weighted.norm().item()
                    layer_stats["ffn"]["params_modified"] += dUp_weighted.numel()
                
                # Down
                if "dDown_T_proj" in weighted_data and weighted_data["dDown_T_proj"] is not None:
                    dDown_T_weighted = weighted_data["dDown_T_proj"]
                    Wd_target = model.model.layers[li].mlp.down_proj.weight.data
                    Wd_target += dDown_T_weighted.T.to(Wd_target.device)
                    
                    layer_stats["ffn"]["norm_down"] = dDown_T_weighted.norm().item()
                    layer_stats["ffn"]["params_modified"] += dDown_T_weighted.numel()
                
                # Aggregate FFN stats
                total_ffn_norm = (layer_stats["ffn"]["norm_gate"] + 
                                  layer_stats["ffn"]["norm_up"] + 
                                  layer_stats["ffn"]["norm_down"])
                layer_stats["ffn"]["norm"] = total_ffn_norm
                
                stats["total_norm_ffn"] += total_ffn_norm
                stats["total_params_modified"] += layer_stats["ffn"]["params_modified"]
            
            stats["layer_stats"][li] = layer_stats
    
    print(f"✅ {mode_str} done:")
    print(f"  Params modified: {stats['total_params_modified']:,}")
    if merge_q:
        print(f"  Q weight norm: {stats['total_norm_q']:.6f}")
    if merge_k:
        print(f"  K weight norm: {stats['total_norm_k']:.6f}")
    if merge_v:
        print(f"  V weight norm: {stats['total_norm_v']:.6f}")
    if merge_o:
        print(f"  O weight norm: {stats['total_norm_o']:.6f}")
    if merge_f:
        print(f"  FFN weight norm: {stats['total_norm_ffn']:.6f}")
    
    return stats


def unified_model_merge(
    base_model_path: str,
    projected_file: str,
    alpha_file: Optional[str] = None,
    scaling_factor: Optional[float] = None,
    output_dir: str = "./unified_merge_output",
    model_name: str = "unified_merged_model",
    verbose: bool = True
) -> Dict[str, Any]:
    """Run the unified model merge"""
    
    print(f"🚀 Unified Model Merge")
    print("=" * 70)
    
    # Validate inputs
    if alpha_file is None and scaling_factor is None:
        raise ValueError("You must provide at least one of alpha_file or scaling_factor.")
    
    # Load projection data
    projected_data = load_projected_task_vectors(projected_file)
    projected_task_vectors = projected_data["projected_task_vectors"]
    config = projected_data["config"]
    
    # Load alpha coefficients (if provided)
    alpha_data = None
    if alpha_file is not None:
        alpha_data = load_alpha_coefficients(alpha_file)
    
    # Merge mode & info
    merge_mode = ""
    merge_info = {}
    
    if alpha_data is not None and scaling_factor is not None:
        merge_mode = f"Alpha weighting × Scaling ({scaling_factor})"
        merge_info = {
            "mode": merge_mode,
            "alpha_info": {
                "merge_types": alpha_data["merge_types"],
                "alpha_stats": {
                    "mean": float(alpha_data["alpha_star"].mean().item()),
                    "std": float(alpha_data["alpha_star"].std().item()),
                    "min": float(alpha_data["alpha_star"].min().item()),
                    "max": float(alpha_data["alpha_star"].max().item()),
                    "total": len(alpha_data["alpha_star"]),
                    "nonzero_ratio": float((alpha_data["alpha_star"].abs() > 0.01).float().mean().item())
                }
            },
            "scaling_factor": scaling_factor
        }
    elif alpha_data is not None:
        merge_mode = "Alpha weighting only"
        merge_info = {
            "mode": merge_mode,
            "alpha_info": {
                "merge_types": alpha_data["merge_types"],
                "alpha_stats": {
                    "mean": float(alpha_data["alpha_star"].mean().item()),
                    "std": float(alpha_data["alpha_star"].std().item()),
                    "min": float(alpha_data["alpha_star"].min().item()),
                    "max": float(alpha_data["alpha_star"].max().item()),
                    "total": len(alpha_data["alpha_star"]),
                    "nonzero_ratio": float((alpha_data["alpha_star"].abs() > 0.01).float().mean().item())
                }
            }
        }
    else:
        merge_mode = f"Scaling Factor only (×{scaling_factor})"
        merge_info = {
            "mode": merge_mode,
            "scaling_factor": scaling_factor
        }
    
    print(f"🔍 Merge mode: {merge_mode}")
    proj_merge_types = config["merge_types"]
    print(f"  Projection merge types: {proj_merge_types.upper()}")
    
    # Apply weights to projected task vectors
    print(f"\n🔧 Applying weights to projected Task Vectors...")
    weighted_task_vectors = apply_weights_to_projected_task_vectors(
        projected_task_vectors, alpha_data, scaling_factor, proj_merge_types, verbose
    )
    
    # Load base model
    print(f"\n📥 Loading base model: {base_model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True
    ).eval()
    
    # Apply weighted merge to model
    print(f"\n🔧 Applying weighted merge to model...")
    merge_stats = apply_weighted_merge_to_model(
        model, weighted_task_vectors, config, merge_info
    )
    
    # Save merged model
    model_output_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)
    
    print(f"\n💾 Saving merged model: {model_output_dir}")
    model.save_pretrained(model_output_dir)
    
    # Save tokenizer too
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.save_pretrained(model_output_dir)
    
    # Save merge stats
    stats_file = os.path.join(output_dir, "unified_merge_stats.json")
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(merge_stats, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"📊 Saved merge stats: {stats_file}")
    
    return {
        "model": model,
        "stats": merge_stats,
        "weighted_task_vectors": weighted_task_vectors
    }


def main():
    parser = argparse.ArgumentParser(description="Unified Model Merge - Supports Alpha weighting and Scaling Factor modes")
    
    # Inputs
    parser.add_argument("--projected_file", type=str, required=True,
                       help="Path to projected task vectors (*.pkl)")
    parser.add_argument("--base_model", type=str, required=True,
                       help="Path or HF id of the base model to merge into")
    
    # Merge options
    parser.add_argument("--alpha_file", type=str, default=None,
                       help="Path to alpha coefficients (*.pt or *.json), optional")
    parser.add_argument("--scaling_factor", type=float, default=None,
                       help="Scaling factor, optional")
    
    # Outputs
    parser.add_argument("--output_dir", type=str, default="./unified_merge_output",
                       help="Output directory")
    parser.add_argument("--model_name", type=str, default="unified_merged_model",
                       help="Name for the merged model directory")
    
    # Misc
    parser.add_argument("--verbose", action="store_true",
                       help="Verbose output")
    
    args = parser.parse_args()
    
    # Validate args
    if args.alpha_file is None and args.scaling_factor is None:
        print("❌ Error: You must provide at least one of --alpha_file or --scaling_factor")
        parser.print_help()
        return
    
    print("🚀 Unified Model Merge")
    print("=" * 70)
    print(f"Projected file: {args.projected_file}")
    print(f"Alpha file: {args.alpha_file if args.alpha_file else 'None'}")
    print(f"Scaling Factor: {args.scaling_factor if args.scaling_factor else 'None'}")
    print(f"Base model: {args.base_model}")
    print(f"Output dir: {args.output_dir}")
    print(f"Model name: {args.model_name}")
    print("=" * 70)
    
    start_time = time.time()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Execute merge
    result = unified_model_merge(
        args.base_model, args.projected_file, args.alpha_file, args.scaling_factor,
        args.output_dir, args.model_name, args.verbose
    )
    
    end_time = time.time()
    
    print(f"\n✅ Unified model merge complete! Elapsed: {end_time - start_time:.1f}s")
    print(f"📁 Output dir: {args.output_dir}")
    print(f"🤖 Merged model: {os.path.join(args.output_dir, args.model_name)}")
    print(f"📊 Merge stats: {os.path.join(args.output_dir, 'unified_merge_stats.json')}")
    print(f"🎉 Next step: Evaluate the merged model on downstream tasks")
    

if __name__ == "__main__":
    main()
