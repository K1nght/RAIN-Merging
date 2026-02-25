#!/usr/bin/env python3
"""
Null-space projection script - Stage 1
Purpose: compute and save projected task vectors without applying the scaling factor
Output: a file with projected task vectors to be used by later scaling
"""

import os
import json
import math
import argparse
import re
import random
import gc
import pickle
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any, Optional
from tqdm import tqdm
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import core functions
from nullspace_merge_qkvo_ffn import (
    ensure_dir, cleanup_memory, print_memory_status,
    PreparedSample, prepare_samples_unified,
    build_constraints_single_layer_unified,
    task_vectors_single_layer_unified,
    cg_single_head_batched, cg_v, cg_o, cg_ffn_down, cg_ffn_gate, cg_ffn_up,
    # High-efficiency dense solvers
    ffn_down_dense_project, ffn_gate_dense_project, ffn_up_dense_project
)


def read_json_samples_robust(path: str, tokenizer, max_n: Optional[int] = None) -> List[str]:
    """Read samples from a JSON file and build full conversations (robust version, supports multiple formats)"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Handle different JSON shapes
    if isinstance(data, list):
        samples = data
    elif isinstance(data, dict):
        # Wrap single dict as a list
        samples = [data]
    else:
        raise ValueError(f"Unsupported JSON format: {type(data)}")
    
    full_prompts = []
    for sample in samples:
        if max_n is not None and len(full_prompts) >= max_n:
            break
        
        # Check sample format and provide defaults
        if isinstance(sample, str):
            # If sample is a raw string
            prompt = sample
            reasoning = ''
            response = ''
        elif isinstance(sample, dict):
            # If sample is a dict, try to extract fields
            prompt = sample.get('prompt', sample.get('text', str(sample)))
            reasoning = sample.get('reasoning', '')
            response = sample.get('response', '')
        else:
            # Fallback: convert to string
            prompt = str(sample)
            reasoning = ''
            response = ''
        
        # Build chat messages
        messages = [{"role": "user", "content": prompt}]
        
        # Apply chat template
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Build full conversation
        full_prompt = formatted_prompt + reasoning + "\n</think>\n\n" + response
        full_prompts.append(full_prompt)
    
    return full_prompts


def compute_nullspace_projections(
    model_base, model_instruct, model_target,
    texts_R: List[str], tokenizer,
    selected_layers: List[int], selected_heads: List[int],
    neigh_radius: int, lambda_ridge: float, cg_maxit: int, cg_tol: float, 
    compute_dtype: torch.dtype = torch.float32,
    merge_types: str = "qk",
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
    use_hooks: bool = True,
    # Sequence length limit
    max_seq_len: int = 7168
) -> Dict[str, Any]:
    """Compute null-space projected task vectors (without applying the scaling factor)"""
    
    print("🚀 Starting computation of null-space projected task vectors...")
    rng = random.Random(seed)
    
    d_model = model_target.config.hidden_size
    n_heads = model_target.config.num_attention_heads
    head_dim = d_model // n_heads
    kv_heads = getattr(model_target.config, 'num_key_value_heads', n_heads)
    
    print(f"📋 Config: d_model={d_model}, n_heads={n_heads}, kv_heads={kv_heads}")
    print(f"Feature extraction: {'Hook-based (recommended)' if use_hooks else 'Original method'}")
    
    # 1) Preprocess samples
    prepped_samples = prepare_samples_unified(
        texts_R, tokenizer, neigh_radius, merge_types,
        q_rows_per_text, k_rows_per_text, v_rows_per_text, o_rows_per_text, ffn_rows_per_text, rng
    )
    
    # Device assignment
    if qk_device == "auto":
        qk_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if vo_device == "auto":
        vo_device = "cuda:1" if torch.cuda.device_count() > 1 else qk_device
    if ffn_device == "auto":
        ffn_device = "cuda:2" if torch.cuda.device_count() > 2 else vo_device
    
    print(f"🔧 Device mapping: QK={qk_device}, VO={vo_device}, FFN={ffn_device}")
    
    # Parse merge types
    merge_q = 'q' in merge_types.lower()
    merge_k = 'k' in merge_types.lower()
    merge_v = 'v' in merge_types.lower()
    merge_o = 'o' in merge_types.lower()
    merge_f = 'f' in merge_types.lower()
    
    print(f"🎯 Merge types: {merge_types.upper()} (Q={merge_q}, K={merge_k}, V={merge_v}, O={merge_o}, F={merge_f})")
    
    # 2) Extract raw task vectors (no scaling factor applied)
    print("\n🎯 Extracting raw task vectors...")
    all_layer_task_vectors_raw = {}
    for li in tqdm(selected_layers, desc="Extract task vectors for all layers"):
        layer_task_vectors = task_vectors_single_layer_unified(
            model_base, model_instruct, li, selected_heads, merge_types, scaling_factor=1.0
        )
        all_layer_task_vectors_raw[li] = layer_task_vectors
    
    # 3) Compute projected task vectors (process layer-by-layer to save VRAM)
    print("\n🔬 Computing null-space projection...")
    projected_task_vectors = {
        "qk": {},  # {layer: {head: {"dQ_proj": tensor, "dK_proj": tensor}}}
        "vo": {},  # {layer: {head: {"dV_proj": tensor, "dO_proj": tensor}}}
        "ffn": {}  # {layer: {"dGate_proj": tensor, "dUp_proj": tensor, "dDown_T_proj": tensor}}
    }
    
    projection_stats = {
        "total_cg_iterations": 0,
        "total_constraint_residual": 0.0,
        "layer_stats": {}
    }
    # Load a temporary model per layer to save VRAM
    print("🔧 Per-layer processing mode (VRAM-friendly)...")
    
    for li_idx, li in enumerate(tqdm(selected_layers, desc="Projection per layer")):
        print(f"\n🔄 Processing layer {li} ({li_idx+1}/{len(selected_layers)})")
        
        # Temporarily load a model for constraint construction
        print(f"  📥 Temporarily loading model...")
        model_R_temp = AutoModelForCausalLM.from_pretrained(
            model_target.config._name_or_path,
            torch_dtype=torch.bfloat16,
            device_map="auto" if torch.cuda.is_available() else "cpu",
            trust_remote_code=True
        ).eval()

        # Build constraints for the current layer
        print(f"  📐 Building constraints for layer {li}...")
        layer_cons = build_constraints_single_layer_unified(
            model_R_temp, prepped_samples, li, selected_heads, merge_types,
            w_q, w_k, q_rows_per_text, k_rows_per_text,
            w_v, w_o, v_rows_per_text, o_rows_per_text,
            w_ffn, ffn_rows_per_text, readout_dirs,
            qk_device, vo_device, ffn_device, compute_dtype, use_hooks, max_seq_len
        )
        
        # Immediately free the temporary model
        del model_R_temp
        cleanup_memory()
        
        layer_stats = {"heads": {}}
        
        # Get task vectors for this layer
        if li not in all_layer_task_vectors_raw:
            continue
            
        layer_task_raw = all_layer_task_vectors_raw[li]
        
        # Initialize storage for this layer
        if merge_q or merge_k:
            projected_task_vectors["qk"][li] = {}
        if merge_v or merge_o:
            projected_task_vectors["vo"][li] = {}
        
        # Per-head CG projection
        for h in tqdm(selected_heads, desc=f"Per-head projection for layer {li}", leave=False):
            head_stat = {
                "constraints_qk": 0, "constraints_v": 0, "constraints_o": 0,
                "residual_norm_qk": 0.0, "residual_norm_v": 0.0, "residual_norm_o": 0.0,
                "cg_iterations": 0
            }
            
            # QK projection
            if (merge_q or merge_k) and "qk" in layer_cons and h in layer_cons["qk"]:
                cons_h_qk = layer_cons["qk"][h]
                total_constraints_qk = cons_h_qk["Xi_q"].shape[0] + cons_h_qk["Xj_k"].shape[0]
                head_stat["constraints_qk"] = total_constraints_qk
                
                if total_constraints_qk > 0 and h in layer_task_raw["qk"]:
                    task_qk = layer_task_raw["qk"][h]
                    task_dQ = task_qk.get("dQ") if merge_q and "dQ" in task_qk else None
                    task_dK = task_qk.get("dK") if merge_k and "dK" in task_qk else None
                    
                    if task_dQ is not None and task_dK is not None:
                        # CG projection solve
                        dQ_proj, dK_proj, cg_info = cg_single_head_batched(
                            cons_h_qk, task_dQ, task_dK, lambda_ridge, cg_maxit, cg_tol, 
                            device=qk_device, compute_dtype=compute_dtype
                        )
                        
                        # Save projected results
                        projected_task_vectors["qk"][li][h] = {}
                        if merge_q:
                            projected_task_vectors["qk"][li][h]["dQ_proj"] = dQ_proj.cpu()
                        if merge_k:
                            projected_task_vectors["qk"][li][h]["dK_proj"] = dK_proj.cpu()
                        
                        head_stat["residual_norm_qk"] = cg_info["residual_norm"]
                        head_stat["cg_iterations"] += cg_info["iterations"]
                        print(f"    ✅ QK projection done: {cg_info.get('solver', 'cg')}, "
                              f"residual={cg_info['residual_norm']:.2e}, time={cg_info.get('time', 0.0):.2f}s")
            
            # V projection
            if merge_v and "vo" in layer_cons and h in layer_cons["vo"]:
                cons_h_v = layer_cons["vo"][h]
                if "Xi_v" in cons_h_v and cons_h_v["Xi_v"].numel() > 0:
                    head_stat["constraints_v"] = cons_h_v["Xi_v"].shape[0]
                
                    if h in layer_task_raw["vo"] and "dV" in layer_task_raw["vo"][h]:
                        dV_task = layer_task_raw["vo"][h]["dV"]
                        dV_proj, info_v = cg_v(cons_h_v, dV_task, lambda_ridge, cg_maxit, cg_tol, 
                                             device=vo_device, compute_dtype=compute_dtype)
                        
                        # Save projected results
                        if li not in projected_task_vectors["vo"]:
                            projected_task_vectors["vo"][li] = {}
                        if h not in projected_task_vectors["vo"][li]:
                            projected_task_vectors["vo"][li][h] = {}
                        projected_task_vectors["vo"][li][h]["dV_proj"] = dV_proj.cpu()
                        
                        head_stat["residual_norm_v"] = info_v["residual_norm"]
                        head_stat["cg_iterations"] += info_v["iterations"]
                        print(f"    ✅ V projection done: {info_v.get('solver', 'cg')}, "
                              f"residual={info_v['residual_norm']:.2e}, time={info_v.get('time', 0.0):.2f}s")
            # O projection
            if merge_o and "vo" in layer_cons and h in layer_cons["vo"]:
                cons_h_o = layer_cons["vo"][h]
                if "c_vec" in cons_h_o and cons_h_o["c_vec"].numel() > 0:
                    head_stat["constraints_o"] = cons_h_o["c_vec"].shape[0]
                
                    if h in layer_task_raw["vo"] and "dO" in layer_task_raw["vo"][h]:
                        dO_task = layer_task_raw["vo"][h]["dO"]
                        dO_proj, info_o = cg_o(cons_h_o, dO_task, lambda_ridge, cg_maxit, cg_tol, 
                                             device=vo_device, compute_dtype=compute_dtype)
                        
                        # Save projected results
                        if li not in projected_task_vectors["vo"]:
                            projected_task_vectors["vo"][li] = {}
                        if h not in projected_task_vectors["vo"][li]:
                            projected_task_vectors["vo"][li][h] = {}
                        projected_task_vectors["vo"][li][h]["dO_proj"] = dO_proj.cpu()
                        
                        head_stat["residual_norm_o"] = info_o["residual_norm"]
                        head_stat["cg_iterations"] += info_o["iterations"]
                        print(f"    ✅ O projection done: {info_o.get('solver', 'cg')}, "
                              f"residual={info_o['residual_norm']:.2e}, time={info_o.get('time', 0.0):.2f}s")
            
            layer_stats["heads"][h] = head_stat
            projection_stats["total_cg_iterations"] += head_stat["cg_iterations"]
            projection_stats["total_constraint_residual"] += (head_stat["residual_norm_qk"] + 
                                                            head_stat["residual_norm_v"] + 
                                                            head_stat["residual_norm_o"])
        
        # FFN projection (once per layer, includes gate/up/down)
        if merge_f and "ffn" in layer_cons and "ffn" in layer_task_raw:
            print(f"    🔧 Processing FFN projection for layer {li}...")
            ffn_cons = layer_cons["ffn"]
            layer_task_ffn = layer_task_raw["ffn"]
            
            # Initialize FFN result storage
            projected_task_vectors["ffn"][li] = {}
            
            # FFN stats
            layer_stats["ffn"] = {
                "constraints_gate": 0, "constraints_up": 0, "constraints_down": 0,
                "residual_norm_gate": 0.0, "residual_norm_up": 0.0, "residual_norm_down": 0.0,
                "cg_iterations": 0
            }
            
            # Gate projection (adaptive solver choice)
            if "dGate" in layer_task_ffn and ffn_cons.get("X_gate", torch.empty(0)).numel() > 0:
                m_gate = ffn_cons["X_gate"].shape[0]
                print(f"      📐 Gate projection (m={m_gate})...")
                
                dGate_task = layer_task_ffn["dGate"]
                start_time = time.time()
                
                # Adaptive choice: explicit for small m, CG for large m
                if m_gate <= 4000:  # heuristic threshold, tune per memory
                    print(f"        🚀 Using Cholesky explicit solver...")
                    dGate_proj, info_gate = ffn_gate_dense_project(ffn_cons, dGate_task,
                                                                  lam=lambda_ridge,
                                                                  device=ffn_device,
                                                                  compute_dtype=compute_dtype)
                else:
                    print(f"        🔄 Using CG iterative solver...")
                    dGate_proj, info_gate = cg_ffn_gate(ffn_cons, dGate_task, lambda_ridge, 
                                                       cg_maxit, cg_tol,
                                                       device=ffn_device, 
                                                       compute_dtype=compute_dtype)
                
                gate_time = time.time() - start_time
                projected_task_vectors["ffn"][li]["dGate_proj"] = dGate_proj.cpu()
                layer_stats["ffn"]["constraints_gate"] = m_gate
                layer_stats["ffn"]["residual_norm_gate"] = info_gate["residual_norm"]
                layer_stats["ffn"]["cg_iterations"] += info_gate["iterations"]
                layer_stats["ffn"]["gate_solver"] = info_gate.get("solver", "cg")
                layer_stats["ffn"]["gate_time"] = gate_time
                print(f"        ✅ Gate solve done: {info_gate.get('solver', 'cg')}, "
                      f"residual={info_gate['residual_norm']:.2e}, time={gate_time:.2f}s")
            elif "dGate" in layer_task_ffn:
                # No constraints: apply directly
                print(f"      📐 Gate applied directly (no constraints)...")
                projected_task_vectors["ffn"][li]["dGate_proj"] = layer_task_ffn["dGate"].cpu()
                layer_stats["ffn"]["constraints_gate"] = 0
                layer_stats["ffn"]["residual_norm_gate"] = 0.0
                layer_stats["ffn"]["gate_solver"] = "direct"
                layer_stats["ffn"]["gate_time"] = 0.0
            
            # Up projection (adaptive solver choice)
            if "dUp" in layer_task_ffn and ffn_cons.get("X_up", torch.empty(0)).numel() > 0:
                m_up = ffn_cons["X_up"].shape[0]
                print(f"      📐 Up projection (m={m_up})...")
                
                dUp_task = layer_task_ffn["dUp"]
                start_time = time.time()
                
                # Adaptive choice: explicit for small m, CG for large m
                if m_up <= 4000:  # heuristic threshold, tune per memory
                    print(f"        🚀 Using Cholesky explicit solver...")
                    dUp_proj, info_up = ffn_up_dense_project(ffn_cons, dUp_task,
                                                            lam=lambda_ridge,
                                                            device=ffn_device,
                                                            compute_dtype=compute_dtype)
                else:
                    print(f"        🔄 Using CG iterative solver...")
                    dUp_proj, info_up = cg_ffn_up(ffn_cons, dUp_task, lambda_ridge, 
                                                 cg_maxit, cg_tol,
                                                 device=ffn_device, 
                                                 compute_dtype=compute_dtype)
                
                up_time = time.time() - start_time
                projected_task_vectors["ffn"][li]["dUp_proj"] = dUp_proj.cpu()
                layer_stats["ffn"]["constraints_up"] = m_up
                layer_stats["ffn"]["residual_norm_up"] = info_up["residual_norm"]
                layer_stats["ffn"]["cg_iterations"] += info_up["iterations"]
                layer_stats["ffn"]["up_solver"] = info_up.get("solver", "cg")
                layer_stats["ffn"]["up_time"] = up_time
                print(f"        ✅ Up solve done: {info_up.get('solver', 'cg')}, "
                      f"residual={info_up['residual_norm']:.2e}, time={up_time:.2f}s")
            elif "dUp" in layer_task_ffn:
                # No constraints: apply directly
                print(f"      📐 Up applied directly (no constraints)...")
                projected_task_vectors["ffn"][li]["dUp_proj"] = layer_task_ffn["dUp"].cpu()
                layer_stats["ffn"]["constraints_up"] = 0
                layer_stats["ffn"]["residual_norm_up"] = 0.0
                layer_stats["ffn"]["up_solver"] = "direct"
                layer_stats["ffn"]["up_time"] = 0.0
            
            # Down projection (adaptive solver choice)
            if "dDown_T" in layer_task_ffn and ffn_cons.get("H", torch.empty(0)).numel() > 0:
                m_down = ffn_cons["H"].shape[0]
                print(f"      📐 Down projection (m={m_down})...")
                
                dDown_T_task = layer_task_ffn["dDown_T"]
                start_time = time.time()
                
                # Adaptive choice: explicit for small m, CG for large m
                if m_down <= 4000:  # heuristic threshold, tune per memory
                    print(f"        🚀 Using Cholesky explicit solver...")
                    dDown_T_proj, info_down = ffn_down_dense_project(ffn_cons, dDown_T_task,
                                                                   lam=lambda_ridge,
                                                                   device=ffn_device,
                                                                   compute_dtype=compute_dtype)
                else:
                    print(f"        🔄 Using CG iterative solver...")
                    dDown_T_proj, info_down = cg_ffn_down(ffn_cons, dDown_T_task, lambda_ridge, 
                                                        cg_maxit, cg_tol,
                                                        device=ffn_device, 
                                                        compute_dtype=compute_dtype)
                
                down_time = time.time() - start_time
                projected_task_vectors["ffn"][li]["dDown_T_proj"] = dDown_T_proj.cpu()
                layer_stats["ffn"]["constraints_down"] = m_down
                layer_stats["ffn"]["residual_norm_down"] = info_down["residual_norm"]
                layer_stats["ffn"]["cg_iterations"] += info_down["iterations"]
                layer_stats["ffn"]["down_solver"] = info_down.get("solver", "cg")
                layer_stats["ffn"]["down_time"] = down_time
                print(f"        ✅ Down solve done: {info_down.get('solver', 'cg')}, "
                      f"residual={info_down['residual_norm']:.2e}, time={down_time:.2f}s")
            elif "dDown_T" in layer_task_ffn:
                # No constraints: apply directly
                print(f"      📐 Down applied directly (no constraints)...")
                projected_task_vectors["ffn"][li]["dDown_T_proj"] = layer_task_ffn["dDown_T"].cpu()
                layer_stats["ffn"]["constraints_down"] = 0
                layer_stats["ffn"]["residual_norm_down"] = 0.0
                layer_stats["ffn"]["down_solver"] = "direct"
                layer_stats["ffn"]["down_time"] = 0.0
            
            # Update totals
            total_residual = (layer_stats["ffn"]["residual_norm_gate"] + 
                            layer_stats["ffn"]["residual_norm_up"] + 
                            layer_stats["ffn"]["residual_norm_down"])
            projection_stats["total_constraint_residual"] += total_residual
            projection_stats["total_cg_iterations"] += layer_stats["ffn"]["cg_iterations"]
        
        projection_stats["layer_stats"][li] = layer_stats
        
        # Free current layer's constraints to save VRAM
        del layer_cons
        cleanup_memory()
        print(f"  🧹 Cleared constraints for layer {li}, freed VRAM")
    
    print(f"\n✅ Null-space projection finished!")
    print(f"  📊 Totals:")
    print(f"     - Total CG iterations: {projection_stats['total_cg_iterations']}")
    print(f"     - Sum of constraint residuals: {projection_stats['total_constraint_residual']:.6f}")
    
    # Solver usage stats
    solver_stats = {"dense_cholesky": 0, "cg": 0, "direct": 0}
    total_solver_time = 0.0
    
    for layer_stat in projection_stats["layer_stats"].values():
        if "ffn" in layer_stat:
            ffn_stat = layer_stat["ffn"]
            for solver_key in ["gate_solver", "up_solver", "down_solver"]:
                if solver_key in ffn_stat:
                    solver_type = ffn_stat[solver_key]
                    solver_stats[solver_type] = solver_stats.get(solver_type, 0) + 1
            
            for time_key in ["gate_time", "up_time", "down_time"]:
                if time_key in ffn_stat:
                    total_solver_time += ffn_stat[time_key]
    
    print(f"  🚀 FFN solver performance:")
    print(f"     - Cholesky explicit: {solver_stats.get('dense_cholesky', 0)} time(s)")
    print(f"     - CG iterative: {solver_stats.get('cg', 0)} time(s)") 
    print(f"     - Direct application: {solver_stats.get('direct', 0)} time(s)")
    print(f"     - Total FFN solve time: {total_solver_time:.2f}s")
    
    return {
        "projected_task_vectors": projected_task_vectors,
        "projection_stats": projection_stats,
        "config": {
            "merge_types": merge_types,
            "selected_layers": selected_layers,
            "selected_heads": selected_heads,
            "d_model": d_model,
            "n_heads": n_heads,
            "head_dim": head_dim,
            "kv_heads": kv_heads,
            "compute_dtype": str(compute_dtype),
            "lambda_ridge": lambda_ridge,
            "cg_maxit": cg_maxit,
            "cg_tol": cg_tol
        }
    }


def save_projected_task_vectors(projected_data: Dict[str, Any], output_path: str):
    """Save the projected task vectors to file"""
    print(f"💾 Saving projections to: {output_path}")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Use pickle (supports torch tensors)
    with open(output_path, 'wb') as f:
        pickle.dump(projected_data, f)
    
    # Also save a JSON config for quick inspection
    config_path = output_path.replace('.pkl', '_config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        config_data = projected_data["config"].copy()
        config_data["stats"] = {
            "total_cg_iterations": projected_data["projection_stats"]["total_cg_iterations"],
            "total_constraint_residual": projected_data["projection_stats"]["total_constraint_residual"]
        }
        json.dump(config_data, f, ensure_ascii=False, indent=2)
    
    # Print file size
    file_size = os.path.getsize(output_path) / 1024 / 1024
    print(f"✅ Saved: {output_path} ({file_size:.1f} MB)")
    print(f"📋 Config info: {config_path}")


def main():
    parser = argparse.ArgumentParser(description="Null-space projection - compute and save projected task vectors")
    
    # Base paths
    parser.add_argument("--base", type=str, 
                       default="/opt/data/private/hzhcode/huggingface/models/Qwen/Qwen2.5-7B")
    parser.add_argument("--instruct", type=str,
                       default="/opt/data/private/hzhcode/huggingface/models/Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--target", type=str,
                       default="/opt/data/private/hzhcode/huggingface/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    
    # Data & constraint params
    parser.add_argument("--texts_r", type=str, required=True, help="Path to JSON sample file")
    parser.add_argument("--max_samples_r", type=int, default=10, help="Max number of samples")
    parser.add_argument("--neigh_radius", type=int, default=5, help="Boundary neighborhood radius")
    
    # Layer & head config
    parser.add_argument("--layers_tail", type=int, default=2, help="Process the last N layers")
    parser.add_argument("--heads", type=str, default="all", help="Heads to process ('all' or comma-separated indices)")
    
    # Weights & solver params
    parser.add_argument("--lambda_ridge", type=float, default=1e-4, help="Ridge parameter (λ)")
    parser.add_argument("--cg_maxit", type=int, default=100, help="Max CG iterations")
    parser.add_argument("--cg_tol", type=float, default=1e-5, help="CG convergence tolerance")
    
    # Compute config
    parser.add_argument("--compute_precision", type=str, choices=["fp32", "fp64"], default="fp32",
                       help="Computation precision")
    
    # Merge types
    parser.add_argument("--merge_types", type=str, default="qk", 
                       help="Merge types: a combination of q/k/v/o/f")
    
    # QK params
    parser.add_argument("--q_rows_per_text", type=int, default=8, help="Q constraint rows per text")
    parser.add_argument("--k_rows_per_text", type=int, default=8, help="K constraint rows per text")
    parser.add_argument("--w_q", type=float, default=1.0, help="Weight for Q constraints")
    parser.add_argument("--w_k", type=float, default=1.0, help="Weight for K constraints")
    
    # VO params
    parser.add_argument("--v_rows_per_text", type=int, default=4, help="V constraint target positions per text")
    parser.add_argument("--o_rows_per_text", type=int, default=4, help="O constraint target positions per text")
    parser.add_argument("--w_v", type=float, default=1.0, help="Weight for V constraints")
    parser.add_argument("--w_o", type=float, default=1.0, help="Weight for O constraints")
    
    # FFN params
    parser.add_argument("--ffn_rows_per_text", type=int, default=4, help="FFN-Down constraint target positions per text")
    parser.add_argument("--readout_dirs", type=int, default=2, help="Number of output readout directions c per head/layer")
    parser.add_argument("--w_ffn", type=float, default=1.0, help="Weight for FFN-Down constraints")
    
    # Multi-device config
    parser.add_argument("--qk_device", type=str, default="auto",
                       help="Computation device for QK constraints")
    parser.add_argument("--vo_device", type=str, default="auto",
                       help="Computation device for VO constraints")
    parser.add_argument("--ffn_device", type=str, default="auto",
                       help="Computation device for FFN constraints")
    
    # Hook config
    parser.add_argument("--use_hooks", action="store_true", default=True,
                       help="Use hooks to capture exact layer internals (recommended)")
    parser.add_argument("--no_hooks", action="store_true",
                       help="Disable hooks and use the legacy extraction method")
    
    # Sequence length limit
    parser.add_argument("--max_seq_len", type=int, default=7168,
                       help="Max sequence length (BF16-optimized attention, default 7168; BF16 halves memory usage)")
    
    # Output config
    parser.add_argument("--output_file", type=str, required=True, 
                       help="Output file path (*.pkl)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Precision
    compute_dtype = torch.float64 if args.compute_precision == "fp64" else torch.float32
    
    print("🚀 Null-space projection - Stage 1")
    print("=" * 70)
    print(f"Base: {args.base}")
    print(f"Instruct: {args.instruct}")
    print(f"Target: {args.target}")
    print(f"Output file: {args.output_file}")
    print(f"Precision: {args.compute_precision.upper()}")
    print(f"Merge types: {args.merge_types.upper()}")
    
    # Hook method selection
    use_hooks = args.use_hooks and not args.no_hooks
    print(f"Feature extraction method: {'Hook-based (recommended)' if use_hooks else 'Legacy method'}")

    start_time = time.time()

    # Load models (on CPU)
    print("\n📥 Loading models on CPU...")
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

    print(f"📋 Run config:")
    print(f"  Layers: {selected_layers}")
    print(f"  Heads: {len(selected_heads)}/{n_heads}")

    # Read data
    texts_R = read_json_samples_robust(args.texts_r, tokenizer, args.max_samples_r)
    print(f"📊 # JSON samples: {len(texts_R)}")

    # Compute null-space projections
    print("\n🔬 Computing null-space projections...")
    projected_data = compute_nullspace_projections(
        model_base, model_instruct, model_target,
        texts_R, tokenizer,
        selected_layers, selected_heads,
        args.neigh_radius, args.lambda_ridge, args.cg_maxit, args.cg_tol,
        compute_dtype, args.merge_types,
        # QK
        args.q_rows_per_text, args.k_rows_per_text, args.w_q, args.w_k,
        # VO
        args.v_rows_per_text, args.o_rows_per_text, args.w_v, args.w_o,
        # FFN
        args.ffn_rows_per_text, args.w_ffn, args.readout_dirs, args.seed,
        # Devices
        args.qk_device, args.vo_device, args.ffn_device,
        # Hooks
        use_hooks,
        # Max seq len
        args.max_seq_len
    )

    # Save results
    end_time = time.time()
    
    # Attach runtime info
    projected_data["runtime_info"] = {
        "runtime_seconds": end_time - start_time,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "args": vars(args)
    }
    
    save_projected_task_vectors(projected_data, args.output_file)

    print(f"\n✅ Null-space projection finished! Elapsed: {end_time - start_time:.1f}s")
    print(f"📁 Output file: {args.output_file}")
    print(f"🚀 Next: use scaling_model_merge.py to apply different scaling factors for model merging")


if __name__ == "__main__":
    main()
