#!/usr/bin/env python3
"""
Three-Stage Model Merging Pipeline
==================================

Stage 1: Null-space Projection Computation (nullspace_projection_compute.py)
- Compute and save projected task vectors without applying scaling factor
- Output: Projected task vectors file (.pkl)

Stage 2: QP Optimization for Alpha Coefficients (qp_true_forward_fast.py)  
- Build instruction attention QP problem, optimize merging coefficients alpha
- Output: Optimized alpha coefficients file (.pt/.json)

Stage 3: Unified Model Merging (unified_model_merge.py)
- Apply alpha coefficients or scaling factor, merge task vectors into inference model
- Output: Complete merged model
"""

import os
import sys
import argparse
import subprocess
import time
import json
from pathlib import Path
from typing import Optional, Dict, Any, List


class ModelMergingPipeline:
    """Three-stage model merging pipeline"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.start_time = time.time()
        self.stage_timings = {}
        
    def log(self, message: str, stage: str = "PIPELINE"):
        """Unified logging output"""
        elapsed = time.time() - self.start_time
        print(f"[{elapsed:6.1f}s] [{stage}] {message}")
        
    def run_command(self, cmd: List[str], stage: str) -> bool:
        """Execute command and return success status"""
        self.log(f"Executing command: {' '.join(cmd)}", stage)
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            self.log(f"Command completed successfully", stage)
            return True
        except subprocess.CalledProcessError as e:
            self.log(f"Command failed: {e}", stage)
            self.log(f"Error output: {e.stderr}", stage)
            return False
            
    def stage1_nullspace_projection(self) -> bool:
        """Stage 1: Null-space projection computation"""
        stage_start = time.time()
        self.log("Starting Stage 1: Null-space projection computation", "STAGE1")
        
        # Build output path
        output_file = os.path.join(self.config["output_dir"], "projected_task_vectors.pkl")
        os.makedirs(self.config["output_dir"], exist_ok=True)
        
        # Build command
        cmd = [
            "python", "nullspace_projection_compute.py",
            "--base", self.config["base_model"],
            "--instruct", self.config["instruct_model"],
            "--target", self.config["target_model"],
            "--texts_r", self.config["data_file"],
            "--output_file", output_file,
            "--max_samples_r", str(self.config.get("max_samples", 10)),
            "--layers_tail", str(self.config.get("layers_tail", 2)),
            "--heads", self.config.get("heads", "all"),
            "--merge_types", self.config.get("merge_types", "qkvof"),
            "--lambda_ridge", str(self.config.get("lambda_ridge", 1e-4)),
            "--cg_maxit", str(self.config.get("cg_maxit", 100)),
            "--cg_tol", str(self.config.get("cg_tol", 1e-5)),
            "--compute_precision", self.config.get("compute_precision", "fp32"),
            "--qk_device", self.config.get("qk_device", "auto"),
            "--vo_device", self.config.get("vo_device", "auto"),
            "--ffn_device", self.config.get("ffn_device", "auto"),
            "--use_hooks"
        ]
        
        success = self.run_command(cmd, "STAGE1")
        
        stage_time = time.time() - stage_start
        self.stage_timings["stage1"] = stage_time
        
        if success:
            self.log(f"Stage 1 completed, time elapsed: {stage_time:.1f}s", "STAGE1")
            self.log(f"Projection results saved to: {output_file}", "STAGE1")
            self.config["projected_file"] = output_file
            return True
        else:
            self.log("Stage 1 failed", "STAGE1")
            return False
            
    def stage2_qp_optimization(self) -> bool:
        """Stage 2: QP optimization for alpha coefficients"""
        stage_start = time.time()
        self.log("Starting Stage 2: QP optimization for alpha coefficients", "STAGE2")
        
        if "projected_file" not in self.config:
            self.log("Error: Projection file from Stage 1 not found", "STAGE2")
            return False
            
        # Build output path
        qp_output_dir = os.path.join(self.config["output_dir"], "qp_optimization")
        os.makedirs(qp_output_dir, exist_ok=True)
        
        # Build command
        cmd = [
            "python", "qp_true_forward_fast.py",
            "--projected_file", self.config["projected_file"],
            "--base_model", self.config["target_model"],
            "--json_data", self.config["data_file"],
            "--layers", self.config.get("heads", "all"),
            "--heads", self.config.get("heads", "all"),
            "--prior_scalar", str(self.config.get("prior_scalar", 1.0)),
            "--l2_prior", str(self.config.get("l2_prior", 0.1)),
            "--l1", str(self.config.get("l1", 0.0)),
            "--box_lo", str(self.config.get("box_lo", 0.0)),
            "--box_hi", str(self.config.get("box_hi", 1.5)),
            "--device", self.config.get("qk_device", "cuda:0"),
            "--out", qp_output_dir,
            "--qp_variant", self.config.get("qp_variant", "two_pass"),
            "--verbose"
        ]
        
        # Add optional parameters
        if self.config.get("decouple_qk", False):
            cmd.append("--decouple_qk")
        if self.config.get("save_model", False):
            cmd.append("--save_model")
            
        success = self.run_command(cmd, "STAGE2")
        
        stage_time = time.time() - stage_start
        self.stage_timings["stage2"] = stage_time
        
        if success:
            self.log(f"Stage 2 completed, time elapsed: {stage_time:.1f}s", "STAGE2")
            
            # Find alpha coefficient files
            alpha_files = [
                os.path.join(qp_output_dir, "alpha_true_forward_two_pass.pt"),
                os.path.join(qp_output_dir, "alpha_true_forward_anchor_only.pt"),
                os.path.join(qp_output_dir, "alpha_true_forward_post_only.pt"),
                os.path.join(qp_output_dir, "alpha_true_forward_two_pass.json"),
                os.path.join(qp_output_dir, "alpha_true_forward_anchor_only.json"),
                os.path.join(qp_output_dir, "alpha_true_forward_post_only.json"),
            ]
            
            alpha_file = None
            for f in alpha_files:
                if os.path.exists(f):
                    alpha_file = f
                    break
                    
            if alpha_file:
                self.log(f"Alpha coefficient file: {alpha_file}", "STAGE2")
                self.config["alpha_file"] = alpha_file
            else:
                self.log("Warning: Alpha coefficient file not found, will use scaling factor mode", "STAGE2")
                
            return True
        else:
            self.log("Stage 2 failed", "STAGE2")
            return False
            
    def stage3_unified_merge(self) -> bool:
        """Stage 3: Unified model merging"""
        stage_start = time.time()
        self.log("Starting Stage 3: Unified model merging", "STAGE3")
        
        if "projected_file" not in self.config:
            self.log("Error: Projection file from Stage 1 not found", "STAGE3")
            return False
            
        # Build output path
        merge_output_dir = os.path.join(self.config["output_dir"], "unified_model_merge")
        os.makedirs(merge_output_dir, exist_ok=True)
        
        # Build command
        cmd = [
            "python", "unified_model_merge.py",
            "--projected_file", self.config["projected_file"],
            "--base_model", self.config["target_model"],
            "--output_dir", merge_output_dir,
            "--model_name", self.config.get("model_name", "merged_model"),
            "--verbose"
        ]
        
        # Add alpha file or scaling factor
        if "alpha_file" in self.config:
            cmd.extend(["--alpha_file", self.config["alpha_file"]])
            self.log(f"Using Alpha weighting mode: {self.config['alpha_file']}", "STAGE3")
        
        if "scaling_factor" in self.config:
            cmd.extend(["--scaling_factor", str(self.config["scaling_factor"])])
            self.log(f"Using Scaling Factor: {self.config['scaling_factor']}", "STAGE3")
        
        if "alpha_file" not in self.config and "scaling_factor" not in self.config:
            # Default to scaling factor = 1.0
            cmd.extend(["--scaling_factor", "1.0"])
            self.log("Using default Scaling Factor: 1.0", "STAGE3")
            
        success = self.run_command(cmd, "STAGE3")
        
        stage_time = time.time() - stage_start
        self.stage_timings["stage3"] = stage_time
        
        if success:
            self.log(f"Stage 3 completed, time elapsed: {stage_time:.1f}s", "STAGE3")
            
            # Determine final model path
            model_name = self.config.get("model_name", "merged_model")
            final_model_path = os.path.join(merge_output_dir, model_name)
            
            if os.path.exists(final_model_path):
                self.log(f"Merged model saved to: {final_model_path}", "STAGE3")
                self.config["final_model"] = final_model_path
            else:
                self.log("Warning: Final merged model not found", "STAGE3")
                
            return True
        else:
            self.log("Stage 3 failed", "STAGE3")
            return False
            
    def run_pipeline(self, stages: List[int] = None) -> bool:
        """Run complete pipeline or specified stages"""
        if stages is None:
            stages = [1, 2, 3]
            
        self.log("Starting three-stage model merging pipeline")
        self.log(f"Configuration: {json.dumps(self.config, indent=2, default=str)}")
        
        total_start = time.time()
        
        # Stage 1: Null-space projection computation
        if 1 in stages:
            if not self.stage1_nullspace_projection():
                return False
                
        # Stage 2: QP optimization for alpha coefficients
        if 2 in stages:
            if not self.stage2_qp_optimization():
                return False
                
        # Stage 3: Unified model merging
        if 3 in stages:
            if not self.stage3_unified_merge():
                return False
                
        total_time = time.time() - total_start
        
        # Output summary
        self.log("=" * 70)
        self.log("Three-stage model merging pipeline completed!")
        self.log(f"Total time elapsed: {total_time:.1f}s")
        
        if self.stage_timings:
            self.log("Stage timing breakdown:")
            for stage, timing in self.stage_timings.items():
                self.log(f"  {stage}: {timing:.1f}s")
                
        if "final_model" in self.config:
            self.log(f"Final model: {self.config['final_model']}")
            
        # Save runtime configuration
        config_file = os.path.join(self.config["output_dir"], "pipeline_config.json")
        with open(config_file, 'w', encoding='utf-8') as f:
            config_to_save = self.config.copy()
            config_to_save["stage_timings"] = self.stage_timings
            config_to_save["total_time"] = total_time
            config_to_save["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
            json.dump(config_to_save, f, ensure_ascii=False, indent=2, default=str)
            
        self.log(f"Configuration saved to: {config_file}")
        self.log("=" * 70)
        
        return True


def main():
    parser = argparse.ArgumentParser(description="Three-stage model merging pipeline")
    
    # Basic configuration
    parser.add_argument("--base_model", type=str, required=True,
                       help="Base model path")
    parser.add_argument("--instruct_model", type=str, required=True,
                       help="Instruction model path")  
    parser.add_argument("--target_model", type=str, required=True,
                       help="Target model path")
    parser.add_argument("--data_file", type=str, required=True,
                       help="Training data file path (.json/.jsonl)")
    parser.add_argument("--output_dir", type=str, default="./pipeline_output",
                       help="Output directory")
                       
    # Model configuration
    parser.add_argument("--max_samples", type=int, default=10,
                       help="Maximum number of samples")
    parser.add_argument("--layers_tail", type=int, default=2,
                       help="Process last N layers")
    parser.add_argument("--heads", type=str, default="all",
                       help="Heads to process ('all' or comma-separated indices)")
    parser.add_argument("--merge_types", type=str, default="qkvof",
                       help="Merge types: combination of q/k/v/o/f")
                       
    # Computation configuration
    parser.add_argument("--compute_precision", type=str, choices=["fp32", "fp64"], 
                       default="fp32", help="Computation precision")
    parser.add_argument("--lambda_ridge", type=float, default=1e-4,
                       help="Ridge regression parameter")
    parser.add_argument("--cg_maxit", type=int, default=100,
                       help="CG maximum iterations")
    parser.add_argument("--cg_tol", type=float, default=1e-5,
                       help="CG convergence tolerance")
                       
    # Device configuration
    parser.add_argument("--qk_device", type=str, default="auto",
                       help="QK constraint computation device")
    parser.add_argument("--vo_device", type=str, default="auto",
                       help="VO constraint computation device")
    parser.add_argument("--ffn_device", type=str, default="auto",
                       help="FFN constraint computation device")
                       
    # QP optimization configuration
    parser.add_argument("--prior_scalar", type=float, default=1.0,
                       help="Alpha prior value")
    parser.add_argument("--l2_prior", type=float, default=0.1,
                       help="L2 regularization parameter")
    parser.add_argument("--l1", type=float, default=0.0,
                       help="L1 regularization parameter")
    parser.add_argument("--box_lo", type=float, default=0.0,
                       help="Box constraint lower bound")
    parser.add_argument("--box_hi", type=float, default=1.5,
                       help="Box constraint upper bound")
    parser.add_argument("--qp_variant", type=str, 
                       choices=["two_pass", "anchor_only", "post_only"],
                       default="two_pass", help="QP construction method")
    parser.add_argument("--decouple_qk", action="store_true",
                       help="Decouple Q and K alpha coefficients")
    parser.add_argument("--save_model", action="store_true",
                       help="Save QP optimized model")
                       
    # Merge configuration
    parser.add_argument("--scaling_factor", type=float, default=None,
                       help="Scaling factor (if not using alpha coefficients)")
    parser.add_argument("--model_name", type=str, default="merged_model",
                       help="Merged model name")
                       
    # Pipeline control
    parser.add_argument("--stages", type=str, default="1,2,3",
                       help="Stages to execute (comma-separated, e.g. '1,2' or '3')")
    parser.add_argument("--projected_file", type=str, default=None,
                       help="Existing projection file (use when skipping stage 1)")
    parser.add_argument("--alpha_file", type=str, default=None,
                       help="Existing alpha file (use when skipping stage 2)")
                       
    args = parser.parse_args()
    
    # Parse execution stages
    stages = [int(s.strip()) for s in args.stages.split(",")]
    
    # Build configuration
    config = {
        "base_model": args.base_model,
        "instruct_model": args.instruct_model,
        "target_model": args.target_model,
        "data_file": args.data_file,
        "output_dir": args.output_dir,
        "max_samples": args.max_samples,
        "layers_tail": args.layers_tail,
        "heads": args.heads,
        "merge_types": args.merge_types,
        "compute_precision": args.compute_precision,
        "lambda_ridge": args.lambda_ridge,
        "cg_maxit": args.cg_maxit,
        "cg_tol": args.cg_tol,
        "qk_device": args.qk_device,
        "vo_device": args.vo_device,
        "ffn_device": args.ffn_device,
        "prior_scalar": args.prior_scalar,
        "l2_prior": args.l2_prior,
        "l1": args.l1,
        "box_lo": args.box_lo,
        "box_hi": args.box_hi,
        "qp_variant": args.qp_variant,
        "decouple_qk": args.decouple_qk,
        "save_model": args.save_model,
        "scaling_factor": args.scaling_factor,
        "model_name": args.model_name,
    }
    
    # Handle pre-existing files
    if args.projected_file:
        config["projected_file"] = args.projected_file
    if args.alpha_file:
        config["alpha_file"] = args.alpha_file
        
    # Validate inputs
    if not os.path.exists(args.data_file):
        print(f"Error: Data file does not exist: {args.data_file}")
        sys.exit(1)
        
    if 1 not in stages and "projected_file" not in config:
        print("Error: Must provide --projected_file when skipping stage 1")
        sys.exit(1)
        
    if 2 not in stages and 3 in stages and "alpha_file" not in config and "scaling_factor" not in config:
        print("Error: Must provide --alpha_file or --scaling_factor when skipping stage 2")
        sys.exit(1)
    
    # Create pipeline and run
    pipeline = ModelMergingPipeline(config)
    success = pipeline.run_pipeline(stages)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
