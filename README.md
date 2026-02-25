# RAIN-Merging: A Gradient-Free Method to Enhance Instruction Following in Large Reasoning Models with Preserved Thinking Format


## Abstract
Large reasoning models (LRMs) excel at a long chain of reasoning but often fail to faithfully follow instructions regarding output format, constraints, or specific requirements. We investigate whether this gap can be closed by integrating an instruction-tuned model (ITM) into an LRM. Analyzing their differences in parameter space, namely task vectors, we find that their principal subspaces are nearly orthogonal across key modules, suggesting a lightweight merging with minimal interference. However, we also demonstrate that naïve merges are fragile because they overlook the output format mismatch between LRMs (with explicit `thinking` and `response` segments) and ITMs (answers-only). We introduce **RAIN-Merging** (Reasoning-Aware Instruction-attention guided Null-space projection Merging), a gradient-free method that integrates instruction following while preserving thinking format and reasoning performance. First, with a small reasoning calibration set, we project the ITM task vector onto the null space of forward features at thinking special tokens, which preserves the LRM's structured reasoning mechanisms. Second, using a small instruction calibration set, we estimate instruction attention to derive module-specific scaling that amplifies instruction-relevant components and suppresses leakage. Across four instruction-following benchmarks and nine reasoning & general capability benchmarks, RAIN-Merging substantially improves instruction adherence while maintaining reasoning quality. The gains are consistent across model scales and architectures, translating to improved performance in agentic scenarios.

## 🚀 Overview


<table align="center">
  <tr>
    <td align="center"> 
      <img src="asset/RAIN-Merge-overview.png" alt="Teaser" style="width: 1000px;"/> 
      <br>
      <em style="font-size: 18px;"><strong style="font-size: 18px;"><strong>Overview of RAIN-Merging</strong></em>
    </td>
  </tr>
</table>

<table align="center">
  <tr>
    <td align="center"> 
      <img src="asset/RAIN-Merge-Method.png" alt="Teaser" style="width: 1000px;"/> 
      <br>
      <em style="font-size: 18px;"><strong style="font-size: 18px;"><strong>Two stages of our RAIN-Merging pipeline</strong></em>
    </td>
  </tr>
</table>


## 📁 Project Structure

```
RAIN-Merging/
├── scripts/                          # Execution scripts
│   ├── run_stage1.sh                 # Stage 1: Reasoning-aware Null-space Projection
│   ├── run_stage2.sh                 # Stage 2: Instruction-attention Guided Merging Coefficients
│   └── run_stage3.sh                 # Stage 3: Model merging
├── nullspace_projection_compute.py   # Stage 1 implementation
├── qp_true_forward_fast.py           # Stage 2 implementation
├── unified_model_merge.py            # Stage 3 implementation
├── pipeline.py                       # End-to-end pipeline
├── data/                             # Calibration set
├── requirements.txt                  # Dependencies
└── README.md                         # This file
```

## 🛠 Installation

<!-- 1. **Clone the repository:**
```bash
git clone https://github.com/your-username/RAIN-Merging.git
cd RAIN-Merging
``` -->

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Optional optimizations:**
```bash
# For Flash Attention (recommended)
pip install flash-attn

# For quantization support
pip install bitsandbytes
```

## 📋 Quick Start

### Three-Stage Pipeline

#### Stage 1: Null-space Projection
Compute null-space projections for selective parameter modification:

```bash
./scripts/run_stage1.sh \\
    /path/to/base_model \\
    /path/to/instruct_model \\
    /path/to/target_model \\
    ./data/reasoning_calibration_set.json \\
    ./stage1_output
```

#### Stage 2: QP Optimization
Optimize merging coefficients using quadratic programming:

```bash
./scripts/run_stage2.sh \\
    /path/to/target_model \\
    ./data/reasoning_calibration_set.json \\
    ./stage1_output/projected_task_vectors.pkl \\
    ./stage2_output
```

#### Stage 3: Model Merging
Apply projections and coefficients to create the final merged model:

```bash
./scripts/run_stage3.sh \\
    /path/to/target_model \\
    ./stage1_output/projected_task_vectors.pkl \\
    ./stage2_output/alpha_coefficients.pt \\
    ./final_merged_model
```

### One-Command Pipeline
For convenience, use the unified pipeline:

```bash
python pipeline.py \\
    --base_model /path/to/base_model \\
    --instruct_model /path/to/instruct_model \\
    --target_model /path/to/target_model \\
    --data ./data/instruction_calibration_set.jsonl \\
    --output ./merged_model_output
```

## ⚙️ Configuration

### Environment Variables

**Stage 1 Options:**
```bash
export MAX_SAMPLES=500              # Number of training samples
export LAYERS_TAIL=28               # Process last N layers
export COMPUTE_PRECISION=fp64      # Computation precision
export MAX_SEQ_LEN=16384           # Maximum sequence length
```

**Stage 2 Options:**
```bash
export BOX_LO=0.0                 # Lower bound for alpha
export BOX_HI=1.0                 # Upper bound for alpha
```

**Stage 3 Options:**
```bash
export MODEL_NAME=merged_model     # Output model name
export SCALING_FACTOR=1.0         # Global scaling factor
```

