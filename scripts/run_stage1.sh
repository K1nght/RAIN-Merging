#!/bin/bash

# Activate conda environment
source activate merge

# Stage 1: Null-space projection computation
# Function: Compute and save projected task vectors, without applying scaling factor
# Output: projected_task_vectors.pkl

# Color definitions
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
DEFAULT_BASE="Qwen/Qwen2.5-7B"
DEFAULT_INSTRUCT="Qwen/Qwen2.5-7B-Instruct"
DEFAULT_TARGET="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
DEFAULT_DATA="./data/reasoning_calibration_set.json"
DEFAULT_OUTPUT="./stage1_output_$(date +%Y%m%d_%H%M%S)"

# Parameter settings
BASE_MODEL=${1:-$DEFAULT_BASE}
INSTRUCT_MODEL=${2:-$DEFAULT_INSTRUCT}
TARGET_MODEL=${3:-$DEFAULT_TARGET}
DATA_FILE=${4:-$DEFAULT_DATA}
OUTPUT_DIR=${5:-$DEFAULT_OUTPUT}

# Configurable parameters (via environment variables)
MAX_SAMPLES=${MAX_SAMPLES:-1000}
LAYERS_TAIL=${LAYERS_TAIL:-27}
HEADS=${HEADS:-"all"}
MERGE_TYPES=${MERGE_TYPES:-"qkvof"}
COMPUTE_PRECISION=${COMPUTE_PRECISION:-"fp32"}
LAMBDA_RIDGE=${LAMBDA_RIDGE:-1e-4}
CG_MAXIT=${CG_MAXIT:-100}
CG_TOL=${CG_TOL:-1e-5}

# Device configuration
QK_DEVICE=${QK_DEVICE:-"auto"}
VO_DEVICE=${VO_DEVICE:-"auto"}
FFN_DEVICE=${FFN_DEVICE:-"auto"}

# Constraint parameter configuration
Q_ROWS=${Q_ROWS:-8}
K_ROWS=${K_ROWS:-8}
V_ROWS=${V_ROWS:-4}
O_ROWS=${O_ROWS:-4}
FFN_ROWS=${FFN_ROWS:-4}
W_Q=${W_Q:-1.0}
W_K=${W_K:-1.0}
W_V=${W_V:-1.0}
W_O=${W_O:-1.0}
W_FFN=${W_FFN:-1.0}
READOUT_DIRS=${READOUT_DIRS:-2}

# Sequence length limit (BF16 optimized, attention matrix uses BF16 to save 50% memory)
MAX_SEQ_LEN=${MAX_SEQ_LEN:-7168}

function show_help() {
    echo -e "${GREEN}Stage 1: Null-space Projection Computation${NC}"
    echo ""
    echo "Usage: $0 [base_model] [instruct_model] [target_model] [data_file] [output_dir]"
    echo ""
    echo -e "${YELLOW}Positional Arguments:${NC}"
    echo "  base_model     Base model path (default: $DEFAULT_BASE)"
    echo "  instruct_model Instruction model path (default: $DEFAULT_INSTRUCT)"
    echo "  target_model   Target model path (default: $DEFAULT_TARGET)"
    echo "  data_file      Training data file (default: $DEFAULT_DATA)"
    echo "  output_dir     Output directory (default: $DEFAULT_OUTPUT)"
    echo ""
    echo -e "${YELLOW}Environment Variable Configuration:${NC}"
    echo "  MAX_SAMPLES        Maximum number of samples (default: 1000)"
    echo "  LAYERS_TAIL        Process last N layers (default: 27)"
    echo "  HEADS              Attention heads to process (default: all)"
    echo "  MERGE_TYPES        Merge types (default: qkvof)"
    echo "  COMPUTE_PRECISION  Compute precision (default: fp32)"
    echo "  LAMBDA_RIDGE       Ridge regression parameter (default: 1e-4)"
    echo "  CG_MAXIT           CG maximum iterations (default: 100)"
    echo "  CG_TOL             CG convergence tolerance (default: 1e-5)"
    echo ""
    echo -e "${YELLOW}Device Configuration:${NC}"
    echo "  QK_DEVICE          QK compute device (default: auto)"
    echo "  VO_DEVICE          VO compute device (default: auto)"
    echo "  FFN_DEVICE         FFN compute device (default: auto)"
    echo ""
    echo -e "${YELLOW}Sequence Length Control:${NC}"
    echo "  MAX_SEQ_LEN        Maximum sequence length limit (default: 7168, BF16 optimized, saves 50% memory for attention matrices)"
    echo ""
    echo -e "${YELLOW}Examples:${NC}"
    echo "  # Basic usage"
    echo "  $0 /path/to/base /path/to/instruct /path/to/target"
    echo ""
    echo "  # High precision computation"
    echo "  COMPUTE_PRECISION=fp64 LAMBDA_RIDGE=1e-5 $0"
    echo ""
    echo "  # Multi-GPU configuration"
    echo "  QK_DEVICE=cuda:0 VO_DEVICE=cuda:1 FFN_DEVICE=cuda:2 $0"
    echo ""
    echo "  # More samples and layers"
    echo "  MAX_SAMPLES=20 LAYERS_TAIL=4 $0"
}

# Check for help argument
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    show_help
    exit 0
fi

echo -e "${BLUE}=======================================================================${NC}"
echo -e "${BLUE}                    Stage 1: Null-space Projection Computation${NC}"
echo -e "${BLUE}=======================================================================${NC}"
echo -e "${GREEN}📁 Base Model: ${NC}$BASE_MODEL"
echo -e "${GREEN}📁 Instruction Model: ${NC}$INSTRUCT_MODEL"  
echo -e "${GREEN}📁 Target Model: ${NC}$TARGET_MODEL"
echo -e "${GREEN}📁 Training Data: ${NC}$DATA_FILE"
echo -e "${GREEN}📁 Output Directory: ${NC}$OUTPUT_DIR"
echo ""
echo -e "${YELLOW}Configuration Parameters:${NC}"
echo "  Max samples: $MAX_SAMPLES"
echo "  Layers: $LAYERS_TAIL"
echo "  Heads: $HEADS"
echo "  Merge types: $MERGE_TYPES"
echo "  Compute precision: $COMPUTE_PRECISION"
echo "  Devices: QK=$QK_DEVICE, VO=$VO_DEVICE, FFN=$FFN_DEVICE"
echo "  Sequence length limit: $MAX_SEQ_LEN tokens (BF16 optimized, saves 50% memory for attention matrices)"
echo -e "${BLUE}=======================================================================${NC}"

# Check required files
if [[ ! -f "$DATA_FILE" ]]; then
    echo -e "${RED}❌ Error: Data file does not exist: $DATA_FILE${NC}"
    echo -e "${YELLOW}Available data files:${NC}"
    ls -la data/ 2>/dev/null || echo "data directory does not exist"
    exit 1
fi

if [[ ! -f "nullspace_projection_compute.py" ]]; then
    echo -e "${RED}❌ Error: nullspace_projection_compute.py does not exist${NC}"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Build output file path
OUTPUT_FILE="$OUTPUT_DIR/projected_task_vectors.pkl"

echo -e "\n${BLUE}🔄 Starting Stage 1: Null-space Projection Computation...${NC}"

# Record start time
START_TIME=$(date +%s)

# Execute projection computation
python nullspace_projection_compute.py \
    --base "$BASE_MODEL" \
    --instruct "$INSTRUCT_MODEL" \
    --target "$TARGET_MODEL" \
    --texts_r "$DATA_FILE" \
    --output_file "$OUTPUT_FILE" \
    --max_samples_r $MAX_SAMPLES \
    --layers_tail $LAYERS_TAIL \
    --heads "$HEADS" \
    --merge_types "$MERGE_TYPES" \
    --compute_precision "$COMPUTE_PRECISION" \
    --lambda_ridge $LAMBDA_RIDGE \
    --cg_maxit $CG_MAXIT \
    --cg_tol $CG_TOL \
    --q_rows_per_text $Q_ROWS \
    --k_rows_per_text $K_ROWS \
    --v_rows_per_text $V_ROWS \
    --o_rows_per_text $O_ROWS \
    --ffn_rows_per_text $FFN_ROWS \
    --w_q $W_Q \
    --w_k $W_K \
    --w_v $W_V \
    --w_o $W_O \
    --w_ffn $W_FFN \
    --readout_dirs $READOUT_DIRS \
    --qk_device "$QK_DEVICE" \
    --vo_device "$VO_DEVICE" \
    --ffn_device "$FFN_DEVICE" \
    --max_seq_len $MAX_SEQ_LEN \
    --use_hooks

EXIT_CODE=$?
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo -e "${BLUE}=======================================================================${NC}"

if [[ $EXIT_CODE -eq 0 ]]; then
    echo -e "${GREEN}✅ Stage 1 completed successfully! Elapsed: ${DURATION}s${NC}"
    echo -e "${GREEN}📁 Output directory: $OUTPUT_DIR${NC}"
    echo -e "${GREEN}📄 Projection file: $OUTPUT_FILE${NC}"
    echo ""
    echo -e "${YELLOW}📊 Output files:${NC}"
    ls -la "$OUTPUT_DIR"
    echo ""
    echo -e "${YELLOW}🚀 Next step: Run Stage 2 (QP optimization)${NC}"
    echo "  ./run_stage2.sh \"$TARGET_MODEL\" \"$DATA_FILE\" \"$OUTPUT_FILE\" \"./stage2_output\""
else
    echo -e "${RED}❌ Stage 1 failed with exit code: $EXIT_CODE${NC}"
    echo -e "${RED}Please check the error messages and retry${NC}"
fi

echo -e "${BLUE}=======================================================================${NC}"

exit $EXIT_CODE
