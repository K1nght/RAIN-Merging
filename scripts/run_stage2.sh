#!/bin/bash

# Activate conda environment
source activate merge

# Stage 2: QP optimization of alpha coefficients
# Function: Optimize merge coefficients alpha based on projected task vectors
# Input: projected_task_vectors.pkl (from Stage 1)
# Output: alpha coefficient file and QP results

# Color definitions
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
DEFAULT_TARGET="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
DEFAULT_DATA="./data/instruction_calibration_set.jsonl"
DEFAULT_PROJECTED=""
DEFAULT_OUTPUT="./stage2_output_$(date +%Y%m%d_%H%M%S)"

# Parameter settings
TARGET_MODEL=${1:-$DEFAULT_TARGET}
DATA_FILE=${2:-$DEFAULT_DATA}
PROJECTED_FILE=${3:-$DEFAULT_PROJECTED}
OUTPUT_DIR=${4:-$DEFAULT_OUTPUT}

# QP optimization parameters (via environment variables)
QP_VARIANT=${QP_VARIANT:-"two_pass"}
PRIOR_SCALAR=${PRIOR_SCALAR:-1.0}
L2_PRIOR=${L2_PRIOR:-0.1}
L1_REG=${L1_REG:-0.0}
BOX_LO=${BOX_LO:-0.0}
BOX_HI=${BOX_HI:-1.5}
DEVICE=${DEVICE:-"cuda:0"}

# QP construction parameters
H_LAMBDA=${H_LAMBDA:-1.0}
H_MU=${H_MU:-1.0}
RHO_DU=${RHO_DU:-0.5}
KAPPA_A=${KAPPA_A:-1.0}
KAPPA_U=${KAPPA_U:-1.0}

# Other options
DECOUPLE_QK=${DECOUPLE_QK:-false}
SAVE_MODEL=${SAVE_MODEL:-false}
LAYERS=${LAYERS:-"all"}
HEADS=${HEADS:-"all"}

function show_help() {
    echo -e "${GREEN}Stage 2: QP Optimization of Alpha Coefficients${NC}"
    echo ""
    echo "Usage: $0 [target_model] [data_file] [projected_file] [output_dir]"
    echo ""
    echo -e "${YELLOW}Positional Arguments:${NC}"
    echo "  target_model   Target model path (default: $DEFAULT_TARGET)"
    echo "  data_file      Training data file (default: $DEFAULT_DATA)"
    echo "  projected_file Projection file (Stage 1 output) (required)"
    echo "  output_dir     Output directory (default: $DEFAULT_OUTPUT)"
    echo ""
    echo -e "${YELLOW}Environment Variable Configuration:${NC}"
    echo "  QP_VARIANT     QP construction mode: two_pass/anchor_only/post_only (default: two_pass)"
    echo "  PRIOR_SCALAR   Alpha prior value (default: 1.0)"
    echo "  L2_PRIOR       L2 regularization parameter (default: 0.1)"
    echo "  L1_REG         L1 regularization parameter (default: 0.0)"
    echo "  BOX_LO         Box constraint lower bound (default: 0.0)"
    echo "  BOX_HI         Box constraint upper bound (default: 1.5)"
    echo "  DEVICE         Compute device (default: cuda:0)"
    echo ""
    echo -e "${YELLOW}Advanced Parameters:${NC}"
    echo "  H_LAMBDA       H matrix diagonal constant (default: 1.0)"
    echo "  H_MU           Posterior leakage weight (default: 1.0)"
    echo "  RHO_DU         Leakage change penalty (default: 0.5)"
    echo "  KAPPA_A        Alignment score scaling (default: 1.0)"
    echo "  KAPPA_U        Leakage score scaling (default: 1.0)"
    echo ""
    echo -e "${YELLOW}Boolean Options:${NC}"
    echo "  DECOUPLE_QK    Decouple Q/K coefficients: true/false (default: false)"
    echo "  SAVE_MODEL     Save QP-optimized model: true/false (default: false)"
    echo ""
    echo -e "${YELLOW}Examples:${NC}"
    echo "  # Basic usage"
    echo "  $0 /path/to/target ./data/instruction_calibration_set.jsonl ./stage1_output/projected_task_vectors.pkl"
    echo ""
    echo "  # anchor_only mode"
    echo "  QP_VARIANT=anchor_only $0 /path/to/target ./data/instruction_calibration_set.jsonl ./projected.pkl"
    echo ""
    echo "  # Decouple Q/K and save model"
    echo "  DECOUPLE_QK=true SAVE_MODEL=true $0 /path/to/target ./data/instruction_calibration_set.jsonl ./projected.pkl"
    echo ""
    echo "  # Custom QP parameters"
    echo "  L2_PRIOR=0.2 BOX_HI=2.0 H_LAMBDA=0.5 $0 /path/to/target ./data/instruction_calibration_set.jsonl ./projected.pkl"
}

# Check for help argument
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    show_help
    exit 0
fi

echo -e "${BLUE}=======================================================================${NC}"
echo -e "${BLUE}                    Stage 2: QP Optimization of Alpha Coefficients${NC}"
echo -e "${BLUE}=======================================================================${NC}"
echo -e "${GREEN}📁 Target Model: ${NC}$TARGET_MODEL"
echo -e "${GREEN}📁 Training Data: ${NC}$DATA_FILE"
echo -e "${GREEN}📁 Projection File: ${NC}$PROJECTED_FILE"
echo -e "${GREEN}📁 Output Directory: ${NC}$OUTPUT_DIR"
echo ""
echo -e "${YELLOW}QP Configuration:${NC}"
echo "  QP variant: $QP_VARIANT"
echo "  Alpha prior: $PRIOR_SCALAR"
echo "  L2 regularization: $L2_PRIOR"
echo "  L1 regularization: $L1_REG"
echo "  Box constraint: [$BOX_LO, $BOX_HI]"
echo "  Compute device: $DEVICE"
echo "  Decouple Q/K: $DECOUPLE_QK"
echo "  Save model: $SAVE_MODEL"
echo -e "${BLUE}=======================================================================${NC}"

# Check required files
if [[ ! -f "$DATA_FILE" ]]; then
    echo -e "${RED}❌ Error: Data file does not exist: $DATA_FILE${NC}"
    exit 1
fi

if [[ ! -f "$PROJECTED_FILE" ]]; then
    echo -e "${RED}❌ Error: Projection file does not exist: $PROJECTED_FILE${NC}"
    echo -e "${YELLOW}Please run Stage 1 first to generate the projection file${NC}"
    exit 1
fi

if [[ ! -f "qp_true_forward_fast.py" ]]; then
    echo -e "${RED}❌ Error: qp_true_forward_fast.py does not exist${NC}"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo -e "\n${BLUE}🔄 Starting Stage 2: QP Optimization of Alpha Coefficients...${NC}"

# Record start time
START_TIME=$(date +%s)

# Build command
CMD="python qp_true_forward_fast.py"
CMD="$CMD --projected_file \"$PROJECTED_FILE\""
CMD="$CMD --base_model \"$TARGET_MODEL\""
CMD="$CMD --json_data \"$DATA_FILE\""
CMD="$CMD --layers \"$LAYERS\""
CMD="$CMD --heads \"$HEADS\""
CMD="$CMD --prior_scalar $PRIOR_SCALAR"
CMD="$CMD --l2_prior $L2_PRIOR"
CMD="$CMD --l1 $L1_REG"
CMD="$CMD --box_lo $BOX_LO"
CMD="$CMD --box_hi $BOX_HI"
CMD="$CMD --device \"$DEVICE\""
CMD="$CMD --out \"$OUTPUT_DIR\""
CMD="$CMD --qp_variant \"$QP_VARIANT\""
CMD="$CMD --verbose"

# Add QP construction parameters
if [[ "$QP_VARIANT" == "two_pass" ]]; then
    CMD="$CMD --H_lambda $H_LAMBDA"
    CMD="$CMD --H_mu $H_MU"
    CMD="$CMD --rho_du $RHO_DU"
fi

if [[ "$QP_VARIANT" == "anchor_only" || "$QP_VARIANT" == "post_only" ]]; then
    CMD="$CMD --H_lambda $H_LAMBDA"
    CMD="$CMD --H_mu $H_MU"
    CMD="$CMD --rho_du $RHO_DU"
    CMD="$CMD --kappa_a $KAPPA_A"
    CMD="$CMD --kappa_u $KAPPA_U"
fi

# Add boolean options
if [[ "$DECOUPLE_QK" == "true" ]]; then
    CMD="$CMD --decouple_qk"
fi

if [[ "$SAVE_MODEL" == "true" ]]; then
    CMD="$CMD --save_model"
fi

echo -e "${YELLOW}Executing command:${NC}"
echo "$CMD"
echo ""

# Execute QP optimization
eval $CMD

EXIT_CODE=$?
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo -e "${BLUE}=======================================================================${NC}"

if [[ $EXIT_CODE -eq 0 ]]; then
    echo -e "${GREEN}✅ Stage 2 completed successfully! Elapsed: ${DURATION}s${NC}"
    echo -e "${GREEN}📁 Output directory: $OUTPUT_DIR${NC}"
    echo ""
    echo -e "${YELLOW}📊 Output files:${NC}"
    ls -la "$OUTPUT_DIR"
    echo ""
    
    # Find alpha file
    ALPHA_FILES=(
        "$OUTPUT_DIR/alpha_true_forward_two_pass.pt"
        "$OUTPUT_DIR/alpha_true_forward_anchor_only.pt"
        "$OUTPUT_DIR/alpha_true_forward_post_only.pt"
        "$OUTPUT_DIR/alpha_true_forward_two_pass.json"
        "$OUTPUT_DIR/alpha_true_forward_anchor_only.json"
        "$OUTPUT_DIR/alpha_true_forward_post_only.json"
    )
    
    ALPHA_FILE=""
    for f in "${ALPHA_FILES[@]}"; do
        if [[ -f "$f" ]]; then
            ALPHA_FILE="$f"
            break
        fi
    done
    
    if [[ -n "$ALPHA_FILE" ]]; then
        echo -e "${GREEN}🎯 Alpha coefficient file: $ALPHA_FILE${NC}"
        echo ""
        echo -e "${YELLOW}🚀 Next step: Run Stage 3 (model merging)${NC}"
        echo "  ./run_stage3.sh \"$TARGET_MODEL\" \"$PROJECTED_FILE\" \"$ALPHA_FILE\" \"./stage3_output\""
    else
        echo -e "${YELLOW}⚠️  No alpha coefficient file found; you can run Stage 3 in scaling factor mode${NC}"
        echo "  ./run_stage3.sh \"$TARGET_MODEL\" \"$PROJECTED_FILE\" \"\" \"./stage3_output\""
    fi
else
    echo -e "${RED}❌ Stage 2 failed with exit code: $EXIT_CODE${NC}"
    echo -e "${RED}Please check the error messages and retry${NC}"
fi

echo -e "${BLUE}=======================================================================${NC}"

exit $EXIT_CODE
