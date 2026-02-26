#!/bin/bash

# Activate conda environment
source activate merge

# Stage 3: Unified model merging
# Function: Merge projected task vectors and alpha coefficients into the target model
# Input: projected_task_vectors.pkl (Stage 1) + alpha coefficient file (Stage 2, optional)
# Output: fully merged model

# Color definitions
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
DEFAULT_TARGET="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
DEFAULT_PROJECTED=""
DEFAULT_ALPHA=""
DEFAULT_OUTPUT="./stage3_output_$(date +%Y%m%d_%H%M%S)"

# Parameter settings
TARGET_MODEL=${1:-$DEFAULT_TARGET}
PROJECTED_FILE=${2:-$DEFAULT_PROJECTED}
ALPHA_FILE=${3:-$DEFAULT_ALPHA}
OUTPUT_DIR=${4:-$DEFAULT_OUTPUT}

# Merge configuration parameters (via environment variables)
MODEL_NAME=${MODEL_NAME:-"merged_model"}
SCALING_FACTOR=${SCALING_FACTOR:-""}
VERBOSE=${VERBOSE:-true}

function show_help() {
    echo -e "${GREEN}Stage 3: Unified Model Merging${NC}"
    echo ""
    echo "Usage: $0 [target_model] [projected_file] [alpha_file] [output_dir]"
    echo ""
    echo -e "${YELLOW}Positional Arguments:${NC}"
    echo "  target_model   Target model path (default: $DEFAULT_TARGET)"
    echo "  projected_file Projection file (Stage 1 output) (required)"
    echo "  alpha_file     Alpha coefficient file (Stage 2 output) (optional)"
    echo "  output_dir     Output directory (default: $DEFAULT_OUTPUT)"
    echo ""
    echo -e "${YELLOW}Environment Variable Configuration:${NC}"
    echo "  MODEL_NAME     Name of merged model (default: merged_model)"
    echo "  SCALING_FACTOR Fixed scaling factor (optional, used when alpha is not provided)"
    echo "  VERBOSE        Verbose output: true/false (default: true)"
    echo ""
    echo -e "${YELLOW}Merge Modes:${NC}"
    echo "  1. Alpha-weighted mode: provide alpha_file, uses QP-optimized coefficients"
    echo "  2. Scaling Factor mode: no alpha_file, uses a fixed scaling factor"
    echo "  3. Combined mode: provide both alpha_file and SCALING_FACTOR"
    echo ""
    echo -e "${YELLOW}Examples:${NC}"
    echo "  # Alpha-weighted mode"
    echo "  $0 /path/to/target ./projected.pkl ./alpha.pt ./output"
    echo ""
    echo "  # Scaling Factor mode"
    echo "  SCALING_FACTOR=0.8 $0 /path/to/target ./projected.pkl \"\" ./output"
    echo ""
    echo "  # Combined mode"
    echo "  SCALING_FACTOR=1.2 $0 /path/to/target ./projected.pkl ./alpha.pt ./output"
    echo ""
    echo "  # Custom model name"
    echo "  MODEL_NAME=my_custom_model $0 /path/to/target ./projected.pkl ./alpha.pt ./output"
}

# Check for help argument
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    show_help
    exit 0
fi

echo -e "${BLUE}=======================================================================${NC}"
echo -e "${BLUE}                    Stage 3: Unified Model Merging${NC}"
echo -e "${BLUE}=======================================================================${NC}"
echo -e "${GREEN}📁 Target Model: ${NC}$TARGET_MODEL"
echo -e "${GREEN}📁 Projection File: ${NC}$PROJECTED_FILE"
echo -e "${GREEN}📁 Alpha File: ${NC}$ALPHA_FILE"
echo -e "${GREEN}📁 Output Directory: ${NC}$OUTPUT_DIR"
echo -e "${GREEN}📁 Model Name: ${NC}$MODEL_NAME"

# Determine merge mode
MERGE_MODE=""
if [[ -n "$ALPHA_FILE" && -f "$ALPHA_FILE" && -n "$SCALING_FACTOR" ]]; then
    MERGE_MODE="Combined mode (Alpha × Scaling)"
    echo -e "${GREEN}🔧 Merge mode: ${NC}$MERGE_MODE"
    echo -e "${GREEN}📊 Scaling factor: ${NC}$SCALING_FACTOR"
elif [[ -n "$ALPHA_FILE" && -f "$ALPHA_FILE" ]]; then
    MERGE_MODE="Alpha-weighted mode"
    echo -e "${GREEN}🔧 Merge mode: ${NC}$MERGE_MODE"
elif [[ -n "$SCALING_FACTOR" ]]; then
    MERGE_MODE="Scaling Factor mode"
    echo -e "${GREEN}🔧 Merge mode: ${NC}$MERGE_MODE"
    echo -e "${GREEN}📊 Scaling factor: ${NC}$SCALING_FACTOR"
else
    MERGE_MODE="Default Scaling Factor mode (1.0)"
    SCALING_FACTOR="1.0"
    echo -e "${GREEN}🔧 Merge mode: ${NC}$MERGE_MODE"
fi

echo -e "${BLUE}=======================================================================${NC}"

# Check required files
if [[ ! -f "$PROJECTED_FILE" ]]; then
    echo -e "${RED}❌ Error: Projection file does not exist: $PROJECTED_FILE${NC}"
    echo -e "${YELLOW}Please run Stage 1 first to generate the projection file${NC}"
    exit 1
fi

if [[ -n "$ALPHA_FILE" && ! -f "$ALPHA_FILE" ]]; then
    echo -e "${RED}❌ Error: Alpha file does not exist: $ALPHA_FILE${NC}"
    echo -e "${YELLOW}Please run Stage 2 first to generate the alpha file, or use Scaling Factor mode${NC}"
    exit 1
fi

if [[ ! -f "unified_model_merge.py" ]]; then
    echo -e "${RED}❌ Error: unified_model_merge.py does not exist${NC}"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo -e "\n${BLUE}🔄 Starting Stage 3: Unified Model Merging...${NC}"

# Record start time
START_TIME=$(date +%s)

# Build command
CMD="python unified_model_merge.py"
CMD="$CMD --projected_file \"$PROJECTED_FILE\""
CMD="$CMD --base_model \"$TARGET_MODEL\""
CMD="$CMD --output_dir \"$OUTPUT_DIR\""
CMD="$CMD --model_name \"$MODEL_NAME\""

# Add verbose option
if [[ "$VERBOSE" == "true" ]]; then
    CMD="$CMD --verbose"
fi

# Add alpha file
if [[ -n "$ALPHA_FILE" && -f "$ALPHA_FILE" ]]; then
    CMD="$CMD --alpha_file \"$ALPHA_FILE\""
fi

# Add scaling factor
if [[ -n "$SCALING_FACTOR" ]]; then
    CMD="$CMD --scaling_factor $SCALING_FACTOR"
fi

echo -e "${YELLOW}Executing command:${NC}"
echo "$CMD"
echo ""

# Execute model merging
eval $CMD

EXIT_CODE=$?
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo -e "${BLUE}=======================================================================${NC}"

if [[ $EXIT_CODE -eq 0 ]]; then
    echo -e "${GREEN}✅ Stage 3 completed successfully! Elapsed: ${DURATION}s${NC}"
    echo -e "${GREEN}📁 Output directory: $OUTPUT_DIR${NC}"
    echo ""
    
    # Check merged model
    MERGED_MODEL_DIR="$OUTPUT_DIR/$MODEL_NAME"
    if [[ -d "$MERGED_MODEL_DIR" ]]; then
        echo -e "${GREEN}🤖 Merged model: $MERGED_MODEL_DIR${NC}"
        echo ""
        echo -e "${YELLOW}📊 Model files:${NC}"
        ls -la "$MERGED_MODEL_DIR"
        echo ""
        
        # Check key files
        KEY_FILES=("config.json" "pytorch_model.bin" "tokenizer.json")
        MISSING_FILES=()
        
        for file in "${KEY_FILES[@]}"; do
            if [[ ! -f "$MERGED_MODEL_DIR/$file" ]]; then
                MISSING_FILES+=("$file")
            fi
        done
        
        if [[ ${#MISSING_FILES[@]} -eq 0 ]]; then
            echo -e "${GREEN}✅ Model files are complete${NC}"
        else
            echo -e "${YELLOW}⚠️  Missing model files: ${MISSING_FILES[*]}${NC}"
        fi
        
        echo ""
        echo -e "${YELLOW}🎉 Three-stage pipeline complete!${NC}"
        echo -e "${GREEN}📄 Using the merged model:${NC}"
        echo "  from transformers import AutoModelForCausalLM, AutoTokenizer"
        echo "  model = AutoModelForCausalLM.from_pretrained('$MERGED_MODEL_DIR')"
        echo "  tokenizer = AutoTokenizer.from_pretrained('$MERGED_MODEL_DIR')"
        
    else
        echo -e "${RED}⚠️  Merged model directory not found: $MERGED_MODEL_DIR${NC}"
    fi
    
    # Check stats file
    STATS_FILE="$OUTPUT_DIR/unified_merge_stats.json"
    if [[ -f "$STATS_FILE" ]]; then
        echo ""
        echo -e "${YELLOW}📊 Merge statistics:${NC}"
        python -c "
import json
try:
    with open('$STATS_FILE', 'r') as f:
        stats = json.load(f)
    print('  Parameters modified:', f\"{stats.get('total_params_modified', 'N/A'):,}\")
    if 'merge_info' in stats:
        print('  Merge mode:', stats['merge_info'].get('mode', 'N/A'))
        if 'alpha_info' in stats['merge_info']:
            alpha_stats = stats['merge_info']['alpha_info']['alpha_stats']
            print(f'  Alpha range: [{alpha_stats[\"min\"]:.3f}, {alpha_stats[\"max\"]:.3f}]')
            print(f'  Alpha mean: {alpha_stats[\"mean\"]:.3f}')
        if 'scaling_factor' in stats['merge_info']:
            print('  Scaling factor:', stats['merge_info']['scaling_factor'])
except Exception as e:
    print('  Failed to read statistics:', e)
"
    fi
    
else
    echo -e "${RED}❌ Stage 3 failed with exit code: $EXIT_CODE${NC}"
    echo -e "${RED}Please check the error messages and retry${NC}"
fi

echo -e "${BLUE}=======================================================================${NC}"

exit $EXIT_CODE
