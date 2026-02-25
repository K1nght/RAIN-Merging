#!/bin/bash

# Activate conda environment
source /opt/conda/etc/profile.d/conda.sh
conda activate merge

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
DEFAULT_BASE="/opt/data/private/hzhcode/huggingface/models/Qwen/Qwen2.5-7B"
DEFAULT_INSTRUCT="/opt/data/private/hzhcode/huggingface/models/Qwen/Qwen2.5-7B-Instruct"
DEFAULT_TARGET="/opt/data/private/hzhcode/huggingface/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
DEFAULT_DATA="./data/reasoning_calibration_set.json"
DEFAULT_OUTPUT="./stage1_output_$(date +%Y%m%d_%H%M%S)"

# Parameter settings
BASE_MODEL=${1:-$DEFAULT_BASE}
INSTRUCT_MODEL=${2:-$DEFAULT_INSTRUCT}
TARGET_MODEL=${3:-$DEFAULT_TARGET}
DATA_FILE=${4:-$DEFAULT_DATA}
OUTPUT_DIR=${5:-$DEFAULT_OUTPUT}

# Configurable parameters (via environment variables)
MAX_SAMPLES=${MAX_SAMPLES:-10}
LAYERS_TAIL=${LAYERS_TAIL:-2}
HEADS=${HEADS:-"all"}
MERGE_TYPES=${MERGE_TYPES:-"qkvof"}
COMPUTE_PRECISION=${COMPUTE_PRECISION:-"fp32"}
LAMBDA_RIDGE=${LAMBDA_RIDGE:-1e-4}
CG_MAXIT=${CG_MAXIT:-100}
CG_TOL=${CG_TOL:-1e-5}

# 设备配置
QK_DEVICE=${QK_DEVICE:-"auto"}
VO_DEVICE=${VO_DEVICE:-"auto"}
FFN_DEVICE=${FFN_DEVICE:-"auto"}

# 约束参数配置
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

# 序列长度限制（基于BF16优化，注意力矩阵使用BF16节省50%显存）
MAX_SEQ_LEN=${MAX_SEQ_LEN:-7168}

function show_help() {
    echo -e "${GREEN}Stage 1: Null-space投影计算${NC}"
    echo ""
    echo "用法: $0 [base_model] [instruct_model] [target_model] [data_file] [output_dir]"
    echo ""
    echo -e "${YELLOW}位置参数:${NC}"
    echo "  base_model     基础模型路径 (默认: $DEFAULT_BASE)"
    echo "  instruct_model 指令模型路径 (默认: $DEFAULT_INSTRUCT)"
    echo "  target_model   目标模型路径 (默认: $DEFAULT_TARGET)"
    echo "  data_file      训练数据文件 (默认: $DEFAULT_DATA)"
    echo "  output_dir     输出目录 (默认: $DEFAULT_OUTPUT)"
    echo ""
    echo -e "${YELLOW}环境变量配置:${NC}"
    echo "  MAX_SAMPLES        最大样本数量 (默认: 10)"
    echo "  LAYERS_TAIL        处理后N层 (默认: 2)"
    echo "  HEADS              处理的头 (默认: all)"
    echo "  MERGE_TYPES        合并类型 (默认: qkvof)"
    echo "  COMPUTE_PRECISION  计算精度 (默认: fp32)"
    echo "  LAMBDA_RIDGE       岭回归参数 (默认: 1e-4)"
    echo "  CG_MAXIT           CG最大迭代 (默认: 100)"
    echo "  CG_TOL             CG收敛容差 (默认: 1e-5)"
    echo ""
    echo -e "${YELLOW}设备配置:${NC}"
    echo "  QK_DEVICE          QK计算设备 (默认: auto)"
    echo "  VO_DEVICE          VO计算设备 (默认: auto)"
    echo "  FFN_DEVICE         FFN计算设备 (默认: auto)"
    echo ""
    echo -e "${YELLOW}序列长度控制:${NC}"
    echo "  MAX_SEQ_LEN        最大序列长度限制 (默认: 7168, BF16优化, 注意力矩阵节省50%显存)"
    echo ""
    echo -e "${YELLOW}示例:${NC}"
    echo "  # 基本用法"
    echo "  $0 /path/to/base /path/to/instruct /path/to/target"
    echo ""
    echo "  # 高精度计算"
    echo "  COMPUTE_PRECISION=fp64 LAMBDA_RIDGE=1e-5 $0"
    echo ""
    echo "  # 多GPU配置"
    echo "  QK_DEVICE=cuda:0 VO_DEVICE=cuda:1 FFN_DEVICE=cuda:2 $0"
    echo ""
    echo "  # 更多样本和层"
    echo "  MAX_SAMPLES=20 LAYERS_TAIL=4 $0"
}

# 检查帮助参数
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    show_help
    exit 0
fi

echo -e "${BLUE}=======================================================================${NC}"
echo -e "${BLUE}                    Stage 1: Null-space投影计算${NC}"
echo -e "${BLUE}=======================================================================${NC}"
echo -e "${GREEN}📁 基础模型: ${NC}$BASE_MODEL"
echo -e "${GREEN}📁 指令模型: ${NC}$INSTRUCT_MODEL"  
echo -e "${GREEN}📁 目标模型: ${NC}$TARGET_MODEL"
echo -e "${GREEN}📁 训练数据: ${NC}$DATA_FILE"
echo -e "${GREEN}📁 输出目录: ${NC}$OUTPUT_DIR"
echo ""
echo -e "${YELLOW}配置参数:${NC}"
echo "  最大样本: $MAX_SAMPLES"
echo "  处理层数: $LAYERS_TAIL"
echo "  处理头数: $HEADS"
echo "  合并类型: $MERGE_TYPES"
echo "  计算精度: $COMPUTE_PRECISION"
echo "  设备配置: QK=$QK_DEVICE, VO=$VO_DEVICE, FFN=$FFN_DEVICE"
echo "  序列长度限制: $MAX_SEQ_LEN tokens (BF16优化, 注意力矩阵节省50%显存)"
echo -e "${BLUE}=======================================================================${NC}"

# 检查必要文件
if [[ ! -f "$DATA_FILE" ]]; then
    echo -e "${RED}❌ 错误: 数据文件不存在: $DATA_FILE${NC}"
    echo -e "${YELLOW}可用的数据文件:${NC}"
    ls -la data/ 2>/dev/null || echo "data目录不存在"
    exit 1
fi

if [[ ! -f "nullspace_projection_compute.py" ]]; then
    echo -e "${RED}❌ 错误: nullspace_projection_compute.py不存在${NC}"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 构建输出文件路径
OUTPUT_FILE="$OUTPUT_DIR/projected_task_vectors.pkl"

echo -e "\n${BLUE}🔄 开始执行Stage 1: Null-space投影计算...${NC}"

# 记录开始时间
START_TIME=$(date +%s)

# 执行投影计算
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
    echo -e "${GREEN}✅ Stage 1 执行成功! 耗时: ${DURATION}秒${NC}"
    echo -e "${GREEN}📁 输出目录: $OUTPUT_DIR${NC}"
    echo -e "${GREEN}📄 投影文件: $OUTPUT_FILE${NC}"
    echo ""
    echo -e "${YELLOW}📊 输出文件:${NC}"
    ls -la "$OUTPUT_DIR"
    echo ""
    echo -e "${YELLOW}🚀 下一步: 运行Stage 2 (QP优化)${NC}"
    echo "  ./run_stage2.sh \"$TARGET_MODEL\" \"$DATA_FILE\" \"$OUTPUT_FILE\" \"./stage2_output\""
else
    echo -e "${RED}❌ Stage 1 执行失败，退出码: $EXIT_CODE${NC}"
    echo -e "${RED}请检查错误信息并重试${NC}"
fi

echo -e "${BLUE}=======================================================================${NC}"

exit $EXIT_CODE

