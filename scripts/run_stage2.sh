#!/bin/bash

# 启动 conda 环境
source /opt/conda/etc/profile.d/conda.sh
conda activate merge

# Stage 2: QP优化alpha系数
# 功能：基于投影后的task vectors优化合并系数alpha
# 输入：projected_task_vectors.pkl (来自Stage 1)
# 输出：alpha系数文件和QP结果

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 默认配置
DEFAULT_TARGET="/opt/data/private/hzhcode/huggingface/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
DEFAULT_DATA="./data/instruction_calibration_set.jsonl"
DEFAULT_PROJECTED=""
DEFAULT_OUTPUT="./stage2_output_$(date +%Y%m%d_%H%M%S)"

# 参数设置
TARGET_MODEL=${1:-$DEFAULT_TARGET}
DATA_FILE=${2:-$DEFAULT_DATA}
PROJECTED_FILE=${3:-$DEFAULT_PROJECTED}
OUTPUT_DIR=${4:-$DEFAULT_OUTPUT}

# QP优化参数（通过环境变量配置）
QP_VARIANT=${QP_VARIANT:-"two_pass"}
PRIOR_SCALAR=${PRIOR_SCALAR:-1.0}
L2_PRIOR=${L2_PRIOR:-0.1}
L1_REG=${L1_REG:-0.0}
BOX_LO=${BOX_LO:-0.0}
BOX_HI=${BOX_HI:-1.5}
DEVICE=${DEVICE:-"cuda:0"}

# QP构建参数
H_LAMBDA=${H_LAMBDA:-1.0}
H_MU=${H_MU:-1.0}
RHO_DU=${RHO_DU:-0.5}
KAPPA_A=${KAPPA_A:-1.0}
KAPPA_U=${KAPPA_U:-1.0}

# 其他选项
DECOUPLE_QK=${DECOUPLE_QK:-false}
SAVE_MODEL=${SAVE_MODEL:-false}
LAYERS=${LAYERS:-"all"}
HEADS=${HEADS:-"all"}

function show_help() {
    echo -e "${GREEN}Stage 2: QP优化alpha系数${NC}"
    echo ""
    echo "用法: $0 [target_model] [data_file] [projected_file] [output_dir]"
    echo ""
    echo -e "${YELLOW}位置参数:${NC}"
    echo "  target_model   目标模型路径 (默认: $DEFAULT_TARGET)"
    echo "  data_file      训练数据文件 (默认: $DEFAULT_DATA)"
    echo "  projected_file 投影文件(Stage1输出) (必需)"
    echo "  output_dir     输出目录 (默认: $DEFAULT_OUTPUT)"
    echo ""
    echo -e "${YELLOW}环境变量配置:${NC}"
    echo "  QP_VARIANT     QP构建方式: two_pass/anchor_only/post_only (默认: two_pass)"
    echo "  PRIOR_SCALAR   Alpha先验值 (默认: 1.0)"
    echo "  L2_PRIOR       L2正则化参数 (默认: 0.1)"
    echo "  L1_REG         L1正则化参数 (默认: 0.0)"
    echo "  BOX_LO         Box约束下界 (默认: 0.0)"
    echo "  BOX_HI         Box约束上界 (默认: 1.5)"
    echo "  DEVICE         计算设备 (默认: cuda:0)"
    echo ""
    echo -e "${YELLOW}高级参数:${NC}"
    echo "  H_LAMBDA       H矩阵对角常数 (默认: 1.0)"
    echo "  H_MU           后验泄漏权重 (默认: 1.0)"
    echo "  RHO_DU         泄漏变化惩罚 (默认: 0.5)"
    echo "  KAPPA_A        对齐打分缩放 (默认: 1.0)"
    echo "  KAPPA_U        泄漏打分缩放 (默认: 1.0)"
    echo ""
    echo -e "${YELLOW}布尔选项:${NC}"
    echo "  DECOUPLE_QK    解耦Q/K系数: true/false (默认: false)"
    echo "  SAVE_MODEL     保存QP优化模型: true/false (默认: false)"
    echo ""
    echo -e "${YELLOW}示例:${NC}"
    echo "  # 基本用法"
    echo "  $0 /path/to/target ./data/instruction_calibration_set.jsonl ./stage1_output/projected_task_vectors.pkl"
    echo ""
    echo "  # anchor_only模式"
    echo "  QP_VARIANT=anchor_only $0 /path/to/target ./data/instruction_calibration_set.jsonl ./projected.pkl"
    echo ""
    echo "  # 解耦Q/K并保存模型"
    echo "  DECOUPLE_QK=true SAVE_MODEL=true $0 /path/to/target ./data/instruction_calibration_set.jsonl ./projected.pkl"
    echo ""
    echo "  # 自定义QP参数"
    echo "  L2_PRIOR=0.2 BOX_HI=2.0 H_LAMBDA=0.5 $0 /path/to/target ./data/instruction_calibration_set.jsonl ./projected.pkl"
}

# 检查帮助参数
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    show_help
    exit 0
fi

echo -e "${BLUE}=======================================================================${NC}"
echo -e "${BLUE}                    Stage 2: QP优化alpha系数${NC}"
echo -e "${BLUE}=======================================================================${NC}"
echo -e "${GREEN}📁 目标模型: ${NC}$TARGET_MODEL"
echo -e "${GREEN}📁 训练数据: ${NC}$DATA_FILE"
echo -e "${GREEN}📁 投影文件: ${NC}$PROJECTED_FILE"
echo -e "${GREEN}📁 输出目录: ${NC}$OUTPUT_DIR"
echo ""
echo -e "${YELLOW}QP配置参数:${NC}"
echo "  QP变体: $QP_VARIANT"
echo "  Alpha先验: $PRIOR_SCALAR"
echo "  L2正则: $L2_PRIOR"
echo "  L1正则: $L1_REG"
echo "  Box约束: [$BOX_LO, $BOX_HI]"
echo "  计算设备: $DEVICE"
echo "  解耦Q/K: $DECOUPLE_QK"
echo "  保存模型: $SAVE_MODEL"
echo -e "${BLUE}=======================================================================${NC}"

# 检查必要文件
if [[ ! -f "$DATA_FILE" ]]; then
    echo -e "${RED}❌ 错误: 数据文件不存在: $DATA_FILE${NC}"
    exit 1
fi

if [[ ! -f "$PROJECTED_FILE" ]]; then
    echo -e "${RED}❌ 错误: 投影文件不存在: $PROJECTED_FILE${NC}"
    echo -e "${YELLOW}请先运行Stage 1生成投影文件${NC}"
    exit 1
fi

if [[ ! -f "qp_true_forward_fast.py" ]]; then
    echo -e "${RED}❌ 错误: qp_true_forward_fast.py不存在${NC}"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

echo -e "\n${BLUE}🔄 开始执行Stage 2: QP优化alpha系数...${NC}"

# 记录开始时间
START_TIME=$(date +%s)

# 构建命令
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

# 添加QP构建参数
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

# 添加布尔选项
if [[ "$DECOUPLE_QK" == "true" ]]; then
    CMD="$CMD --decouple_qk"
fi

if [[ "$SAVE_MODEL" == "true" ]]; then
    CMD="$CMD --save_model"
fi

echo -e "${YELLOW}执行命令:${NC}"
echo "$CMD"
echo ""

# 执行QP优化
eval $CMD

EXIT_CODE=$?
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo -e "${BLUE}=======================================================================${NC}"

if [[ $EXIT_CODE -eq 0 ]]; then
    echo -e "${GREEN}✅ Stage 2 执行成功! 耗时: ${DURATION}秒${NC}"
    echo -e "${GREEN}📁 输出目录: $OUTPUT_DIR${NC}"
    echo ""
    echo -e "${YELLOW}📊 输出文件:${NC}"
    ls -la "$OUTPUT_DIR"
    echo ""
    
    # 查找alpha文件
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
        echo -e "${GREEN}🎯 Alpha系数文件: $ALPHA_FILE${NC}"
        echo ""
        echo -e "${YELLOW}🚀 下一步: 运行Stage 3 (模型合并)${NC}"
        echo "  ./run_stage3.sh \"$TARGET_MODEL\" \"$PROJECTED_FILE\" \"$ALPHA_FILE\" \"./stage3_output\""
    else
        echo -e "${YELLOW}⚠️  未找到alpha系数文件，可使用scaling factor模式运行Stage 3${NC}"
        echo "  ./run_stage3.sh \"$TARGET_MODEL\" \"$PROJECTED_FILE\" \"\" \"./stage3_output\""
    fi
else
    echo -e "${RED}❌ Stage 2 执行失败，退出码: $EXIT_CODE${NC}"
    echo -e "${RED}请检查错误信息并重试${NC}"
fi

echo -e "${BLUE}=======================================================================${NC}"

exit $EXIT_CODE

