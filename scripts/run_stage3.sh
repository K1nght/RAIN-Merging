#!/bin/bash

# 启动 conda 环境
source /opt/conda/etc/profile.d/conda.sh
conda activate merge

# Stage 3: 统一模型合并
# 功能：将投影后的task vectors和alpha系数合并到目标模型
# 输入：projected_task_vectors.pkl (Stage1) + alpha系数文件 (Stage2，可选)
# 输出：合并后的完整模型

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 默认配置
DEFAULT_TARGET="/opt/data/private/hzhcode/huggingface/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
DEFAULT_PROJECTED=""
DEFAULT_ALPHA=""
DEFAULT_OUTPUT="./stage3_output_$(date +%Y%m%d_%H%M%S)"

# 参数设置
TARGET_MODEL=${1:-$DEFAULT_TARGET}
PROJECTED_FILE=${2:-$DEFAULT_PROJECTED}
ALPHA_FILE=${3:-$DEFAULT_ALPHA}
OUTPUT_DIR=${4:-$DEFAULT_OUTPUT}

# 合并配置参数（通过环境变量）
MODEL_NAME=${MODEL_NAME:-"merged_model"}
SCALING_FACTOR=${SCALING_FACTOR:-""}
VERBOSE=${VERBOSE:-true}

function show_help() {
    echo -e "${GREEN}Stage 3: 统一模型合并${NC}"
    echo ""
    echo "用法: $0 [target_model] [projected_file] [alpha_file] [output_dir]"
    echo ""
    echo -e "${YELLOW}位置参数:${NC}"
    echo "  target_model   目标模型路径 (默认: $DEFAULT_TARGET)"
    echo "  projected_file 投影文件(Stage1输出) (必需)"
    echo "  alpha_file     Alpha系数文件(Stage2输出) (可选)"
    echo "  output_dir     输出目录 (默认: $DEFAULT_OUTPUT)"
    echo ""
    echo -e "${YELLOW}环境变量配置:${NC}"
    echo "  MODEL_NAME     合并后模型名称 (默认: merged_model)"
    echo "  SCALING_FACTOR 固定缩放因子 (可选，如不使用alpha)"
    echo "  VERBOSE        详细输出: true/false (默认: true)"
    echo ""
    echo -e "${YELLOW}合并模式:${NC}"
    echo "  1. Alpha加权模式: 提供alpha_file，使用QP优化的系数"
    echo "  2. Scaling Factor模式: 不提供alpha_file，使用固定缩放因子"
    echo "  3. 组合模式: 同时提供alpha_file和SCALING_FACTOR"
    echo ""
    echo -e "${YELLOW}示例:${NC}"
    echo "  # Alpha加权模式"
    echo "  $0 /path/to/target ./projected.pkl ./alpha.pt ./output"
    echo ""
    echo "  # Scaling Factor模式"
    echo "  SCALING_FACTOR=0.8 $0 /path/to/target ./projected.pkl \"\" ./output"
    echo ""
    echo "  # 组合模式"
    echo "  SCALING_FACTOR=1.2 $0 /path/to/target ./projected.pkl ./alpha.pt ./output"
    echo ""
    echo "  # 自定义模型名称"
    echo "  MODEL_NAME=my_custom_model $0 /path/to/target ./projected.pkl ./alpha.pt ./output"
}

# 检查帮助参数
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    show_help
    exit 0
fi

echo -e "${BLUE}=======================================================================${NC}"
echo -e "${BLUE}                    Stage 3: 统一模型合并${NC}"
echo -e "${BLUE}=======================================================================${NC}"
echo -e "${GREEN}📁 目标模型: ${NC}$TARGET_MODEL"
echo -e "${GREEN}📁 投影文件: ${NC}$PROJECTED_FILE"
echo -e "${GREEN}📁 Alpha文件: ${NC}$ALPHA_FILE"
echo -e "${GREEN}📁 输出目录: ${NC}$OUTPUT_DIR"
echo -e "${GREEN}📁 模型名称: ${NC}$MODEL_NAME"

# 确定合并模式
MERGE_MODE=""
if [[ -n "$ALPHA_FILE" && -f "$ALPHA_FILE" && -n "$SCALING_FACTOR" ]]; then
    MERGE_MODE="组合模式 (Alpha × Scaling)"
    echo -e "${GREEN}🔧 合并模式: ${NC}$MERGE_MODE"
    echo -e "${GREEN}📊 缩放因子: ${NC}$SCALING_FACTOR"
elif [[ -n "$ALPHA_FILE" && -f "$ALPHA_FILE" ]]; then
    MERGE_MODE="Alpha加权模式"
    echo -e "${GREEN}🔧 合并模式: ${NC}$MERGE_MODE"
elif [[ -n "$SCALING_FACTOR" ]]; then
    MERGE_MODE="Scaling Factor模式"
    echo -e "${GREEN}🔧 合并模式: ${NC}$MERGE_MODE"
    echo -e "${GREEN}📊 缩放因子: ${NC}$SCALING_FACTOR"
else
    MERGE_MODE="默认Scaling Factor模式 (1.0)"
    SCALING_FACTOR="1.0"
    echo -e "${GREEN}🔧 合并模式: ${NC}$MERGE_MODE"
fi

echo -e "${BLUE}=======================================================================${NC}"

# 检查必要文件
if [[ ! -f "$PROJECTED_FILE" ]]; then
    echo -e "${RED}❌ 错误: 投影文件不存在: $PROJECTED_FILE${NC}"
    echo -e "${YELLOW}请先运行Stage 1生成投影文件${NC}"
    exit 1
fi

if [[ -n "$ALPHA_FILE" && ! -f "$ALPHA_FILE" ]]; then
    echo -e "${RED}❌ 错误: Alpha文件不存在: $ALPHA_FILE${NC}"
    echo -e "${YELLOW}请先运行Stage 2生成Alpha文件，或使用Scaling Factor模式${NC}"
    exit 1
fi

if [[ ! -f "unified_model_merge.py" ]]; then
    echo -e "${RED}❌ 错误: unified_model_merge.py不存在${NC}"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

echo -e "\n${BLUE}🔄 开始执行Stage 3: 统一模型合并...${NC}"

# 记录开始时间
START_TIME=$(date +%s)

# 构建命令
CMD="python unified_model_merge.py"
CMD="$CMD --projected_file \"$PROJECTED_FILE\""
CMD="$CMD --base_model \"$TARGET_MODEL\""
CMD="$CMD --output_dir \"$OUTPUT_DIR\""
CMD="$CMD --model_name \"$MODEL_NAME\""

# 添加verbose选项
if [[ "$VERBOSE" == "true" ]]; then
    CMD="$CMD --verbose"
fi

# 添加Alpha文件
if [[ -n "$ALPHA_FILE" && -f "$ALPHA_FILE" ]]; then
    CMD="$CMD --alpha_file \"$ALPHA_FILE\""
fi

# 添加Scaling Factor
if [[ -n "$SCALING_FACTOR" ]]; then
    CMD="$CMD --scaling_factor $SCALING_FACTOR"
fi

echo -e "${YELLOW}执行命令:${NC}"
echo "$CMD"
echo ""

# 执行模型合并
eval $CMD

EXIT_CODE=$?
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo -e "${BLUE}=======================================================================${NC}"

if [[ $EXIT_CODE -eq 0 ]]; then
    echo -e "${GREEN}✅ Stage 3 执行成功! 耗时: ${DURATION}秒${NC}"
    echo -e "${GREEN}📁 输出目录: $OUTPUT_DIR${NC}"
    echo ""
    
    # 检查合并模型
    MERGED_MODEL_DIR="$OUTPUT_DIR/$MODEL_NAME"
    if [[ -d "$MERGED_MODEL_DIR" ]]; then
        echo -e "${GREEN}🤖 合并模型: $MERGED_MODEL_DIR${NC}"
        echo ""
        echo -e "${YELLOW}📊 模型文件:${NC}"
        ls -la "$MERGED_MODEL_DIR"
        echo ""
        
        # 检查关键文件
        KEY_FILES=("config.json" "pytorch_model.bin" "tokenizer.json")
        MISSING_FILES=()
        
        for file in "${KEY_FILES[@]}"; do
            if [[ ! -f "$MERGED_MODEL_DIR/$file" ]]; then
                MISSING_FILES+=("$file")
            fi
        done
        
        if [[ ${#MISSING_FILES[@]} -eq 0 ]]; then
            echo -e "${GREEN}✅ 模型文件完整${NC}"
        else
            echo -e "${YELLOW}⚠️  缺少模型文件: ${MISSING_FILES[*]}${NC}"
        fi
        
        echo ""
        echo -e "${YELLOW}🎉 三阶段流水线完成!${NC}"
        echo -e "${GREEN}📄 使用合并模型:${NC}"
        echo "  from transformers import AutoModelForCausalLM, AutoTokenizer"
        echo "  model = AutoModelForCausalLM.from_pretrained('$MERGED_MODEL_DIR')"
        echo "  tokenizer = AutoTokenizer.from_pretrained('$MERGED_MODEL_DIR')"
        
    else
        echo -e "${RED}⚠️  未找到合并模型目录: $MERGED_MODEL_DIR${NC}"
    fi
    
    # 检查统计文件
    STATS_FILE="$OUTPUT_DIR/unified_merge_stats.json"
    if [[ -f "$STATS_FILE" ]]; then
        echo ""
        echo -e "${YELLOW}📊 合并统计信息:${NC}"
        python -c "
import json
try:
    with open('$STATS_FILE', 'r') as f:
        stats = json.load(f)
    print('  修改参数数量:', f\"{stats.get('total_params_modified', 'N/A'):,}\")
    if 'merge_info' in stats:
        print('  合并模式:', stats['merge_info'].get('mode', 'N/A'))
        if 'alpha_info' in stats['merge_info']:
            alpha_stats = stats['merge_info']['alpha_info']['alpha_stats']
            print(f'  Alpha范围: [{alpha_stats[\"min\"]:.3f}, {alpha_stats[\"max\"]:.3f}]')
            print(f'  Alpha均值: {alpha_stats[\"mean\"]:.3f}')
        if 'scaling_factor' in stats['merge_info']:
            print('  Scaling Factor:', stats['merge_info']['scaling_factor'])
except Exception as e:
    print('  读取统计信息失败:', e)
"
    fi
    
else
    echo -e "${RED}❌ Stage 3 执行失败，退出码: $EXIT_CODE${NC}"
    echo -e "${RED}请检查错误信息并重试${NC}"
fi

echo -e "${BLUE}=======================================================================${NC}"

exit $EXIT_CODE

