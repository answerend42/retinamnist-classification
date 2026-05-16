#!/bin/bash

# 训练不使用预训练权重的模型
# 使用从预训练实验中找到的最佳超参数

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# 模型列表
EXPERIMENTS=(
    "no_pretrain_resnet18"
    "no_pretrain_efficientnet_b0"
    "no_pretrain_convnext_tiny"
)

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  无预训练权重对比实验${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${GREEN}实验目的:${NC}"
echo -e "  使用最佳超参数训练不使用预训练权重的模型"
echo -e "  对比预训练 vs 非预训练的性能差异"
echo ""
echo -e "${CYAN}模型列表:${NC}"
for exp in "${EXPERIMENTS[@]}"; do
    echo -e "  - ${YELLOW}$exp${NC}"
done
echo ""

# 检查虚拟环境
if [ ! -d ".venv" ]; then
    echo -e "${RED}错误: 未找到虚拟环境 .venv${NC}"
    echo -e "${YELLOW}请先运行: uv sync${NC}"
    exit 1
fi

# 激活虚拟环境
echo -e "${GREEN}激活虚拟环境...${NC}"
source .venv/bin/activate

# 检查并启动 MLflow UI
echo -e "${YELLOW}检查 MLflow UI 状态...${NC}"
pkill -f "mlflow ui" 2>/dev/null || true
sleep 1

if lsof -ti:5000 > /dev/null 2>&1; then
    echo -e "${YELLOW}清理端口 5000...${NC}"
    lsof -ti:5000 | xargs kill -9 2>/dev/null || true
    sleep 2
fi

echo -e "${GREEN}启动 MLflow UI...${NC}"
mlflow ui --backend-store-uri sqlite:///mlflow.db > /tmp/mlflow_ui.log 2>&1 &
MLFLOW_PID=$!
sleep 3

if ps -p $MLFLOW_PID > /dev/null 2>&1; then
    echo -e "${GREEN}✓ MLflow UI 已启动 (PID: $MLFLOW_PID)${NC}"
    echo -e "${GREEN}✓ 访问地址: http://127.0.0.1:5000${NC}"
else
    echo -e "${RED}✗ MLflow UI 启动失败${NC}"
    exit 1
fi

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  开始训练${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# 创建结果文件
RESULTS_FILE="no_pretrain_results_$(date +%Y%m%d_%H%M%S).txt"
echo "无预训练权重对比实验结果" > $RESULTS_FILE
echo "实验时间: $(date)" >> $RESULTS_FILE
echo "========================================" >> $RESULTS_FILE
echo "" >> $RESULTS_FILE

# 实验计数器
TOTAL_EXPERIMENTS=${#EXPERIMENTS[@]}
CURRENT=0
SUCCESSFUL=0
FAILED=0

# 遍历所有实验
for exp in "${EXPERIMENTS[@]}"; do
    CURRENT=$((CURRENT + 1))

    echo ""
    echo -e "${MAGENTA}========================================${NC}"
    echo -e "${MAGENTA}  实验 [$CURRENT/$TOTAL_EXPERIMENTS]${NC}"
    echo -e "${MAGENTA}  配置: $exp${NC}"
    echo -e "${MAGENTA}========================================${NC}"
    echo ""

    echo "" >> $RESULTS_FILE
    echo "配置: $exp" >> $RESULTS_FILE
    echo "----------------------------------------" >> $RESULTS_FILE

    # 构建训练命令
    TRAIN_CMD="python src/train.py experiment=$exp"

    echo -e "${BLUE}执行命令:${NC}"
    echo -e "${YELLOW}$TRAIN_CMD${NC}"
    echo ""

    # 记录开始时间
    START_TIME=$(date +%s)

    # 执行训练
    if eval $TRAIN_CMD; then
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        HOURS=$((DURATION / 3600))
        MINUTES=$(((DURATION % 3600) / 60))
        SECONDS=$((DURATION % 60))

        echo -e "${GREEN}✓ 训练成功 (耗时: ${HOURS}h ${MINUTES}m ${SECONDS}s)${NC}"
        echo "  状态: 成功 (${HOURS}h ${MINUTES}m ${SECONDS}s)" >> $RESULTS_FILE
        SUCCESSFUL=$((SUCCESSFUL + 1))
    else
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        HOURS=$((DURATION / 3600))
        MINUTES=$(((DURATION % 3600) / 60))
        SECONDS=$((DURATION % 60))

        echo -e "${RED}✗ 训练失败 (耗时: ${HOURS}h ${MINUTES}m ${SECONDS}s)${NC}"
        echo "  状态: 失败 (${HOURS}h ${MINUTES}m ${SECONDS}s)" >> $RESULTS_FILE
        FAILED=$((FAILED + 1))
    fi

    echo ""
done

# 实验完成总结
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  实验完成总结${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${GREEN}总实验数: $TOTAL_EXPERIMENTS${NC}"
echo -e "${GREEN}成功: $SUCCESSFUL${NC}"
echo -e "${RED}失败: $FAILED${NC}"
echo ""
echo -e "${YELLOW}详细结果已保存到: $RESULTS_FILE${NC}"
echo ""

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  查看结果${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "1. 访问 MLflow UI: ${YELLOW}http://127.0.0.1:5000${NC}"
echo -e "2. 比较预训练 vs 非预训练模型的性能"
echo -e "3. 查看 test/auc 指标进行对比"
echo ""
echo -e "${YELLOW}停止 MLflow UI: kill $MLFLOW_PID${NC}"
echo ""
