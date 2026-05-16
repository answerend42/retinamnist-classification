#!/bin/bash

# 预训练权重对比实验脚本
# 对比使用预训练权重 vs 从头训练的性能差异

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# 训练参数
MAX_EPOCHS=200
BATCH_SIZE=16
TRAINER="mps"

# 模型列表 - 每个模型都有预训练和非预训练两个版本
MODELS=(
    "resnet18:resnet18_pretrained"
    "resnet34:resnet34_pretrained"
    "efficientnet_b0:efficientnet_b0_pretrained"
    "mobilenetv3_large_100:mobilenetv3_large_100_pretrained"
    "convnext_tiny:convnext_tiny_pretrained"
)

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  预训练权重对比实验${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${GREEN}实验设计:${NC}"
echo -e "  每个模型训练两次："
echo -e "  ${YELLOW}1. 从头训练 (pretrained=false)${NC}"
echo -e "  ${YELLOW}2. 使用预训练权重 (pretrained=true)${NC}"
echo ""
echo -e "${GREEN}训练配置:${NC}"
echo -e "  训练轮数: ${YELLOW}$MAX_EPOCHS${NC}"
echo -e "  批次大小: ${YELLOW}$BATCH_SIZE${NC}"
echo -e "  训练器: ${YELLOW}$TRAINER${NC}"
echo -e "  模型数量: ${YELLOW}${#MODELS[@]}${NC}"
echo -e "  总实验数: ${YELLOW}$((${#MODELS[@]} * 2))${NC}"
echo ""
echo -e "${CYAN}模型列表:${NC}"
for model_pair in "${MODELS[@]}"; do
    model_base=$(echo $model_pair | cut -d: -f1)
    echo -e "  - ${YELLOW}$model_base${NC} (从头训练 + 预训练)"
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
echo -e "${BLUE}  开始对比实验${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# 创建结果文件
RESULTS_FILE="pretrained_comparison_$(date +%Y%m%d_%H%M%S).txt"
echo "预训练权重对比实验结果" > $RESULTS_FILE
echo "实验时间: $(date)" >> $RESULTS_FILE
echo "========================================" >> $RESULTS_FILE
echo "" >> $RESULTS_FILE
echo "格式: 模型名 | 从头训练 | 预训练权重" >> $RESULTS_FILE
echo "========================================" >> $RESULTS_FILE
echo "" >> $RESULTS_FILE

# 训练计数器
TOTAL_EXPERIMENTS=$((${#MODELS[@]} * 2))
CURRENT=0
SUCCESSFUL=0
FAILED=0

# 遍历所有模型进行对比训练
for model_pair in "${MODELS[@]}"; do
    # 解析模型名称
    model_scratch=$(echo $model_pair | cut -d: -f1)
    model_pretrained=$(echo $model_pair | cut -d: -f2)

    echo "" >> $RESULTS_FILE
    echo "模型: $model_scratch" >> $RESULTS_FILE
    echo "----------------------------------------" >> $RESULTS_FILE

    # ========================================
    # 实验 1: 从头训练
    # ========================================
    CURRENT=$((CURRENT + 1))

    echo ""
    echo -e "${MAGENTA}========================================${NC}"
    echo -e "${MAGENTA}  实验 [$CURRENT/$TOTAL_EXPERIMENTS]${NC}"
    echo -e "${MAGENTA}  模型: $model_scratch${NC}"
    echo -e "${MAGENTA}  模式: ${RED}从头训练 (no pretrain)${NC}"
    echo -e "${MAGENTA}========================================${NC}"
    echo ""

    # 构建训练命令
    TRAIN_CMD="python src/train.py \
        logger=mlflow \
        model=$model_scratch \
        trainer=$TRAINER \
        trainer.max_epochs=$MAX_EPOCHS \
        data.batch_size=$BATCH_SIZE \
        tags=\"['retina-mnist', '$model_scratch', 'no-pretrain', 'comparison']\""

    echo -e "${BLUE}执行命令:${NC}"
    echo -e "${YELLOW}$TRAIN_CMD${NC}"
    echo ""

    # 记录开始时间
    START_TIME=$(date +%s)

    # 执行训练
    if eval $TRAIN_CMD; then
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        MINUTES=$((DURATION / 60))
        SECONDS=$((DURATION % 60))

        echo -e "${GREEN}✓ 从头训练成功 (耗时: ${MINUTES}m ${SECONDS}s)${NC}"
        echo "  从头训练: 成功 (${MINUTES}m ${SECONDS}s)" >> $RESULTS_FILE
        SUCCESSFUL=$((SUCCESSFUL + 1))
    else
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        MINUTES=$((DURATION / 60))
        SECONDS=$((DURATION % 60))

        echo -e "${RED}✗ 从头训练失败 (耗时: ${MINUTES}m ${SECONDS}s)${NC}"
        echo "  从头训练: 失败 (${MINUTES}m ${SECONDS}s)" >> $RESULTS_FILE
        FAILED=$((FAILED + 1))
    fi

    # ========================================
    # 实验 2: 使用预训练权重
    # ========================================
    CURRENT=$((CURRENT + 1))

    echo ""
    echo -e "${MAGENTA}========================================${NC}"
    echo -e "${MAGENTA}  实验 [$CURRENT/$TOTAL_EXPERIMENTS]${NC}"
    echo -e "${MAGENTA}  模型: $model_pretrained${NC}"
    echo -e "${MAGENTA}  模式: ${GREEN}预训练权重 (pretrained)${NC}"
    echo -e "${MAGENTA}========================================${NC}"
    echo ""

    # 构建训练命令
    TRAIN_CMD="python src/train.py \
        logger=mlflow \
        model=$model_pretrained \
        trainer=$TRAINER \
        trainer.max_epochs=$MAX_EPOCHS \
        data.batch_size=$BATCH_SIZE \
        tags=\"['retina-mnist', '$model_scratch', 'pretrained', 'comparison']\""

    echo -e "${BLUE}执行命令:${NC}"
    echo -e "${YELLOW}$TRAIN_CMD${NC}"
    echo ""

    # 记录开始时间
    START_TIME=$(date +%s)

    # 执行训练
    if eval $TRAIN_CMD; then
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        MINUTES=$((DURATION / 60))
        SECONDS=$((DURATION % 60))

        echo -e "${GREEN}✓ 预训练权重训练成功 (耗时: ${MINUTES}m ${SECONDS}s)${NC}"
        echo "  预训练权重: 成功 (${MINUTES}m ${SECONDS}s)" >> $RESULTS_FILE
        SUCCESSFUL=$((SUCCESSFUL + 1))
    else
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        MINUTES=$((DURATION / 60))
        SECONDS=$((DURATION % 60))

        echo -e "${RED}✗ 预训练权重训练失败 (耗时: ${MINUTES}m ${SECONDS}s)${NC}"
        echo "  预训练权重: 失败 (${MINUTES}m ${SECONDS}s)" >> $RESULTS_FILE
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

# 添加总结到结果文件
echo "" >> $RESULTS_FILE
echo "========================================" >> $RESULTS_FILE
echo "总结:" >> $RESULTS_FILE
echo "  总实验数: $TOTAL_EXPERIMENTS" >> $RESULTS_FILE
echo "  成功: $SUCCESSFUL" >> $RESULTS_FILE
echo "  失败: $FAILED" >> $RESULTS_FILE
echo "" >> $RESULTS_FILE
echo "查看对比结果: http://127.0.0.1:5000" >> $RESULTS_FILE
echo "在 MLflow UI 中使用 tags 过滤:" >> $RESULTS_FILE
echo "  - no-pretrain: 从头训练的实验" >> $RESULTS_FILE
echo "  - pretrained: 使用预训练权重的实验" >> $RESULTS_FILE

# 显示结果文件内容
echo -e "${CYAN}实验结果:${NC}"
cat $RESULTS_FILE

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  如何查看对比结果${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "1. 访问 MLflow UI: ${YELLOW}http://127.0.0.1:5000${NC}"
echo -e "2. 使用 tags 过滤实验:"
echo -e "   - ${RED}no-pretrain${NC}: 从头训练"
echo -e "   - ${GREEN}pretrained${NC}: 预训练权重"
echo -e "3. 对比指标:"
echo -e "   - val/acc: 验证准确率"
echo -e "   - val/loss: 验证损失"
echo -e "   - test/acc: 测试准确率"
echo ""
echo -e "${GREEN}MLflow UI 保持运行${NC}"
echo -e "${YELLOW}停止命令: kill $MLFLOW_PID${NC}"
echo ""