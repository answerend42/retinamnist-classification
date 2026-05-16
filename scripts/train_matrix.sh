#!/bin/bash

# 矩阵化训练脚本 - 训练多个模型并比较性能
# 使用 timm 库中的优秀模型

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 训练参数
MAX_EPOCHS=200
BATCH_SIZE=16
TRAINER="mps"

# 模型列表 - 使用 timm 中表现优秀的模型
MODELS=(
    "resnet18"
    "resnet34"
    "efficientnet_b0"
    "mobilenetv3_large_100"
    "convnext_tiny"
)

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  矩阵化训练脚本${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${GREEN}训练配置:${NC}"
echo -e "  训练轮数: ${YELLOW}$MAX_EPOCHS${NC}"
echo -e "  批次大小: ${YELLOW}$BATCH_SIZE${NC}"
echo -e "  训练器: ${YELLOW}$TRAINER${NC}"
echo -e "  模型数量: ${YELLOW}${#MODELS[@]}${NC}"
echo ""
echo -e "${CYAN}模型列表:${NC}"
for model in "${MODELS[@]}"; do
    echo -e "  - ${YELLOW}$model${NC}"
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
echo -e "${BLUE}  开始矩阵化训练${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# 创建结果文件
RESULTS_FILE="training_results_$(date +%Y%m%d_%H%M%S).txt"
echo "训练结果汇总" > $RESULTS_FILE
echo "训练时间: $(date)" >> $RESULTS_FILE
echo "========================================" >> $RESULTS_FILE
echo "" >> $RESULTS_FILE

# 训练计数器
TOTAL_MODELS=${#MODELS[@]}
CURRENT=0
SUCCESSFUL=0
FAILED=0

# 遍历所有模型进行训练
for model in "${MODELS[@]}"; do
    CURRENT=$((CURRENT + 1))

    echo ""
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}  训练模型 [$CURRENT/$TOTAL_MODELS]: $model${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo ""

    # 首先创建模型配置文件（如果不存在）
    CONFIG_FILE="configs/model/${model}.yaml"

    if [ ! -f "$CONFIG_FILE" ]; then
        echo -e "${YELLOW}创建模型配置: $CONFIG_FILE${NC}"
        cat > "$CONFIG_FILE" << EOF
_target_: src.models.retina_mnist_module.RetinaMNISTLitModule

optimizer:
  _target_: torch.optim.Adam
  _partial_: true
  lr: 0.001
  weight_decay: 0.0

scheduler:
  _target_: torch.optim.lr_scheduler.ReduceLROnPlateau
  _partial_: true
  mode: min
  factor: 0.1
  patience: 10

net:
  _target_: src.models.components.timm_classifier.TimmClassifier
  model_name: $model
  num_classes: 5
  pretrained: false
  drop_rate: 0.0
  drop_path_rate: 0.0

num_classes: 5

compile: false
EOF
    fi

    # 构建训练命令
    TRAIN_CMD="python src/train.py \
        logger=mlflow \
        model=$model \
        trainer=$TRAINER \
        trainer.max_epochs=$MAX_EPOCHS \
        data.batch_size=$BATCH_SIZE \
        tags=\"['retina-mnist', '$model', 'matrix-training']\""

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

        echo -e "${GREEN}✓ $model 训练成功 (耗时: ${MINUTES}m ${SECONDS}s)${NC}"
        echo "$model: 成功 (耗时: ${MINUTES}m ${SECONDS}s)" >> $RESULTS_FILE
        SUCCESSFUL=$((SUCCESSFUL + 1))
    else
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        MINUTES=$((DURATION / 60))
        SECONDS=$((DURATION % 60))

        echo -e "${RED}✗ $model 训练失败 (耗时: ${MINUTES}m ${SECONDS}s)${NC}"
        echo "$model: 失败 (耗时: ${MINUTES}m ${SECONDS}s)" >> $RESULTS_FILE
        FAILED=$((FAILED + 1))
    fi

    echo ""
done

# 训练完成总结
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  训练完成总结${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${GREEN}总模型数: $TOTAL_MODELS${NC}"
echo -e "${GREEN}成功: $SUCCESSFUL${NC}"
echo -e "${RED}失败: $FAILED${NC}"
echo ""
echo -e "${YELLOW}详细结果已保存到: $RESULTS_FILE${NC}"
echo ""

# 添加总结到结果文件
echo "" >> $RESULTS_FILE
echo "========================================" >> $RESULTS_FILE
echo "总结:" >> $RESULTS_FILE
echo "  总模型数: $TOTAL_MODELS" >> $RESULTS_FILE
echo "  成功: $SUCCESSFUL" >> $RESULTS_FILE
echo "  失败: $FAILED" >> $RESULTS_FILE
echo "" >> $RESULTS_FILE
echo "查看结果: http://127.0.0.1:5000" >> $RESULTS_FILE

# 显示结果文件内容
echo -e "${CYAN}训练结果:${NC}"
cat $RESULTS_FILE

echo ""
echo -e "${GREEN}MLflow UI 保持运行${NC}"
echo -e "${GREEN}访问地址: http://127.0.0.1:5000${NC}"
echo -e "${YELLOW}停止命令: kill $MLFLOW_PID${NC}"
echo ""