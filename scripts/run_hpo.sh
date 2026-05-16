#!/bin/bash

# 超参数优化 (HPO) 脚本
# 使用 Hydra Optuna Sweeper 自动搜索最优超参数

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# 模型列表 - 选择3个最优模型进行HPO
MODELS=(
    "resnet18_optuna:hpo_resnet18"
    "efficientnet_b0_optuna:hpo_efficientnet_b0"
    "convnext_tiny_optuna:hpo_convnext_tiny"
)

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  超参数优化 (HPO) 实验${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${GREEN}实验设计:${NC}"
echo -e "  使用 Hydra Optuna Sweeper 进行贝叶斯优化"
echo -e "  每个模型运行 ${YELLOW}15 个 trials${NC}"
echo -e "  优化目标: ${YELLOW}val/auc (最大化)${NC}"
echo -e "  最大训练轮数: ${YELLOW}100 epochs${NC}"
echo -e "  早停耐心: ${YELLOW}30 epochs${NC}"
echo -e "  ${CYAN}预计总时间: 8-10 小时${NC}"
echo ""
echo -e "${GREEN}搜索空间:${NC}"
echo -e "  - Learning Rate: 1e-5 ~ 1e-2 (log scale)"
echo -e "  - Weight Decay: [0.0, 1e-5, 1e-4, 1e-3, 1e-2]"
echo -e "  - Batch Size: [16, 32, 64]"
echo -e "  - Dropout Rate: 0.0 ~ 0.4"
echo -e "  - Drop Path Rate: 0.0 ~ 0.3"
echo -e "  - Scheduler Patience: [5, 10, 15, 20]"
echo -e "  - Scheduler Factor: [0.1, 0.2, 0.5]"
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

# 检查并启动 Optuna Dashboard
echo ""
echo -e "${YELLOW}检查 Optuna Dashboard 状态...${NC}"
pkill -f "optuna-dashboard" 2>/dev/null || true
sleep 1

if lsof -ti:8080 > /dev/null 2>&1; then
    echo -e "${YELLOW}清理端口 8080...${NC}"
    lsof -ti:8080 | xargs kill -9 2>/dev/null || true
    sleep 2
fi

echo -e "${GREEN}启动 Optuna Dashboard...${NC}"
optuna-dashboard sqlite:///optuna_master.db > /tmp/optuna_dashboard.log 2>&1 &
OPTUNA_PID=$!
sleep 3

if ps -p $OPTUNA_PID > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Optuna Dashboard 已启动 (PID: $OPTUNA_PID)${NC}"
    echo -e "${GREEN}✓ 访问地址: http://127.0.0.1:8080${NC}"
else
    echo -e "${YELLOW}⚠ Optuna Dashboard 启动失败（可能未安装）${NC}"
    echo -e "${YELLOW}  安装命令: pip install optuna-dashboard${NC}"
    OPTUNA_PID=""
fi

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  开始 HPO 实验${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# 创建结果文件
RESULTS_FILE="hpo_results_$(date +%Y%m%d_%H%M%S).txt"
echo "超参数优化 (HPO) 实验结果" > $RESULTS_FILE
echo "实验时间: $(date)" >> $RESULTS_FILE
echo "========================================" >> $RESULTS_FILE
echo "" >> $RESULTS_FILE

# 实验计数器
TOTAL_MODELS=${#MODELS[@]}
CURRENT=0
SUCCESSFUL=0
FAILED=0

# 遍历所有模型进行HPO
for model_pair in "${MODELS[@]}"; do
    # 解析模型名称
    hparams_search=$(echo $model_pair | cut -d: -f1)
    experiment=$(echo $model_pair | cut -d: -f2)

    CURRENT=$((CURRENT + 1))

    echo ""
    echo -e "${MAGENTA}========================================${NC}"
    echo -e "${MAGENTA}  HPO 实验 [$CURRENT/$TOTAL_MODELS]${NC}"
    echo -e "${MAGENTA}  模型: $experiment${NC}"
    echo -e "${MAGENTA}========================================${NC}"
    echo ""

    echo "" >> $RESULTS_FILE
    echo "模型: $experiment" >> $RESULTS_FILE
    echo "----------------------------------------" >> $RESULTS_FILE

    # 构建训练命令
    TRAIN_CMD="python src/train.py -m hparams_search=$hparams_search experiment=$experiment"

    echo -e "${BLUE}执行命令:${NC}"
    echo -e "${YELLOW}$TRAIN_CMD${NC}"
    echo ""

    # 记录开始时间
    START_TIME=$(date +%s)

    # 执行HPO
    if eval $TRAIN_CMD; then
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        HOURS=$((DURATION / 3600))
        MINUTES=$(((DURATION % 3600) / 60))
        SECONDS=$((DURATION % 60))

        echo -e "${GREEN}✓ HPO 成功 (耗时: ${HOURS}h ${MINUTES}m ${SECONDS}s)${NC}"
        echo "  状态: 成功 (${HOURS}h ${MINUTES}m ${SECONDS}s)" >> $RESULTS_FILE
        SUCCESSFUL=$((SUCCESSFUL + 1))
    else
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        HOURS=$((DURATION / 3600))
        MINUTES=$(((DURATION % 3600) / 60))
        SECONDS=$((DURATION % 60))

        echo -e "${RED}✗ HPO 失败 (耗时: ${HOURS}h ${MINUTES}m ${SECONDS}s)${NC}"
        echo "  状态: 失败 (${HOURS}h ${MINUTES}m ${SECONDS}s)" >> $RESULTS_FILE
        FAILED=$((FAILED + 1))
    fi

    echo ""
done

# 实验完成总结
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  HPO 实验完成总结${NC}"
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
echo "查看结果:" >> $RESULTS_FILE
echo "  1. MLflow UI: http://127.0.0.1:5000" >> $RESULTS_FILE
echo "  2. Optuna 数据库: optuna_study.db" >> $RESULTS_FILE
echo "" >> $RESULTS_FILE
echo "分析最优参数:" >> $RESULTS_FILE
echo "  在 MLflow UI 中按 val/auc 排序，查看 Top 1 实验的超参数配置" >> $RESULTS_FILE

# 显示结果文件内容
echo -e "${CYAN}实验结果:${NC}"
cat $RESULTS_FILE

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  如何分析 HPO 结果${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "1. 访问 MLflow UI: ${YELLOW}http://127.0.0.1:5000${NC}"
echo -e "2. 按 ${YELLOW}val/auc${NC} 指标排序，找到最优实验"
echo -e "3. 查看最优实验的超参数配置"
echo -e ""
if [ -n "$OPTUNA_PID" ]; then
    echo -e "4. 访问 Optuna Dashboard: ${YELLOW}http://127.0.0.1:8080${NC}"
    echo -e "   - 左侧菜单切换不同模型的 studies"
    echo -e "   - 查看 Hyperparameter Importance (参数重要性)"
    echo -e "   - 查看 Slice Plot (单参数影响)"
    echo -e "   - 查看 Optimization History (优化历史)"
else
    echo -e "4. ${YELLOW}Optuna Dashboard 未启动${NC}"
    echo -e "   手动启动: ${CYAN}optuna-dashboard sqlite:///optuna_master.db${NC}"
fi
echo -e ""
echo -e "${YELLOW}论文素材获取:${NC}"
echo -e "  1. MLflow Parallel Coordinates Plot (参数对比图)"
echo -e "  2. Optuna Hyperparameter Importance (参数重要性分析)"
echo -e "  3. Optuna Slice Plot (单参数影响曲线)"
echo -e "  4. Optuna Optimization History (优化历史)"
echo -e "  5. Optuna Study Comparison (跨模型对比)"
echo ""
echo -e "${GREEN}Dashboard 保持运行${NC}"
if [ -n "$OPTUNA_PID" ]; then
    echo -e "${YELLOW}停止命令: kill $MLFLOW_PID $OPTUNA_PID${NC}"
else
    echo -e "${YELLOW}停止命令: kill $MLFLOW_PID${NC}"
fi
echo ""
