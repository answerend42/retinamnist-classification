
#!/bin/bash

# Retina MNIST 训练启动脚本
# 自动启动 MLflow UI 和训练任务

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 默认参数
MODEL="resnet18"
MAX_EPOCHS=200
BATCH_SIZE=16
TRAINER="mps"
KEEP_MLFLOW=false

# 打印帮助信息
print_help() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -m, --model MODEL        ResNet 模型 (默认: resnet18)"
    echo "                           可选: resnet18, resnet34"
    echo "  -e, --epochs EPOCHS      训练轮数 (默认: 200)"
    echo "  -b, --batch-size SIZE    批次大小 (默认: 16)"
    echo "  -t, --trainer TRAINER    训练器类型 (默认: mps)"
    echo "                           可选: mps, gpu, default"
    echo "  -k, --keep-mlflow        训练结束后保持 MLflow UI 运行"
    echo "  -h, --help               显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0                                    # ResNet-18, 默认参数"
    echo "  $0 -m resnet34 -e 100                # ResNet-34, 100 epochs"
    echo "  $0 -m resnet18 -b 8 -k               # ResNet-18, batch=8, 保持 MLflow"
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -e|--epochs)
            MAX_EPOCHS="$2"
            shift 2
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -t|--trainer)
            TRAINER="$2"
            shift 2
            ;;
        -k|--keep-mlflow)
            KEEP_MLFLOW=true
            shift
            ;;
        -h|--help)
            print_help
            exit 0
            ;;
        *)
            echo -e "${RED}错误: 未知参数 $1${NC}"
            print_help
            exit 1
            ;;
    esac
done

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Retina MNIST 训练启动脚本${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${GREEN}配置信息:${NC}"
echo -e "  模型: ${YELLOW}$MODEL${NC}"
echo -e "  训练轮数: ${YELLOW}$MAX_EPOCHS${NC}"
echo -e "  批次大小: ${YELLOW}$BATCH_SIZE${NC}"
echo -e "  训练器: ${YELLOW}$TRAINER${NC}"
echo ""

# 检查虚拟环境
if [ ! -d ".venv" ]; then
    echo -e "${RED}错误: 未找到虚拟环境 .venv${NC}"
    echo -e "${YELLOW}请先运行: uv sync${NC}"
    exit 1
fi

# 激活虚拟环���
echo -e "${GREEN}激活虚拟环境...${NC}"
source .venv/bin/activate

# 检查 MLflow UI 是否已经在运行
echo -e "${YELLOW}检查 MLflow UI 状态...${NC}"

# 先尝试清理可能存在的僵尸进程
pkill -f "mlflow ui" 2>/dev/null || true
sleep 1

# 检查端口是否被占用
if lsof -ti:5000 > /dev/null 2>&1; then
    echo -e "${RED}端口 5000 仍被占用${NC}"
    echo -e "${YELLOW}尝试强制清理...${NC}"
    lsof -ti:5000 | xargs kill -9 2>/dev/null || true
    sleep 2
fi

# 启动 MLflow UI
echo -e "${GREEN}启动 MLflow UI...${NC}"
mlflow ui --backend-store-uri sqlite:///mlflow.db > /tmp/mlflow_ui.log 2>&1 &
MLFLOW_PID=$!
MLFLOW_STARTED_BY_SCRIPT=true

# 等待 MLflow UI 启动
echo -e "${YELLOW}等待 MLflow UI 启动...${NC}"
sleep 5

# 检查 MLflow UI 是否成功启动
if ps -p $MLFLOW_PID > /dev/null 2>&1; then
    echo -e "${GREEN}✓ MLflow UI 已启动 (PID: $MLFLOW_PID)${NC}"
    echo -e "${GREEN}✓ 访问地址: http://127.0.0.1:5000${NC}"
else
    echo -e "${RED}✗ MLflow UI 启动失败${NC}"
    echo -e "${YELLOW}查看日志: cat /tmp/mlflow_ui.log${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  开始训练${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# 构建训练命令
TRAIN_CMD="python src/train.py \
    logger=mlflow \
    model=$MODEL \
    trainer=$TRAINER \
    trainer.max_epochs=$MAX_EPOCHS \
    data.batch_size=$BATCH_SIZE \
    tags=\"['retina-mnist', '$MODEL', 'auto-run']\""

echo -e "${BLUE}执行命令:${NC}"
echo -e "${YELLOW}$TRAIN_CMD${NC}"
echo ""

# 执行训练
eval $TRAIN_CMD
TRAIN_EXIT_CODE=$?

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  训练完成${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ 训练成功完成${NC}"
else
    echo -e "${RED}✗ 训练失败 (退出码: $TRAIN_EXIT_CODE)${NC}"
fi

# 处理 MLflow UI
if [ "$MLFLOW_STARTED_BY_SCRIPT" = true ]; then
    if [ "$KEEP_MLFLOW" = true ]; then
        echo -e "${YELLOW}MLflow UI 保持运行 (PID: $MLFLOW_PID)${NC}"
        echo -e "${YELLOW}访问地址: http://127.0.0.1:5000${NC}"
        echo -e "${YELLOW}停止命令: kill $MLFLOW_PID${NC}"
    else
        echo -e "${YELLOW}关闭 MLflow UI...${NC}"
        kill $MLFLOW_PID 2>/dev/null || true
        echo -e "${GREEN}✓ MLflow UI 已关闭${NC}"
    fi
else
    echo -e "${YELLOW}MLflow UI 由外部启动，保持运行${NC}"
    echo -e "${YELLOW}访问地址: http://127.0.0.1:5000${NC}"
fi

echo ""
echo -e "${BLUE}查看训练日志: logs/train/runs/$(ls -t logs/train/runs/ | head -1)${NC}"
echo ""

exit $TRAIN_EXIT_CODE