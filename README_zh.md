# RetinaMNIST — 基于深度学习和超参数优化的糖尿病视网膜病变分级

**医学人工智能与机器学习 · 2025-2026 学年第一学期 · 结课项目**

使用 ImageNet 预训练 CNN 与 Optuna 超参数优化，对糖尿病视网膜病变进行五级自动分级。
基于 PyTorch Lightning + Hydra 框架构建。

[English](README.md) · [中文](README_zh.md)

---

## 实验结果

| 模型 | 验证 AUC | 测试 AUC | 测试准确率 | 参数量 | 文件大小 |
|------|---------|---------|-----------|--------|---------|
| **ConvNeXt-Tiny** | 0.8845 | **0.8968** | **0.6800** | 28.6M | 319 MB |
| EfficientNet-B0 | **0.8873** | 0.8815 | 0.6700 | 5.3M | 46 MB |
| ResNet-18 | 0.8619 | 0.7893 | 0.5875 | 11.7M | 128 MB |

*验证 AUC 来自 HPO 最优 trial；测试指标为本机复现结果。*

## 数据集

**RetinaMNIST** 是 MedMNIST 医学影像基准集的一部分。

- 任务：糖尿病视网膜病变五级分级（Grade 0–4）
- 图像尺寸：224×224 RGB
- 训练集：1,080 张 · 验证集：120 张 · 测试集：400 张

## 快速开始

### 环境要求

- Python ≥ 3.12
- macOS (MPS) / Linux / Windows (CUDA)
- [uv](https://docs.astral.sh/uv/) 包管理器

### 1. 安装

```bash
git clone git@github.com:YOUR_USERNAME/retinamnist-classification.git
cd retinamnist-classification
uv sync
source .venv/bin/activate
```

### 2. 下载数据与模型

从百度网盘下载以下文件，放置到对应目录：

| 文件 | 大小 | 放置路径 |
|------|------|---------|
| `retinamnist_224.npz` | 122 MB | `data/retinamnist_224.npz` |
| `efficientnet_b0_best.ckpt` | 46 MB | `best_models/efficientnet_b0_best.ckpt` |
| `convnext_tiny_best.ckpt` | 319 MB | `best_models/convnext_tiny_best.ckpt` |
| `resnet18_best.ckpt` | 128 MB | `best_models/resnet18_best.ckpt` |
| `optuna_master.db` | 176 KB | `optuna_master.db` |
| `mlflow.db` | 13 MB | `mlflow.db` |

> 百度网盘：[data_and_models](https://pan.baidu.com/s/1QJ7LB4FUdBnmkyfUpWEDfA?pwd=78hj)（提取码: 78hj）

### 3. 生成论文图表

```bash
python generate_figures.py   # 约 10 秒，无需 GPU
```

输出（`paper_figures/`）：HPO 优化历史、模型性能对比、训练曲线、混淆矩阵、
多分类 ROC 曲线、分类报告。全部提供 PNG + PDF 双格式。

### 4. 模型评估

```bash
# EfficientNet-B0（体积最小）
python src/train.py experiment=hpo_efficientnet_b0 train=false test=true \
  ckpt_path=best_models/efficientnet_b0_best.ckpt

# ConvNeXt-Tiny（测试集最优）
python src/train.py experiment=hpo_convnext_tiny train=false test=true \
  ckpt_path=best_models/convnext_tiny_best.ckpt

# ResNet-18
python src/train.py experiment=hpo_resnet18 train=false test=true \
  ckpt_path=best_models/resnet18_best.ckpt
```

## 方法

### 模型架构

所有模型基于 timm 库加载 ImageNet 预训练权重，替换分类头适配 5 分类。

| 模型 | 架构 |
|------|------|
| ResNet-18 | 残差网络，含跳跃连接 |
| EfficientNet-B0 | 复合缩放高效网络 |
| ConvNeXt-Tiny | 借鉴 ViT 设计的现代化 CNN |

### 超参数优化

Optuna TPE 采样器，每模型 50 次搜索，优化验证集 AUC。

- 搜索空间：学习率、批量大小、优化器（Adam/AdamW）、权重衰减、dropout rate、drop path rate
- 学习率调度：ReduceLROnPlateau（patience=10, factor=0.1）
- 早停策略：patience=30，监控 val/auc

### 训练配置

- 框架：PyTorch Lightning 2.0 + Hydra 1.3
- 损失函数：CrossEntropyLoss
- 数据增强：RandomHorizontalFlip、RandomRotation、ImageNet 归一化
- 最大训练轮次：100

## 从头训练

### 单模型训练

```bash
python src/train.py experiment=hpo_efficientnet_b0
```

### 完整超参数搜索

```bash
./scripts/run_hpo.sh              # 所有模型（约 6–8 小时）
python src/train.py -m \
  hparams_search=efficientnet_b0_optuna \
  experiment=hpo_efficientnet_b0   # 单模型
```

### 预训练对比实验

```bash
./scripts/train_pretrained_comparison.sh
```

## 实验记录

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# 访问 http://localhost:5000
```

## 项目结构

```
retinamnist-classification/
├── generate_figures.py           # 图表生成
├── pyproject.toml                # 依赖配置
├── uv.lock                       # 依赖锁文件
├── .project-root
│
├── src/                          # 源码
│   ├── train.py                  # 训练入口
│   ├── eval.py                   # 评估入口
│   ├── data/
│   │   └── retina_mnist_datamodule.py  # 数据加载与增强
│   ├── models/
│   │   ├── retina_mnist_module.py      # Lightning 模块
│   │   └── components/
│   │       ├── timm_classifier.py      # timm 模型封装
│   │       ├── simple_dense_net.py
│   │       └── unet_classifier.py
│   └── utils/
│
├── configs/                      # Hydra 配置
│   ├── train.yaml                # 主训练配置
│   ├── experiment/               # 最优超参数预设
│   ├── hparams_search/           # HPO 搜索空间
│   ├── model/                    # 模型配置
│   ├── data/                     # 数据配置
│   ├── trainer/                  # 训练器（CPU/GPU/MPS）
│   └── logger/                   # 日志记录器
│
├── scripts/                      # 运行脚本
│
├── data/                         # 数据集（需下载）
├── best_models/                  # 模型检查点（需下载）
├── optuna_master.db              # HPO 记录（需下载）
└── mlflow.db                     # 实验记录（需下载）
```

## 关键发现

1. **预训练权重至关重要** — ImageNet 预训练相比从头训练提升约 5% AUC。
2. **ConvNeXt-Tiny 泛化最优** — 虽然验证 AUC 略低于 EfficientNet-B0，测试 AUC 达 0.8968。
3. **超参数优化收益显著** — 所有模型均有 2–3% 提升。
4. **EfficientNet-B0 性价比最高** — 仅 5.3M 参数 / 46MB，性能接近 ConvNeXt-Tiny（28.6M / 319MB）。

## 技术栈

| 组件 | 技术 |
|------|------|
| 深度学习框架 | PyTorch 2.0+, PyTorch Lightning 2.0+ |
| 配置管理 | Hydra 1.3 |
| 模型库 | timm (PyTorch Image Models) |
| 超参数优化 | Optuna (TPE Sampler) |
| 实验追踪 | MLflow |
| 评估指标 | torchmetrics, scikit-learn |
| 可视化 | matplotlib, seaborn |
| 依赖管理 | uv |
