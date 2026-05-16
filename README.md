# RetinaMNIST — Diabetic Retinopathy Grading with Deep Learning and HPO

**Medical AI & Machine Learning · 2025-2026 Semester 1 · Final Project**

Automated five-level grading of diabetic retinopathy using ImageNet-pretrained CNNs
and Optuna hyperparameter optimization. Built with PyTorch Lightning + Hydra.

[English](README.md) · [中文](README_zh.md)

---

## Results

| Model | Val AUC | Test AUC | Test Acc | Params | Size |
|-------|---------|----------|----------|--------|------|
| **ConvNeXt-Tiny** | 0.8845 | **0.8968** | **0.6800** | 28.6M | 319 MB |
| EfficientNet-B0 | **0.8873** | 0.8815 | 0.6700 | 5.3M | 46 MB |
| ResNet-18 | 0.8619 | 0.7893 | 0.5875 | 11.7M | 128 MB |

*Val AUC from HPO best trial; test metrics reproduced locally.*

## Dataset

**RetinaMNIST** is a medical imaging benchmark from the MedMNIST collection.

- Task: 5-level DR severity grading (Grade 0–4)
- Image size: 224×224 RGB
- Train: 1,080 · Val: 120 · Test: 400

## Quick Start

### Prerequisites

- Python ≥ 3.12
- macOS (MPS) / Linux / Windows (CUDA)
- [uv](https://docs.astral.sh/uv/) package manager

### 1. Install

```bash
git clone git@github.com:YOUR_USERNAME/retinamnist-classification.git
cd retinamnist-classification
uv sync
source .venv/bin/activate
```

### 2. Download data & models

Download from Baidu Netdisk and place files as follows:

| File | Size | Destination |
|------|------|-------------|
| `retinamnist_224.npz` | 122 MB | `data/retinamnist_224.npz` |
| `efficientnet_b0_best.ckpt` | 46 MB | `best_models/efficientnet_b0_best.ckpt` |
| `convnext_tiny_best.ckpt` | 319 MB | `best_models/convnext_tiny_best.ckpt` |
| `resnet18_best.ckpt` | 128 MB | `best_models/resnet18_best.ckpt` |
| `optuna_master.db` | 176 KB | `optuna_master.db` |
| `mlflow.db` | 13 MB | `mlflow.db` |

> Baidu Netdisk: [data_and_models](https://pan.baidu.com/s/1QJ7LB4FUdBnmkyfUpWEDfA?pwd=78hj) (pwd: 78hj)

### 3. Generate paper figures

```bash
python generate_figures.py   # ~10s, no GPU needed
```

Output (`paper_figures/`): HPO optimization history, model comparison, training curves,
confusion matrices, multi-class ROC curves, classification report. All in PNG + PDF.

### 4. Evaluate models

```bash
# EfficientNet-B0 (smallest)
python src/train.py experiment=hpo_efficientnet_b0 train=false test=true \
  ckpt_path=best_models/efficientnet_b0_best.ckpt

# ConvNeXt-Tiny (best test AUC)
python src/train.py experiment=hpo_convnext_tiny train=false test=true \
  ckpt_path=best_models/convnext_tiny_best.ckpt

# ResNet-18
python src/train.py experiment=hpo_resnet18 train=false test=true \
  ckpt_path=best_models/resnet18_best.ckpt
```

## Methodology

### Models

All models use timm with ImageNet pretrained weights, classifier head replaced for 5 classes.

| Model | Architecture |
|-------|-------------|
| ResNet-18 | Residual network with skip connections |
| EfficientNet-B0 | Compound-scaled efficient CNN |
| ConvNeXt-Tiny | Modernized CNN with ViT-inspired design |

### Hyperparameter Optimization

Optuna with TPE sampler, 50 trials per model, optimizing validation AUC.

- Search space: learning rate, batch size, optimizer (Adam/AdamW), weight decay, dropout rate, drop path rate
- LR scheduler: ReduceLROnPlateau (patience=10, factor=0.1)
- Early stopping: patience=30, monitor val/auc

### Training

- Framework: PyTorch Lightning 2.0 + Hydra 1.3
- Loss: CrossEntropyLoss
- Augmentation: RandomHorizontalFlip, RandomRotation, ImageNet normalization
- Max epochs: 100

## Training from Scratch

### Single model

```bash
python src/train.py experiment=hpo_efficientnet_b0
```

### Full HPO

```bash
./scripts/run_hpo.sh              # all models (~6–8h)
python src/train.py -m \
  hparams_search=efficientnet_b0_optuna \
  experiment=hpo_efficientnet_b0   # single model
```

### Pretrained vs. scratch comparison

```bash
./scripts/train_pretrained_comparison.sh
```

## Experiment Tracking

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Open http://localhost:5000
```

## Project Structure

```
retinamnist-classification/
├── generate_figures.py           # Figure generation
├── pyproject.toml                # Dependencies
├── uv.lock                       # Lock file
├── .project-root
│
├── src/                          # Source code
│   ├── train.py                  # Training entry
│   ├── eval.py                   # Evaluation entry
│   ├── data/
│   │   └── retina_mnist_datamodule.py  # Data loading & augmentation
│   ├── models/
│   │   ├── retina_mnist_module.py      # Lightning module
│   │   └── components/
│   │       ├── timm_classifier.py      # timm model wrapper
│   │       ├── simple_dense_net.py
│   │       └── unet_classifier.py
│   └── utils/
│
├── configs/                      # Hydra configs
│   ├── train.yaml                # Main training config
│   ├── experiment/               # Optimal HPO presets
│   ├── hparams_search/           # HPO search spaces
│   ├── model/                    # Model configs
│   ├── data/                     # Data configs
│   ├── trainer/                  # Trainer (CPU/GPU/MPS)
│   └── logger/                   # Loggers (MLflow, WandB, etc.)
│
├── scripts/                      # Shell scripts
│
├── data/                         # Dataset (download required)
├── best_models/                  # Checkpoints (download required)
├── optuna_master.db              # HPO records (download required)
└── mlflow.db                     # Experiment records (download required)
```

## Key Findings

1. **Pretrained weights are critical** — ImageNet pretraining boosts AUC by ~5% over training from scratch.
2. **ConvNeXt-Tiny generalizes best** — top test AUC (0.8968) despite slightly lower validation AUC than EfficientNet-B0.
3. **HPO yields consistent gains** — 2–3% improvement across all models.
4. **EfficientNet-B0 best value** — only 5.3M params / 46MB, close to ConvNeXt-Tiny (28.6M / 319MB) in performance.

## Tech Stack

| Component | Technology |
|-----------|------------|
| Deep learning | PyTorch 2.0+, PyTorch Lightning 2.0+ |
| Config management | Hydra 1.3 |
| Model zoo | timm (PyTorch Image Models) |
| HPO | Optuna (TPE Sampler) |
| Experiment tracking | MLflow |
| Metrics | torchmetrics, scikit-learn |
| Visualization | matplotlib, seaborn |
| Package manager | uv |
