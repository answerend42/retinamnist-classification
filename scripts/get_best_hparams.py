#!/usr/bin/env python3
"""
从Optuna数据库获取最佳超参数（使用optuna API）
"""

import optuna

# 连接到Optuna数据库
storage = "sqlite:///optuna_master.db"

studies = {
    "ResNet-18": "resnet18_hpo",
    "EfficientNet-B0": "efficientnet_b0_hpo",
    "ConvNeXt-Tiny": "convnext_tiny_hpo"
}

for model_name, study_name in studies.items():
    study = optuna.load_study(study_name=study_name, storage=storage)
    best_trial = study.best_trial

    print(f"\n{model_name} ({study_name}):")
    print(f"  Best Trial: {best_trial.number}")
    print(f"  Best AUC: {best_trial.value:.6f}")
    print(f"  Best Hyperparameters:")

    for param_name, param_value in best_trial.params.items():
        print(f"    {param_name}: {param_value}")

print("\n\n现在使用这些超参数创建实验配置文件...")
