#!/usr/bin/env python3
"""
从Optuna数据库获取最佳超参数，并创建不使用预训练权重的实验配置
"""

import sqlite3
import yaml
import os
from pathlib import Path

# 连接到Optuna数据库
db_path = Path(__file__).parent.parent / "optuna_master.db"
conn = sqlite3.connect(str(db_path))
cursor = conn.cursor()

# 定义模型映射
models = {
    "resnet18_hpo": "resnet18",
    "efficientnet_b0_hpo": "efficientnet_b0",
    "convnext_tiny_hpo": "convnext_tiny"
}

# 为每个模型创建配置
for study_name, model_name in models.items():
    # 查询最佳试验
    query = """
    SELECT t.number, v.value
    FROM trials t
    JOIN studies s ON t.study_id = s.study_id
    JOIN trial_values v ON t.trial_id = v.trial_id
    WHERE s.study_name = ?
      AND t.state = 'COMPLETE'
    ORDER BY v.value DESC
    LIMIT 1
    """
    cursor.execute(query, (study_name,))
    best_trial = cursor.fetchone()

    if not best_trial:
        print(f"Warning: No completed trials found for {study_name}")
        continue

    trial_num, best_auc = best_trial

    # 查询该试验的参数（需要解码）
    # Optuna存储的是编码值，需要从distribution_json解码
    query = """
    SELECT p.param_name, p.param_value, p.distribution_json
    FROM trial_params p
    JOIN trials t ON p.trial_id = t.trial_id
    JOIN studies s ON t.study_id = s.study_id
    WHERE s.study_name = ? AND t.number = ?
    """
    cursor.execute(query, (study_name, trial_num))
    params_raw = cursor.fetchall()

    print(f"\n{study_name}:")
    print(f"  Best trial: {trial_num}, AUC: {best_auc:.4f}")

    # 创建实验配置
    config = {
        "_target_": "src.models.retina_mnist_module.RetinaMNISTLitModule",
        "optimizer": {
            "_target_": "torch.optim.AdamW",
            "_partial_": True,
            "lr": 0.001,  # 默认值，会被下面的值覆盖
            "weight_decay": 0.0
        },
        "scheduler": {
            "_target_": "torch.optim.lr_scheduler.ReduceLROnPlateau",
            "_partial_": True,
            "mode": "min",
            "factor": 0.1,
            "patience": 10
        },
        "net": {
            "_target_": "src.models.components.timm_classifier.TimmClassifier",
            "model_name": model_name,
            "num_classes": 5,
            "pretrained": False,  # 关键：不使用预训练
            "drop_rate": 0.0,
            "drop_path_rate": 0.0
        },
        "num_classes": 5,
        "compile": False
    }

    # 解码参数并应用到配置
    for param_name, param_value, dist_json in params_raw:
        import json
        dist = json.loads(dist_json)

        print(f"    {param_name}: {param_value} ({dist.get('name', 'unknown')})")

        # 根据参数名称应用到配置
        if param_name == "model.optimizer.lr":
            config["optimizer"]["lr"] = float(param_value)
        elif param_name == "model.optimizer.weight_decay":
            # weight_decay是choice选择，需要解码
            if dist["name"] == "CategoricalDistribution":
                choices = dist["choices"]
                idx = int(param_value)
                config["optimizer"]["weight_decay"] = float(choices[idx])
            else:
                config["optimizer"]["weight_decay"] = float(param_value)
        elif param_name == "data.batch_size":
            if dist["name"] == "CategoricalDistribution":
                choices = dist["choices"]
                idx = int(param_value)
                batch_size = int(choices[idx])
            else:
                batch_size = int(param_value)
        elif param_name == "model.net.drop_rate":
            config["net"]["drop_rate"] = float(param_value)
        elif param_name == "model.net.drop_path_rate":
            config["net"]["drop_path_rate"] = float(param_value)
        elif param_name == "model.scheduler.patience":
            if dist["name"] == "CategoricalDistribution":
                choices = dist["choices"]
                idx = int(param_value)
                config["scheduler"]["patience"] = int(choices[idx])
        elif param_name == "model.scheduler.factor":
            if dist["name"] == "CategoricalDistribution":
                choices = dist["choices"]
                idx = int(param_value)
                config["scheduler"]["factor"] = float(choices[idx])

    print(f"\n  Decoded config:")
    print(f"    lr: {config['optimizer']['lr']}")
    print(f"    weight_decay: {config['optimizer']['weight_decay']}")
    print(f"    batch_size: {batch_size}")
    print(f"    drop_rate: {config['net']['drop_rate']}")
    print(f"    drop_path_rate: {config['net']['drop_path_rate']}")
    print(f"    scheduler.patience: {config['scheduler']['patience']}")
    print(f"    scheduler.factor: {config['scheduler']['factor']}")

conn.close()

print("\n配置信息已获取，接下来手动创建实验配置文件...")
