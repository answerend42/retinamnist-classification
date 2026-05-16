

from typing import Any, Dict, List, Optional, Tuple
import functools

import hydra
import lightning as L
import rootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

# Fix PyTorch 2.6 checkpoint loading issue
# Disable weights_only for checkpoint loading to avoid compatibility issues
# This is safe for checkpoints created by ourselves

# Monkey patch torch.load to always use weights_only=False
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    # Force weights_only=False regardless of what the caller specifies
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    logger_configs = cfg.get("logger")
    if logger_configs and "mlflow" in logger_configs:
        model_name = cfg.get("model", {}).get("net", {}).get("model_name", "unknown")
        if model_name == "unknown":
            model_name = cfg.get("model", {}).get("_target_", "").split(".")[-1]

        job_num = cfg.get("hydra", {}).get("job", {}).get("num", None)
        if job_num is not None:
            mlflow_run_name = f"{model_name}_e{cfg.trainer.max_epochs}_b{cfg.data.batch_size}_job{job_num}"
        else:
            import time
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            mlflow_run_name = f"{model_name}_e{cfg.trainer.max_epochs}_b{cfg.data.batch_size}_{timestamp}"

        from omegaconf import OmegaConf
        logger_configs = OmegaConf.to_container(logger_configs, resolve=True)
        logger_configs["mlflow"]["run_name"] = mlflow_run_name
        logger_configs = OmegaConf.create(logger_configs)

    logger: List[Logger] = instantiate_loggers(logger_configs)
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log_hyperparameters(object_dict)

    if cfg.get("train"):
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        # 优先使用用户提供的 ckpt_path，否则使用训练过程中的最佳模型
        ckpt_path = cfg.get("ckpt_path")
        if not ckpt_path and trainer.checkpoint_callback:
            ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
