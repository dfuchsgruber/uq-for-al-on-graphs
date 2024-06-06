# Code by https://github.com/martenlienen?tab=repositories

from dataclasses import asdict
import inspect
import logging

import rich
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from rich.syntax import Syntax
import wandb
from typeguard import typechecked
from typing import Any, Dict, Iterable

def get_logger():
    caller = inspect.stack()[1]
    module = inspect.getmodule(caller.frame)
    logger_name = None
    if module is not None:
        logger_name = module.__name__.split(".")[-1]
    return logging.getLogger(logger_name)


def print_config(config: DictConfig) -> None:
    content = OmegaConf.to_yaml(config, resolve=True)
    rich.print(Syntax(content, "yaml"))

def print_table(data: Dict[Any, Iterable[float | None]], title: str | None = None):
    """ Prints some data over multiple runs as a table. """
    import rich
    from rich.table import Table
    import torch
    import numpy as np
    
    table = Table('Metric', 'Mean', 'Std', title=title)
    for key in sorted(data.keys(), key=str):
        values = data[key]
        values = [v.item() if isinstance(v, torch.Tensor) and len(v.size()) == 0 else v for v in values]
        values = [float(value) for value in values if isinstance(value, float)]
        table.add_row(str(key), f'{np.mean(values) if len(values) > 0 else np.nan:.2f}', 
                      f'{np.std(values) if len(values) > 1 else np.nan:.2f}')
    rich.print(table)



def count_params(model: nn.Module):
    return {
        "params-total": sum(p.numel() for p in model.parameters()),
        "params-trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "params-not-trainable": sum(
            p.numel() for p in model.parameters() if not p.requires_grad
        ),
    }
    
@typechecked
def log_hyperparameters(
    config: Any,
    model: nn.Module,
):
    hparams: Dict = OmegaConf.to_container(OmegaConf.create(asdict(config))) # type: ignore
    hparams.setdefault("model", {}).update(count_params(model))

    if wandb.run is None:
        get_logger().warn(f'Could not log hyperparameters because there is no active wandb run.')
        return
        wandb.run.config.update(hparams, allow_val_change=True)