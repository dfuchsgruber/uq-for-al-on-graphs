import wandb
from pathlib import Path
import os
from typeguard import typechecked
from dataclasses import fields

from graph_al.config import Config, WandbConfig


_metrics_dir: Path | None = None

@typechecked
def wandb_get_metrics_dir(config: Config, dirname: str = 'metrics', makedirs: bool=True) -> Path | None:
    """Gets a directory to which to log metrics to. Those will be synced to the server after wandb.finish

    Args:
        dirname (str, optional): name of the directory to create. Defaults to 'metrics'.
        makedirs (bool, optional): whether to make the directory if its not existent yet. Defaults to True.

    Returns:
        Path | None: the path
    """
    if wandb.run is not None:
        path = Path(wandb.run.dir) / dirname
    else:
        global _metrics_dir
        if _metrics_dir is None:
            from datetime import datetime
            from uuid import uuid4
            # Create a logging directory from the wandb config
            assert config.wandb.dir is not None
            _metrics_dir = Path(config.wandb.dir) / f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_{uuid4()}'
        path = _metrics_dir
    if makedirs:
        os.makedirs(path, exist_ok=True)
    return path

def wandb_initialize(config: WandbConfig):
    """ Initializes a wandb run

    Args:
        config (WandbConfig): the configuration with which to init

    Returns:
        Run: the wandb run
    """
    non_init_keys = ['log_internal_dir', 'disable']
    init_kwargs = {field.name : getattr(config, field.name) for field in fields(config)
                   if field.name not in non_init_keys}
    if config.dir is not None:
        os.makedirs(config.dir, exist_ok=True)
    wandb_run = wandb.init(**init_kwargs, resume=(config.mode == "online") and "allow",
                           settings=wandb.Settings(
                               log_internal=str(config.log_internal_dir)
                               )
                           )
    return wandb_run
