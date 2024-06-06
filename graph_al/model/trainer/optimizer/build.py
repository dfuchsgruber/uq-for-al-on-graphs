import torch

from typeguard import typechecked

from graph_al.model.trainer.optimizer.enum import *
from graph_al.model.trainer.optimizer.config import OptimizerConfig

@typechecked
def get_optimizer(config: OptimizerConfig, parameters) -> torch.optim.Optimizer:
    """ Gets the optimizer. """
    match config.type_:
        case OptimizerType.ADAM:
            config_adam: OptimizerConfigAdam = config # type: ignore
            return torch.optim.Adam(parameters, lr=config_adam.lr, weight_decay=config_adam.weight_decay)
        case type_:
            raise ValueError(f'Unsupported optimizer {type_}')