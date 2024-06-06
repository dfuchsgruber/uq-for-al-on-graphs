import torch

from typeguard import typechecked

from graph_al.model.trainer.config import (TrainerConfig, SGDTrainerConfig, 
                                           GPNTrainerConfig, OracleTrainerConfig, SGCTrainerConfig,
                                           SEALTrainerConfig
                                           )
from graph_al.model.base import BaseModel
from graph_al.data.base import Dataset
from graph_al.model.trainer.base import BaseTrainer
from graph_al.model.trainer.sgd import SGDTrainer
from graph_al.model.trainer.gpn import GPNTrainer
from graph_al.model.trainer.gt_trainer import OracleTrainer
from graph_al.model.trainer.sgc import SGCTrainer
from graph_al.model.trainer.seal import SEALTrainer
from graph_al.model.trainer.enum import *

@typechecked
def get_trainer(config: TrainerConfig, model: BaseModel, dataset: Dataset, generator: torch.Generator) -> BaseTrainer:
    """ Gets the trainer. """
    match config.name:
        case SGDTrainerConfig.name:
            trainer = SGDTrainer(config, model, dataset, generator) # type: ignore
        case GPNTrainerConfig.name:
            trainer = GPNTrainer(config, model, dataset, generator) # type: ignore
        case OracleTrainerConfig.name:
            trainer = OracleTrainer(config, model, dataset, generator) # type: ignore
        case SGCTrainerConfig.name:
            trainer = SGCTrainer(config, model, dataset, generator) # type: ignore
        case SEALTrainerConfig.name:
            trainer = SEALTrainer(config, model, dataset, generator) # type: ignore
        case _:
            raise ValueError(f'Unsupported trainer type {config.name}')
    return trainer