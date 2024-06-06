from dataclasses import field, dataclass
from omegaconf import DictConfig, MISSING
from hydra.core.config_store import ConfigStore

from graph_al.model.trainer.optimizer.enum import *

@dataclass
class OptimizerConfig:
    type_: OptimizerType = MISSING
    
@dataclass
class OptimizerConfigAdam(OptimizerConfig):
    type_: OptimizerType = OptimizerType.ADAM
    lr: float = 1e-3
    weight_decay: float = 1e-4
    
cs = ConfigStore.instance()
cs.store(name="base_adam", node=OptimizerConfigAdam, group='model/trainer/optimizer')