from dataclasses import field, dataclass
from typing import List
from omegaconf import DictConfig, MISSING
from hydra.core.config_store import ConfigStore

from graph_al.evaluation.enum import MetricTemplate, MetricName, DatasetSplit
from graph_al.model.trainer.evaluation.config import TrainerEvaluationConfig
from graph_al.model.trainer.enum import *
from graph_al.model.trainer.optimizer.config import OptimizerConfig

@dataclass
class EarlyStoppingConfig:
    """ Early stopping configuration. """
    
    monitor: MetricTemplate = field(default_factory=lambda: MetricTemplate(name=MetricName.LOSS, dataset_split=DatasetSplit.TRAIN))
    higher_is_better: bool = False
    patience: int = 10
    min_delta: float = 1e-3 # Improvement of less than this value will be discarded
    save_model_state: bool = False


@dataclass
class TrainerConfig:
    """ Base training configuration. """
    
    name: TrainerType = MISSING
    
    progress_bar: bool = False # Progress bar for training a single model
    use_gpu: bool = True
    verbose: bool = False
    evaluation: TrainerEvaluationConfig = field(default_factory=TrainerEvaluationConfig)
    
    
@dataclass
class SGDTrainerConfig(TrainerConfig):
    
    name: TrainerType = TrainerType.SGD
    
    max_epochs: int = 10000
    min_epochs: int = 0
    commit_to_wandb_every_epoch: int | None = None
    log_every_epoch: int | None = 0 # don't log in epochs
    loss: LossFunction = LossFunction.CROSS_ENTROPY
    balanced_loss: bool = True # Account for class imbalance, i.e. weight terms
    balanced_loss_beta: float = 0.999 # tradeoff between no balancing (->0) and inverse class frequency (->1)
    balanced_loss_normalize: bool = True # loss weights for each class sum to num_classes

    logits_propagated: bool = True
    
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    early_stopping: EarlyStoppingConfig | None = None
    
    kl_divergence_loss_weight: float = 1e0

    summary_metrics: List[MetricTemplate] = field(default_factory=list)

@dataclass
class OracleTrainerConfig(TrainerConfig):    
    name: TrainerType = TrainerType.ORACLE    

@dataclass
class SGCTrainerConfig(TrainerConfig):
    name: TrainerType = TrainerType.SGC
    
@dataclass
class GPNTrainerConfig(SGDTrainerConfig):
    name: TrainerType = TrainerType.GPN
    
    loss: LossFunction = LossFunction.NONE # the uce + entropy reg loss is hardcoded into the GPN trainer class
    
    warmup: GPNWarmup = GPNWarmup.FLOW
    warmup_optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    
    flow_lr: float = 1e-2
    flow_weight_decay: float = 0
    entropy_regularization_loss_weight: float = 1e-4 # regularization strength for the entropy of the posterior
    num_warmup_epochs: int = 5
    
@dataclass
class SEALTrainerConfig(SGDTrainerConfig):
    name: TrainerType = TrainerType.SEAL
    discriminator_optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    num_discriminator_epochs: int = 1
    num_samples: int | None = None # How many samples to draw for calculating expectations
    discriminator_supervised_loss_weight: float = 0.6 # alpha in the paper


cs = ConfigStore.instance()
cs.store(name="base", node=TrainerConfig, group='model/trainer')
cs.store(name="base_sgd", node=SGDTrainerConfig, group='model/trainer')
cs.store(name="base_gpn", node=GPNTrainerConfig, group='model/trainer')
cs.store(name="base", node=EarlyStoppingConfig, group='model/trainer/early_stopping')
cs.store(name="base_oracle", node=OracleTrainerConfig, group='model/trainer')
cs.store(name="base_sgc", node=SGCTrainerConfig, group='model/trainer')
cs.store(name="base_seal", node=SEALTrainerConfig, group='model/trainer')
