from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from typing import List

from graph_al.data.config import DataConfig
from graph_al.model.config import ModelConfig
from graph_al.acquisition.config import AcquisitionStrategyConfig
from graph_al.evaluation.config import EvaluationConfig, MetricTemplate
from graph_al.evaluation.enum import DatasetSplit, MetricName

@dataclass
class WandbConfig:
    """ Configuration for weights and biases. """
    id: str | None = None
    entity: str | None = None
    project: str | None = MISSING
    group: str = MISSING
    mode: str | None = None
    name: str | None = None
    dir: str | None = None
    tags: List[str] | None = None
    
    log_internal_dir: str | None = None
    disable: bool = True
    

@dataclass
class Config:
    """ General configuration. """
    
    seed: int | None = None
    output_base_dir: str = MISSING # where to log output files to
    
    progress_bar: bool = True # Progress bar for the acquisition run (i.e. one active learning run)
    progress_bar_metrics: List[MetricTemplate] = field(default_factory=lambda: [MetricTemplate(name=MetricName.ACCURACY, dataset_split=DatasetSplit.VAL)])
    
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    acquisition_strategy: AcquisitionStrategyConfig = field(default_factory=AcquisitionStrategyConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    # Initial acquisition before model training
    initial_acquisition_strategy: AcquisitionStrategyConfig = field(default_factory=AcquisitionStrategyConfig)
    
    retrain_after_acquisition: bool = True
    wandb: WandbConfig = field(default_factory=WandbConfig)
    

    print_summary: bool = False

@dataclass
class OptimizeBestSplitOrderConfig(Config):

    num_samples: int = MISSING # How many orders to generate
    best_split_dir: str = MISSING
    best_split_metric: MetricTemplate = field(default_factory=lambda: MetricTemplate(name=MetricName.ACCURACY, dataset_split=DatasetSplit.VAL))
    
cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)
cs.store(name="base_optimize_best_split_order", node=OptimizeBestSplitOrderConfig)
