from dataclasses import field, dataclass
from hydra.core.config_store import ConfigStore

from graph_al.model.enum import PredictionAttribute

@dataclass
class TrainerEvaluationECEConfig:
    """ Configuration for evaluate ECE. """
    num_bins : int = 20
    eps: float = 1e-12

@dataclass
class TrainerEvaluationUncertaintyProxy:
    attribute: PredictionAttribute = PredictionAttribute.MAX_SCORE
    higher_is_certain: bool = True

@dataclass
class TrainerEvaluationConfig:
    """ Evaluation for after model training is done. """

    ece: TrainerEvaluationECEConfig = field(default_factory=TrainerEvaluationECEConfig)
    
cs = ConfigStore.instance()
cs.store(name="base_trainer_evaluation", node=TrainerEvaluationConfig, group='model/trainer/evaluation')
cs.store(name="base_ece", node=TrainerEvaluationECEConfig, group='model/trainer/evaluation/ece')
