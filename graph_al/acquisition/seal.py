from graph_al.acquisition.base import BaseAcquisitionStrategy
from graph_al.acquisition.config import AcquireByPredictionAttributeConfig, PredictionAttribute, AcquisitionStrategySEALConfig
from graph_al.data.base import BaseDataset, Dataset
from graph_al.model.base import BaseModel
from graph_al.model.config import ModelConfig
from graph_al.model.prediction import Prediction
from graph_al.acquisition.attribute import AcquisitionStrategyByAttribute
from graph_al.utils.logging import get_logger
from graph_al.model.seal import SEAL

from jaxtyping import Float, Int, jaxtyped, Shaped, Bool
from typeguard import typechecked
from torch import Tensor, Generator
from typing import Tuple, Dict, Any

from torch_geometric.data import Data
import torch

class AcquisitionStrategySEAL(AcquisitionStrategyByAttribute):
    
    """ Strategy for the SEAL model"""
    
    def __init__(self, config: AcquisitionStrategySEALConfig):
        super().__init__(config)
        
    @jaxtyped(typechecker=typechecked)
    def get_attribute(self, prediction: Prediction | None, model: SEAL, dataset: Dataset, generator: Generator, model_config: ModelConfig) -> Shaped[Tensor, 'num_nodes']:
        if prediction is None:
            raise ValueError(f'Can not derive prediction attribute if no prediction is given')
        
        probability_is_labeled = prediction.get_adversarial_is_labeled_probabilities()
        if probability_is_labeled is None:
            raise RuntimeError(f'Can not derive prediction attribute if no is-labeled probability is given')
        probability_is_labeled = probability_is_labeled.mean(0) # ensemble average
        
        predicted_probabilities = prediction.get_probabilities(propagated=True)
        if predicted_probabilities is None:
            raise RuntimeError(f'Can not derive prediction attribute if no prediction probability is given')
        predicted_probabilities = predicted_probabilities.mean(0)
        
        probability_is_labeled[dataset.data.is_pseudo_labeled(predicted_probabilities)] = 1.0
        return probability_is_labeled
        
    @property
    def retrain_after_each_acquisition(self) -> bool | None:
        return False
        
        
