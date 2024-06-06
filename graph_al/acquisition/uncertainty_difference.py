
from graph_al.acquisition.config import AcquisitionStrategyUncertaintyDifferenceConfig
from graph_al.data.base import Dataset
from graph_al.model.base import BaseModel
from graph_al.model.config import ModelConfig
from graph_al.model.prediction import Prediction
from graph_al.acquisition.attribute import AcquisitionStrategyByAttribute

import torch

from jaxtyping import jaxtyped, Shaped
from typeguard import typechecked
from torch import Tensor, Generator

class AcquisitionStrategyUncertaintyDifference(AcquisitionStrategyByAttribute):
    
    """ Strategy for the SEAL model"""
    
    def __init__(self, config: AcquisitionStrategyUncertaintyDifferenceConfig):
        super().__init__(config)
        self.combine = config.combine
        self.average = config.average
        
    @jaxtyped(typechecker=typechecked)
    def get_attribute(self, prediction: Prediction | None, model: BaseModel, dataset: Dataset, generator: Generator, model_config: ModelConfig) -> Shaped[Tensor, 'num_nodes']:
        if prediction is None:
            raise ValueError(f'Can not derive prediction attribute if no prediction is given')
        
        probabilities = prediction.get_probabilities(propagated=True)
        probabilities_unpropagated = prediction.get_probabilities(propagated=False)
        if probabilities is None or probabilities_unpropagated is None:
            raise RuntimeError(f'Can not derive prediction attribute if no prediction probability is given')
        
        predicted_labels = probabilities.mean(0).argmax(-1)
        match self.combine:
            case 'ratio':
                uncertainty = (probabilities_unpropagated / probabilities).mean(0)
            case 'difference':
                uncertainty = (probabilities_unpropagated - probabilities).mean(0)
            case _:
                raise ValueError(f'Unknown combine method {self.combine}')
        
        match self.average:
            case 'prediction':
                uncertainty = uncertainty[torch.arange(predicted_labels.size(0)), predicted_labels]
            case 'average':
                uncertainty = uncertainty.mean(-1)
            case _:
                raise ValueError(f'Unknown average method {self.average}')
        return uncertainty
        
        
