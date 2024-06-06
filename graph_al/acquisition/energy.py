from graph_al.acquisition.config import AcquireByLogitEnergyConfig
from graph_al.data.base import Dataset
from graph_al.model.base import BaseModel
from graph_al.model.prediction import Prediction
from graph_al.acquisition.prediction_attribute import AcquisitionStrategyByPredictionAttribute

from jaxtyping import Shaped, jaxtyped
from typeguard import typechecked
from torch import Tensor

import torch

class AcquisitionStrategyByLogitEnergy(AcquisitionStrategyByPredictionAttribute):
    
    """ Acquisition strategy that picks samples with high logit energy scores:
    E[logits] = - t * log[ sum_i exp(logits_i / t)] where t is a temperature parameter """
    
    def __init__(self, config: AcquireByLogitEnergyConfig):
        super().__init__(config)
        self.temperature = config.temperature
        
    @jaxtyped(typechecker=typechecked)
    def get_attribute(self, prediction: Prediction | None, model: BaseModel, dataset: Dataset, generator: torch.Generator) -> Shaped[Tensor, 'num_nodes']:
        if prediction is None:
            raise RuntimeError('Energy expected a prediction')
        logits = prediction.logits if self.propagated else prediction.logits_unpropagated
        if logits is None:
            raise RuntimeError(f'Logit energy can only be computed when logits are available')
        energy = - self.temperature * torch.logsumexp(logits / self.temperature, dim=-1).mean(0)
        return energy
