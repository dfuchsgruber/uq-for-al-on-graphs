from graph_al.acquisition.base import BaseAcquisitionStrategy
from graph_al.acquisition.config import AcquireByPredictionAttributeConfig, PredictionAttribute
from graph_al.data.base import BaseDataset, Dataset
from graph_al.model.base import BaseModel
from graph_al.model.config import ModelConfig
from graph_al.model.prediction import Prediction
from graph_al.acquisition.attribute import AcquisitionStrategyByAttribute
from graph_al.utils.logging import get_logger

from jaxtyping import Float, Int, jaxtyped, Shaped, Bool
from typeguard import typechecked
from torch import Tensor, Generator
from typing import Tuple, Dict, Any

from torch_geometric.data import Data
import torch

class AcquisitionStrategyByPredictionAttribute(AcquisitionStrategyByAttribute):
    
    """ Strategy that acquires by selecting greedily the best nodes based
    on an attribute of the prediction. """
    
    def __init__(self, config: AcquireByPredictionAttributeConfig):
        super().__init__(config)
        self.attribute = config.attribute
        self.higher_is_better = config.higher_is_better
        self.propagated = config.propagated
        
    @jaxtyped(typechecker=typechecked)
    def get_attribute(self, prediction: Prediction | None, model: BaseModel, dataset: Dataset, generator: Generator, model_config: ModelConfig) -> Shaped[Tensor, 'num_nodes']:
        if prediction is None:
            raise ValueError(f'Can not derive prediction attribute if no prediction is given')
        return prediction.get_attribute(self.attribute, self.propagated)
