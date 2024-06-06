from graph_al.acquisition.base import BaseAcquisitionStrategy
from graph_al.acquisition.config import AcquisitionStrategyByAttributeConfig
from graph_al.data.base import Dataset
from graph_al.model.base import BaseModel
from graph_al.model.prediction import Prediction
from graph_al.model.config import ModelConfig
from graph_al.utils.logging import get_logger

from jaxtyping import jaxtyped, Shaped, Bool
from typeguard import typechecked
from torch import Tensor, Generator
from typing import Tuple, Dict

import torch

class AcquisitionStrategyByAttribute(BaseAcquisitionStrategy):
    
    """ Strategy that acquires by selecting greedily the best nodes based
    on an attribute over all nodes. """
    
    def __init__(self, config: AcquisitionStrategyByAttributeConfig):
        super().__init__(config)
        self.higher_is_better = config.higher_is_better
    
    @jaxtyped(typechecker=typechecked)
    def get_attribute(self, prediction: Prediction | None, model: BaseModel, dataset: Dataset, generator: Generator, model_config: ModelConfig) -> Shaped[Tensor, 'num_nodes']:
        raise NotImplementedError
    
    def acquire_one(self, mask_acquired: Bool[Tensor, 'num_nodes'], prediction: Prediction | None, model: BaseModel, dataset: Dataset, 
            model_config: ModelConfig, generator: Generator) -> Tuple[int, Dict[str, Tensor | None]]:
        attribute = self.get_attribute(prediction, model, dataset, generator, model_config)
        if self.higher_is_better:
            attribute = -attribute
        idx_pool = torch.where(self.pool(mask_acquired, model, dataset, generator))[0]
        if idx_pool.size(0) == 0:
            get_logger().warn(f'Trying to acquire label, but only none are in the pool.')
        attribute_pool = attribute[idx_pool]
        idx_sampled = idx_pool[torch.argmin(attribute_pool)].item()
        return int(idx_sampled), {'acquisition_attribute' : attribute.detach().cpu()}