

from graph_al.acquisition.config import AcquisitionStrategyByDataAttributeConfig, DataAttribute, AcquisitionStrategyByAPPRConfig
from graph_al.acquisition.attribute import AcquisitionStrategyByAttribute
from graph_al.data.base import Dataset
from graph_al.model.base import BaseModel
from graph_al.model.config import ModelConfig
from graph_al.model.prediction import Prediction
from graph_al.utils.ppr import approximate_ppr_scores

import torch

from typeguard import typechecked
from jaxtyping import jaxtyped, Shaped
from torch import Tensor

    
class AcquisitionStrategyByDataAttribute(AcquisitionStrategyByAttribute):
    """ Acquires by the maximum degree. """
    
    def __init__(self, config: AcquisitionStrategyByDataAttributeConfig):
        super().__init__(config)
        self.attribute = config.attribute
     
    @jaxtyped(typechecker=typechecked)
    def get_attribute(self, prediction: Prediction | None, model: BaseModel, dataset: Dataset, generator: torch.Generator,
            model_config: ModelConfig) -> Shaped[Tensor, 'num_nodes']:
        match self.attribute:
            case DataAttribute.IN_DEGREE:
                return dataset.node_degrees_in.to(dataset.data.mask_train_pool.device)
            case DataAttribute.OUT_DEGREE:
                return dataset.node_degrees_out.to(dataset.data.mask_train_pool.device)
            case _:
                raise ValueError(f'Unsupported data attribute {self.attribute}')
               
class AcquisitionStrategyByAPPR(AcquisitionStrategyByDataAttribute):
    """ Acquires nodes with the highest approximate PPR centrality """
    def __init__(self, config: AcquisitionStrategyByAPPRConfig):
        super().__init__(config)
        self.alpha = config.alpha
        self.k = config.k
        
    @jaxtyped(typechecker=typechecked)
    def get_attribute(self, prediction: Prediction | None, model: BaseModel, dataset: Dataset, 
                      generator: torch.Generator, model_config: ModelConfig) -> Shaped[Tensor, 'num_nodes']:
        return dataset.data.get_appr_scores(self.alpha, self.k).to(dataset.data.mask_train_pool.device)