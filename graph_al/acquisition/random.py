from graph_al.acquisition.base import BaseAcquisitionStrategy
from graph_al.acquisition.config import AcquireRandomConfig
from graph_al.data.base import BaseDataset, Dataset
from graph_al.model.base import BaseModel
from graph_al.model.prediction import Prediction
from graph_al.model.config import ModelConfig
from graph_al.utils.logging import get_logger

from jaxtyping import Float, Int, jaxtyped, Bool
from typeguard import typechecked
from torch import Tensor
from typing import Tuple, Dict, Any

from torch_geometric.data import Data
import torch

class AcquisitionStrategyRandom(BaseAcquisitionStrategy):
    
    """ Strategy that selects randomly. """
    
    def __init__(self, config: AcquireRandomConfig):
        super().__init__(config)
     
    def acquire_one(self, mask_acquired: Bool[Tensor, 'num_nodes'], prediction: Prediction | None, model: BaseModel, dataset: Dataset,  
            model_config: ModelConfig, generator: torch.Generator) -> Tuple[int, Dict[str, Tensor | None]]:
        idxs_pool = torch.where(self.pool(mask_acquired, model, dataset, generator))[0]
        idx_sampled = int(idxs_pool[int(torch.randint(idxs_pool.size(0), size=(1,), generator=generator).item())].item())
        return idx_sampled, {}
        