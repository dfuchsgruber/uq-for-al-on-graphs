

from graph_al.acquisition.base import BaseAcquisitionStrategy
from graph_al.acquisition.config import AcquisitionStrategyFeatPropConfig
from graph_al.data.base import BaseDataset, Dataset
from graph_al.model.base import BaseModel
from graph_al.model.prediction import Prediction
from graph_al.model.config import ModelConfig

from jaxtyping import Float, Int, jaxtyped, Bool, Shaped
from typeguard import typechecked
from torch import Tensor
from typing import Dict, Tuple, Any
from sklearn_extra.cluster import KMedoids
import torch
import numpy as np

from graph_al.model.trainer.config import TrainerConfig

class AcquisitonStrategyFeatProp(BaseAcquisitionStrategy):
    """ Feat Prop strategy

    References:
    [1] : https://arxiv.org/pdf/1910.07567.pdf
    """

    def __init__(self, config: AcquisitionStrategyFeatPropConfig):
        super().__init__(config)
        self.k = config.k
        self.improved = config.improved
        self.add_self_loops = config.add_self_loops
        self.normalize = config.normalize
                
    @torch.no_grad()
    def acquire_one(self, mask_acquired: Bool[Tensor, 'num_nodes'], prediction: Prediction | None, model: BaseModel, dataset: Dataset, model_config: ModelConfig, 
        generator: torch.Generator) -> Tuple[int, Dict[str, Tensor | None]]:

        mask_train_pool = dataset.data.mask_train_pool
        
        kmedoids = KMedoids(n_clusters=1) # only acquire one
        x_propagated = dataset.data.get_diffused_nodes_features(k=self.k, normalize=self.normalize, improved=self.improved,
            add_self_loops=self.add_self_loops)
        x_propagated = x_propagated.cpu().numpy()[mask_train_pool.cpu()]
        idxs_train_pool = torch.where(mask_train_pool)[0]
        kmedoids.fit(x_propagated)
        idx_acquired = int(idxs_train_pool[kmedoids.medoid_indices_[0]].item() ) 
        return idx_acquired, {}