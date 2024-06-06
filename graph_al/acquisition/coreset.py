from graph_al.acquisition.base import BaseAcquisitionStrategy
from graph_al.acquisition.config import CoresetConfig, CoresetDistance, CoresetAPPRConfig
from graph_al.data.base import BaseDataset, Dataset
from graph_al.model.base import BaseModel
from graph_al.model.prediction import Prediction
from graph_al.model.config import ModelConfig
from graph_al.utils.logging import get_logger

from jaxtyping import Float, Int, jaxtyped, Bool, Shaped
from typeguard import typechecked
from torch import Tensor
from typing import Dict, Tuple, Any

import torch

class AcquisitionStrategyCoreset(BaseAcquisitionStrategy):
    """ Strategy that uses k-greedy approximation to Coreset [1] for sampling.
     
    When acquiring only one node at a time, this approximation is optimal
     
    References:
    [1] : https://openreview.net/pdf?id=H1aIuk-RW
    """
    
    def __init__(self, config: CoresetConfig):
        super().__init__(config)
        self.propagated = config.propagated
        self.distance_metric = config.distance
        self.distance_norm = config.distance_norm
        
    @jaxtyped(typechecker=typechecked)
    def _latent_features_distance(self, mask_train: Bool[Tensor, 'num_nodes'], 
                  mask_train_pool: Bool[Tensor, 'num_nodes'], 
                  prediction: Prediction | None) -> Float[Tensor, 'num_nodes_train num_nodes_train_pool']:
        if prediction is None:
            raise RuntimeError(f'Latent feature distance requires model predictions')
        if self.propagated:
            x = prediction.embeddings
        else:
            x = prediction.embeddings_unpropagated
        
        if x is None:
            raise RuntimeError(f'Model does not predict attribute requested by Coreset {self.distance_metric}')
        assert len(x.size()) == 3 # num_samples, num_nodes, d
        x = x.mean(0) # TODO: this is probably a bad idea for models that have more than 1 sample
        return torch.cdist(x[mask_train], x[mask_train_pool], p=self.distance_norm)
    
    @jaxtyped(typechecker=typechecked)
    def _input_features_distance(self, mask_train: Bool[Tensor, 'num_nodes'], 
                  mask_train_pool: Bool[Tensor, 'num_nodes'], 
                  dataset: Dataset) -> Float[Tensor, 'num_nodes_train num_nodes_train_pool']:
        x = dataset.data.x
        return torch.cdist(x[mask_train], x[mask_train_pool], p=self.distance_norm) 
        
    @jaxtyped(typechecker=typechecked)
    def _distance(self, mask_train: Bool[Tensor, 'num_nodes'], 
                  mask_train_pool: Bool[Tensor, 'num_nodes'],
                  model: BaseModel, dataset: Dataset, prediction: Prediction | None,
                  generator: torch.Generator) -> Float[Tensor, 'num_nodes_train num_nodes_train_pool']:
        """ computes the distance metric used for the coreset algorithm """
        
        match self.distance_metric:
            case CoresetDistance.LATENT_FEATURES:
                return self._latent_features_distance(mask_train, mask_train_pool, prediction)
            case CoresetDistance.INPUT_FEATURES:
                return self._input_features_distance(mask_train, mask_train_pool, dataset)
            case _:
                raise ValueError(f'Unsupported coreset distance {self.distance_metric}')
        
    @torch.no_grad()
    def acquire_one(self, mask_acquired: Bool[Tensor, 'num_nodes'], prediction: Prediction | None, model: BaseModel, 
                    dataset: Dataset, model_config: ModelConfig, generator: torch.Generator) -> Tuple[int, Dict[str, Tensor | None]]:
        
        mask_pool = self.pool(mask_acquired, model, dataset, generator)
        idx_pool = torch.where(mask_pool)[0]
        mask_train_or_acquired = dataset.data.mask_train | mask_acquired
        
        if mask_train_or_acquired.sum() == 0:
            # No training instances, sample one index randomly from pool
            distance = None
            sampled_idx = idx_pool[torch.randint(idx_pool.size(0), (1,), generator=generator).item()].item() # type: ignore
        else:
            # Select the instance from the pool that has the maximal min-distance to all train nodes
            # i.e. the one thats furthest away to all of them
            distance = self._distance(mask_train_or_acquired, mask_pool, model, dataset, prediction, generator)
            min_dist = distance.min(0)[0]
            sampled_idx = idx_pool[min_dist.argmax().item()].item()
        return int(sampled_idx), {'train_to_train_pool_distances' : distance}
        
class AcquisitionStrategyCoresetAPPR(AcquisitionStrategyCoreset):
    """ Aquisition strategy that uses coreset on approximate Personalized Page Rank scores as a distance. """
    
    def __init__(self, config: CoresetAPPRConfig):
        super().__init__(config)
        self.k = config.k
        self.alpha = config.alpha
    
    @jaxtyped(typechecker=typechecked)
    def _distance(self, mask_train: Bool[Tensor, 'num_nodes'], 
                  mask_train_pool: Bool[Tensor, 'num_nodes'],
                  model: BaseModel, dataset: Dataset, prediction: Prediction,
                  generator: torch.Generator) -> Float[Tensor, 'num_nodes_train num_nodes_train_pool']:
        log_appr_matrix = dataset.data.log_appr_matrix(teleport_probability=self.alpha, num_iterations=self.k).T
        # We transpose as we want the importance of a pool node to a training node
        # The distance matrix returned has train nodes on dim=0 and pool nodes on dim=1
        # Also: High ppr scores mean low distance, so we use the inverse
        return -log_appr_matrix[mask_train.cpu()][:, mask_train_pool.cpu()]