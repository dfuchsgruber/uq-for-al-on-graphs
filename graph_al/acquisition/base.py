from abc import abstractmethod
from graph_al.acquisition.config import AcquisitionStrategyConfig
from graph_al.model.base import BaseModel
from graph_al.data.base import Dataset
from graph_al.model.prediction import Prediction
from graph_al.model.config import ModelConfig
from graph_al.utils.logging import get_logger

from collections import defaultdict

import torch
import torch_scatter
from torch import Tensor, Generator
from jaxtyping import Int, Bool, jaxtyped
from typeguard import typechecked
from typing import Tuple, Dict, Any, List

class BaseAcquisitionStrategy:
    """Base class for acquisition strategies.""" 
    
    def __init__(self, config: AcquisitionStrategyConfig):
        self.name = config.name
        self.balanced = config.balanced
        self.requires_model_prediction = config.requires_model_prediction
        self.verbose = config.verbose

    @property
    def retrain_after_each_acquisition(self) -> bool | None:
        """ If the model should be retrained after each acquisition. """
        return None

    @abstractmethod
    def is_stateful(self) -> bool:
        """ If the acquisition strategy is stateful, i.e. has a state that persists over multiple acquisitions. """
        return False

    def reset(self):
        """ Resets the acquisition strategies state. """
        ...

    def update(self, idxs_acquired: List[int], prediction: Prediction | None, dataset: Dataset, model: BaseModel):
        """ Updates the acquisition strategy state after acquisition.

        Args:
            idxs_acquired (List[int]): acquired indices
            prediction (Prediction | None): The model predictions that were used in the acquisition
            dataset (Dataset): the dataset
            model (BaseModel): the model in its state before `idxs_acquires` were acquired.
        """
        ...
        
    @abstractmethod
    def acquire_one(self, mask_acquired: Bool[Tensor, 'num_nodes'], prediction: Prediction | None, model: BaseModel, dataset: Dataset, model_config: ModelConfig, 
            generator: Generator) -> Tuple[int, Dict[str, Tensor | None]]:
        """ Acquires one label

        Args:
            mask_acquired (Bool[Tensor, &#39;num_nodes&#39;]): which nodes have been acquired in this iteration already
            prediction (Prediction | None): an optional model prediction if the acquisition needs that
            model (BaseModel): the classifier 
            dataset (Dataset): the dataset
            generator (Generator): a rng

        Returns:
            int: the acquired node label
            Dict[str, Tensor | None]: meta information from this aggregation
        """
        ...
        
    def acquire(self, model: BaseModel, dataset: Dataset, num: int, model_config: ModelConfig, generator: Generator) -> Tuple[Int[Tensor, 'num'], Dict[str, Any]]:
        """ Computes the nodes to acquire in this iteration. It iteratively calls `acquire_one`.
        
        Returns:
            Tuple[Int[Tensor, 'num']: the indices of acquired nodes
            Dict[str, int | float]: Metrics over the acquistion
        """
        acquired_idxs = []
        acquired_meta = defaultdict(list)
        mask_acquired_idxs = torch.zeros_like(dataset.data.mask_train_pool)
        
        if self.requires_model_prediction:
            with torch.no_grad():
                prediction = model.predict(dataset.data, acquisition=True)
        else:
            prediction = None    
    
        for _ in range(num):
            idx, acquired_meta_iteration = self.acquire_one(mask_acquired_idxs, prediction, model, dataset, model_config, generator)
            for k, v in acquired_meta_iteration.items():
                acquired_meta[k].append(v)
            acquired_idxs.append(idx)
            mask_acquired_idxs[idx] = True

        if self.is_stateful:
            self.update(acquired_idxs, prediction, dataset, model)
        
        return torch.tensor(acquired_idxs), self._aggregate_acquired_meta(acquired_meta)
    
    def _aggregate_acquired_meta(self, acquired_meta: Dict[str, Any]):
        # Filter out Nones
        acquired_meta = {k : [vi for vi in v if vi is not None] for k, v in acquired_meta.items()}
        acquired_meta = {k : v for k, v in acquired_meta.items() if len(v) > 0}
        
        aggregated = {}
        for k, v in acquired_meta.items():
            if all(isinstance(vi, Tensor) for vi in v):
                if len(set([tuple(vi.size()) for vi in v])) == 1: # Homogeneous tensors, we can stack
                    aggregated[k] = torch.stack(v)
                else:
                    aggregated[k] = v
            else:
                raise ValueError(f'Unsupported acquisition meta attribute of type(s) {list(type(vi) for vi in v)}')
        return aggregated
    
    def base_sampling_mask(self, model: BaseModel, dataset: Dataset, generator: torch.Generator) -> Bool[Tensor, 'num_nodes']:
        """ A mask from which it is legal to sample from."""
        return torch.ones_like(dataset.data.mask_train_pool)
    
    def pool(self, mask_sampled: Bool[Tensor, 'num_nodes'], model: BaseModel, dataset: Dataset, generator: Generator) -> Bool[Tensor, 'num_nodes']:
        """ Provides the pool to sample from according to the acquisition strategy realization.
        
        Args:
            mask_sampled: Bool[Tensor, 'num_nodes']: Mask for all nodes that are already selected in this current acquisition iteration (in case more than one labels are acquired in one iteration)

        Returns:
            Bool[Tensor, 'num_nodes']: Mask for the pool from which to sample
        """
        mask = dataset.data.mask_train_pool & (~mask_sampled) & self.base_sampling_mask(model, dataset, generator)
        if self.balanced:
            y = dataset.data.y[dataset.data.mask_train | mask_sampled]
            counts = torch_scatter.scatter_add(torch.ones_like(y), y, dim_size=dataset.data.num_classes)
            for label in torch.argsort(counts).detach().cpu().tolist():
                mask_class = mask & (dataset.data.y == label)
                if mask_class.sum() > 0:
                    return mask_class
                else:
                    # get_logger().warn(f'Can not sample balanced from class {label}. Not enough instances!')
                    ...
            else:
                raise RuntimeError(f'Could not sample from any class!')
        else:
            return mask
    
    @property
    def mask_not_in_val(self) -> Bool[Tensor, 'num_nodes'] | None:
        """ An optional mask of indices that should never be in the validation set and thus
        always available in the training pool. Needed for acquisitions with a fixed pool set. """
        return None

def mask_not_in_val(*acquisition_strategies) -> Bool[Tensor, 'num_nodes'] | None:
    """
    Returns the optional mask of idxs that should not be in validation masks given multiple acquisition strategies.
    """
    masks = [strategy.mask_not_in_val for strategy in acquisition_strategies if strategy.mask_not_in_val is not None]
    if len(masks) == 0:
        return None
    else:
        mask = masks[0]
        for other_mask in masks[1:]:
            mask |= other_mask
    return mask
