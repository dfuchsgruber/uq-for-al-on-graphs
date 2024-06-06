from abc import abstractmethod

from graph_al.data.base import BaseDataset
from graph_al.data.config import GenerativeDataConfig

from jaxtyping import jaxtyped, Float, Int, Bool
from typeguard import typechecked
from typing import Hashable
import torch.distributions
import numpy as np
from torch import Tensor
from typing import Dict

class GenerativeDataset(BaseDataset):
    """ Base dataset class that allows access to the likelihood over features and adjacency. """
    
    def __init__(self, config: GenerativeDataConfig, node_features: Float[Tensor, 'num_nodes num_node_features'], labels: Int[Tensor, 'num_nodes'], edge_idxs: Int[Tensor, '2 num_edges'], 
            num_classes: int | None = None, mask_train: Bool[Tensor, 'num_nodes'] | None = None, mask_val: Bool[Tensor, 'num_nodes'] | None = None, 
            mask_test: Bool[Tensor, 'num_nodes'] | None = None, node_to_idx: Dict[str, int] | None = None, label_to_idx: Dict[str, int] | None = None, feature_to_idx: Dict[str, int] | None = None):
        super().__init__(node_features, labels, edge_idxs, num_classes, mask_train, mask_val, mask_test, node_to_idx, label_to_idx, feature_to_idx)
        self.likelihood_cache_database_path = config.likelihood_cache_database_path
        self.likelihood_cache_database_lockfile_path = config.likelihood_cache_database_lockfile_path
        self.likelihood_cache_storage_path = config.likelihood_cache_storage_path
        self._in_memory_likelihood_cache = {}

    @abstractmethod
    @jaxtyped(typechecker=typechecked)
    def conditional_log_likelihood(self, labels: Int[np.ndarray, 'num_assignments num_nodes'],
                              features: Float[np.ndarray, '#num_assignments num_nodes feature_dim'],
                              use_adjacency: bool = True,
                              use_features: bool = True,
                              use_non_edges: bool = True) -> Float[np.ndarray, 'num_assignments']:
        """ Computes log p(A, X | assignment) for each assignment in `labels`
        
        Args:
            labels (Int[np.ndarray, 'num_assignments num_nodes']): Labels for which to evaluate the log likelihood
            features (Float[np.ndarray, 'num_nodes feature_dim']): Features for every node in 
            
        Returns:
            Float[np.ndarray, 'num_assignments']: For each assignment s, log p(A, X | s)
        """
        raise NotImplementedError
    
    @jaxtyped(typechecker=typechecked)
    def conditional_log_likelihood_delta_with_one_label_changed(self, labels: Int[np.ndarray, 'num_assignments num_nodes'],
                              features: Float[np.ndarray, '#num_assignments num_nodes feature_dim'],
                              mask_nodes_to_change: Bool[np.ndarray, 'num_nodes'],
                              labels_log_likelihood: Float[np.ndarray, 'num_assignments'] | None = None,
                              use_adjacency: bool = True,
                              use_features: bool = True,
                              use_non_edges: bool = True) -> Float[np.ndarray, 'num_assignments num_nodes_to_change num_classes']:
        """ Computes log p(A, X | assignment) for all assignments that differ in one labeling to each assignment in `labels`.
        That is, for each assignment `a` in `labels` and each node `v` that is in the mask `mask_nodes_to_change` and each class `c`, it computes
        the log likelihood when changing `v`'s label to `c`

        
        Args:
            labels (Int[np.ndarray, 'num_assignments num_nodes']): Labels for which to evaluate the log likelihood
            features (Float[np.ndarray, 'num_nodes feature_dim']): Features for every node in 
            mask_nodes_to_change (Bool[np.ndarray, 'num_nodes']): Which nodes should change their labels
            labels_log_likelihood (Float[np.ndarray, 'num_assignments'] | None): Optionally some pre-computed log likelihoods for the labels. If not given, they will be re-computed.
            
        Returns:
            Float[np.ndarray, 'num_assignments num_nodes_to_change num_classes']: For each assignment `a`, node to change `v` and class `c`, log p(A, X | a where y[v]=c)
        """
        # TODO: one could provide a base implementation using `conditional_log_likelihood`
        raise NotImplementedError

    @property
    def key(self) -> Hashable:
        raise NotImplementedError()