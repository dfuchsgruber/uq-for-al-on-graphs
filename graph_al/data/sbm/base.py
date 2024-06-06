from graph_al.data.generative import GenerativeDataset
from graph_al.data.config import SBMConfig
from graph_al.utils.logging import get_logger
from graph_al.utils.utils import ndarray_to_tuple

from jaxtyping import jaxtyped, Float, Int, Bool
from typing import Hashable, Any
from typeguard import typechecked
from torch import Tensor

import torch
import torch.distributions
import numpy as np
from graph_al.utils.sbm import (
    count_in_class_by_adjacency, 
    count_in_class_by_triangular_upper_adjacency,
    class_counts_by_node_to_affiliation_counts
)

class SBMDataset(GenerativeDataset):
    """ Class for a stochastic block model dataset. """
    
    def __init__(self, config: SBMConfig, torch_rng: torch.Generator,
                 class_prior: Float[Tensor, 'num_classes'],
                 affiliation_matrix: Float[np.ndarray, 'num_classes num_classes'],
                 class_means: Float[Tensor, 'num_classes feature_dim'],
                 feature_sigma: float,
                 x: Float[Tensor, 'num_nodes features_dim'],
                 y: Int[Tensor, 'num_nodes'],
                 edge_idxs: Int[Tensor, '2 num_edges'],
                 ):
        num_classes = config.num_classes
        self.directed = config.directed
        if config.class_prior is None:
            class_prior = torch.full((num_classes,), 1 / num_classes, dtype=torch.float)
        if not self.directed: 
            # Sanity check
            assert np.allclose(affiliation_matrix, affiliation_matrix.T), f'Undirected graphs can not be sampled from asymmetric affiliation matrices'
            assert set((u, v) for u, v in edge_idxs.T.tolist()) == set((v, u) for u, v in edge_idxs.T.tolist())
        self.class_prior = class_prior
        self.class_means = class_means
        self.feature_sigma = feature_sigma
        self.affiliation_matrix = affiliation_matrix
        super().__init__(config, x, y, edge_idxs, num_classes=num_classes)
    
    @property
    def key(self) -> Any:
        """ Hashable (tuple) representation of the dataset """
        return (
            'CSBM',
            ndarray_to_tuple(self.affiliation_matrix.round(5)),
            ndarray_to_tuple(self.class_means.numpy().round(5)),
            ndarray_to_tuple(self.edge_idxs.numpy()),
            ndarray_to_tuple(self.labels.numpy()),
            ndarray_to_tuple(self.class_prior.numpy().round(5)),
            int(self.num_classes),
        )

    
    @jaxtyped(typechecker=typechecked)
    def feature_log_likelihood(self, features: Float[np.ndarray, 'num_nodes feature_dim']) -> Float[np.ndarray, 'num_nodes num_classes']:
        """ Computes the feature log likelihood of each instance belonging to a certain class."""
        means = self.class_means.numpy()
        differences = features[:, None, :] - means[None, ...]
        log_likelihood = -(differences**2).sum(-1) / (2 * self.feature_sigma**2)
        log_likelihood -= (features.shape[-1] / 2) * np.log(2 * np.pi * self.feature_sigma**2)
        return log_likelihood

    @jaxtyped(typechecker=typechecked)
    def conditional_feature_log_likelihood(self, 
                                           assignments: Int[np.ndarray, 'num_assignments num_nodes'],
                                           features: Float[np.ndarray, '#num_assignments num_nodes feature_dim']) \
        -> Float[np.ndarray, 'num_assignments']:
        """ Computes the log likeihood of *batch features for each class, i.e. log p(X_i | y_i = c) """
        means = self.class_means.numpy()[assignments] # num_assignments, num_nodes, feature_dim
        differences = means - features
        log_likelihood = -(differences**2).sum(-1) / (2 * self.feature_sigma**2)
        log_likelihood -= (features.shape[-1] / 2) * np.log(2 * np.pi * self.feature_sigma**2)
        return log_likelihood.sum(-1) # sum over all nodes
    
    
    
    @jaxtyped(typechecker=typechecked)
    def conditional_adjacency_log_likelihood(self, assignments: Int[np.ndarray, 'num_assignments num_nodes'], check: bool = False,
                                             use_non_edges: bool=True) \
        -> Float[np.ndarray, 'num_assignments']:
        """ Computes the log likeihood of `num_assignment` class assignments for each node being in each class.
        
        Args:
            assignments: (Int[np.ndarray, 'num_assignments num_nodes'])]): assignments[s, i] is the assignment of the i-th node in the s-th assignment
            check (bool): If set, compares results to an unvectorized implementation for sanity checking. Will slow down everything immensely.
            use_non_edges (bool): If to use non-present edges for the likelihood as well
        
        Returns:
            Float[np.ndarray, 'num_assignments']: log p(A | y = s)
        """
        edge_idxs = self.edge_idxs.numpy().astype(int)
        if self.directed:
            _, class_counts_edges, class_counts_non_edges = count_in_class_by_adjacency(assignments, edge_idxs, self.num_classes)
        else:
            mask_upper_triangular = edge_idxs[0] < edge_idxs[1]
            edge_idxs = edge_idxs[:, mask_upper_triangular]
            _, class_counts_edges, class_counts_non_edges = count_in_class_by_triangular_upper_adjacency(assignments, edge_idxs, self.num_classes)
        
        counts_edges = class_counts_by_node_to_affiliation_counts(class_counts_edges, assignments)
        counts_non_edges = class_counts_by_node_to_affiliation_counts(class_counts_non_edges, assignments)
        
        log_likelihood = (counts_edges * np.log(self.affiliation_matrix)[None, :, :])
        if use_non_edges:
            log_likelihood += (counts_non_edges * np.log(1 - self.affiliation_matrix)[None, :, :])
        
        log_likelihood = log_likelihood.sum(-1).sum(-1)
        if check:
            log_likelihood_not_vectorized = self.conditional_adjacency_log_likelihood_no_vectorization(assignments, use_non_edges=use_non_edges)
            assert np.allclose(log_likelihood, log_likelihood_not_vectorized), (log_likelihood, log_likelihood_not_vectorized)
        return log_likelihood
        

    @jaxtyped(typechecker=typechecked)
    def conditional_adjacency_log_likelihood_no_vectorization(self, assignments: Int[np.ndarray, 'num_assignments num_nodes'],
                                                              use_non_edges: bool=True) \
        -> Float[np.ndarray, 'num_assignments']:
        """ Computes the log likeihood of `num_assignment` class assignments for each node being in each class.
        It does so without any vectorization and is slow.
        
        Args:
            assignments: (Int[np.ndarray, 'num_assignments num_nodes'])]): assignments[s, i] is the assignment of the i-th node in the s-th assignment
            use_non_edges (bool): If to use non-present edges for the likelihood as well
        
        Returns:
            Float[np.ndarray, 'num_assignments']: log p(A | y = s)
        """
        A = np.zeros((self.num_nodes, self.num_nodes))
        edge_idxs = self.edge_idxs.numpy()
        A[edge_idxs[0], edge_idxs[1]] = 1
        log_likelihoods = []
        for assignment in assignments:
            log_likelihood = 0.0
            for i in range(self.num_nodes):
                for j in range(i + 1, self.num_nodes) if not self.directed else range(self.num_nodes):
                    if i == j: continue
                    log_likelihood += A[i, j] * np.log(self.affiliation_matrix[assignment[i], assignment[j]])
                    if use_non_edges:
                        log_likelihood += (1 - A[i, j]) * np.log(1 - self.affiliation_matrix[assignment[i], assignment[j]])
            log_likelihoods.append(log_likelihood)
        return np.array(log_likelihoods)
        

    @jaxtyped(typechecker=typechecked)
    def conditional_log_likelihood(self, labels: Int[np.ndarray, 'num_assignments num_nodes'],
                              features: Float[np.ndarray, '#num_assignments num_nodes feature_dim'],
                              use_adjacency: bool = True,
                              use_features: bool = True,
                              use_non_edges: bool = True) -> Float[np.ndarray, 'num_assignments']:
        """ Computes log p(A, X | assignment) for each class and node
        
        Args:
            labels (Int[np.ndarray, 'num_assignments num_nodes']): Labels for which to evaluate the log likelihood
            features (Float[np.ndarray, 'num_nodes feature_dim']): Features for every node in 
            
        Returns:
            Float[np.ndarray, 'num_assignments']: For each assignment s, log p(A, X | s)
        """
        log_likelihood = np.zeros(list(labels.shape)[:-1]) # num_assignments
        if use_adjacency:
            log_likelihood += self.conditional_adjacency_log_likelihood(labels, use_non_edges=use_non_edges)
        if use_features:
            log_likelihood += self.conditional_feature_log_likelihood(labels, features)
        return log_likelihood


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
            Float[np.ndarray, 'num_classes num_assignments num_nodes_to_change']: For each assignment `a`, node to change `v` and class `c`, log p(A, X | a where y[v]=c)
        """
        assert not self.directed, f'Efficient incremental log likelihood computation is only supported for undirected CSBMs at the moment'
        if labels_log_likelihood is None:
            labels_log_likelihood = self.conditional_log_likelihood(labels, features, use_adjacency=use_adjacency, use_features=use_features,
            use_non_edges=use_non_edges)
        delta = np.zeros((self.num_classes, labels.shape[0], mask_nodes_to_change.sum()), dtype=float)

        if use_adjacency:
            log_affiliation, log_1_minus_affiliation = np.log(self.affiliation_matrix), np.log(1 - self.affiliation_matrix)
            edge_idxs = self.edge_idxs.numpy().astype(int)
            _, class_counts_edges, class_counts_non_edges = count_in_class_by_adjacency(labels, edge_idxs, self.num_classes)
            delta += ((log_affiliation[:, None, None, :] - log_affiliation[labels[:, mask_nodes_to_change]]) * class_counts_edges[:, mask_nodes_to_change][None, ...]).sum(-1)
            if use_non_edges:
                delta += ((log_1_minus_affiliation[:, None, None, :] - log_1_minus_affiliation[labels[:, mask_nodes_to_change]]) * class_counts_non_edges[:, mask_nodes_to_change][None, ...]).sum(-1)
        if use_features:
            class_means = self.class_means.numpy()
            delta += (
            # shape [num_classes, beam_size, num_nodes_to_change]: distances of the class means and the features w.r.t. new assignment
            ((class_means[:, None, None, :] - features[:, mask_nodes_to_change][None, :])**2).sum(-1) - \
            # shape [1, beam_size or 1, num_nodes_to_change]: distance of the class means and the features of the w.r.t. current assignment
            ((features[:, mask_nodes_to_change] - class_means[labels[:, mask_nodes_to_change]])**2).sum(-1)[None, ...] \
        ) / (-2 * self.feature_sigma**2)
        
        return np.transpose(labels_log_likelihood[None, :, None] + delta, (1, 2, 0))
        



