from graph_al.data.config import RandomSBMConfig, SBMClassMeanSampling, SBMEdgeProbabilitiesType
from graph_al.data.sbm import SBMDataset
from graph_al.utils.logging import get_logger

from jaxtyping import jaxtyped, Float, Int
from typeguard import typechecked
from torch import Tensor
from typing import Tuple

import torch
import torch.distributions
import numpy as np
import networkx as nx
from scipy.stats import special_ortho_group

class RandomSBMDataset(SBMDataset):
    """ Class that samples the CSBM from the defined distribution. """
    
    def __init__(self, config: RandomSBMConfig, torch_rng: torch.Generator):
        self.seed = config.seed if config.seed is not None else torch_rng.seed()
        torch_rng = torch.Generator()
        torch_rng.manual_seed(self.seed)
        if isinstance(config.num_nodes, int):
            num_nodes = config.num_nodes
        else:
            num_nodes = int(torch.randint(*config.num_nodes, size=(1,), generator=torch_rng, dtype=torch.int).item())
        
        num_classes = config.num_classes
        if config.class_prior is None:
            class_prior = torch.full((num_classes,), 1 / num_classes, dtype=torch.float)
        else:
            raise NotImplementedError(f'The ground truth oracle does not support class priors at the moment')
            assert len(config.class_prior) == num_classes, f'Class prior has wrong size'
            assert torch.allclose(torch.tensor(sum(config.class_prior)), torch.tensor(1.0)), f'Class prior should sum to 1'
            class_prior = torch.tensor(config.class_prior, dtype=torch.float)
        
        # Sample the graph
        y = torch.multinomial(class_prior, num_nodes, generator=torch_rng, replacement=True)
        edge_probability_intra_community, edge_probability_inter_community = self._compute_edge_probabilities(config)
        affiliation_matrix = np.full((num_classes, num_classes), edge_probability_inter_community, dtype=float) # fill with q
        np.fill_diagonal(affiliation_matrix, edge_probability_intra_community) # diagonal to p

        nx_seed = int(torch.randint(2**16, size=(1,), generator=torch_rng, dtype=torch.int).item())
    
        graph = nx.stochastic_block_model(
            [(y == yi).sum().item() for yi in range(num_classes)],
            affiliation_matrix,
            seed = nx_seed,
            directed = config.directed,
        )
            
        y = torch.full_like(y, -1)
        for yi, idxs in enumerate(graph.graph['partition']):
            y[list(idxs)] = yi
        assert (y >= 0).all()
        get_logger().info(f'labels are {y.tolist()}')
        if len(graph.edges) == 0:
            get_logger().warn(f'Graph does not have any edges.')
            edge_idxs = torch.tensor([], dtype=torch.int).resize(2, 0)
        else:
            edge_idxs = torch.tensor(list(graph.edges)).T.contiguous()
        if not config.directed: 
            # undirected graphs introduce the reverse-edges as well: Carefull for the SBM
            # as the adjacency matrix only factorizes into its upper triangular part then
            # so effectively you have to ignore edges (i, j) where j <(=) i
            edge_idxs = torch.cat((edge_idxs, edge_idxs.flip(0)), dim=-1)
                
        # Sample the features
        class_means = self._sample_class_means(config, torch_rng)
        node_features = class_means[y]
        node_features += torch.randn(*node_features.size(), generator=torch_rng) * config.feature_sigma
        super().__init__(config, torch_rng, class_prior, affiliation_matrix, class_means,
                         config.feature_sigma, node_features, y, edge_idxs)
    
    def _compute_edge_probabilities(self, config: RandomSBMConfig, eps: float = 1e-6) -> Tuple[float, float]:
        match config.edge_probabilities:
            case SBMEdgeProbabilitiesType.CONSTANT:
                return config.edge_probability_intra_community, config.edge_probability_inter_community
            case SBMEdgeProbabilitiesType.BY_SNR_AND_EXPECTED_DEGREE:
                assert config.expected_degree is not None, f'If the edge probabilites are to be infered from a snr ratio, you need to specify some degree distribution'
                edge_probability_inter = config.expected_degree * config.num_classes / (config.num_nodes - 1) / \
                    (config.edge_probability_snr + config.num_classes - 1) # q
                if edge_probability_inter > 1.0:
                    get_logger().warn(f'Inter-community connection probability q for {config.num_classes} classes '
                                      f'and {config.num_nodes} nodes with an expected degree of {config.expected_degree} '
                                      f'and an SNR of {config.edge_probability_snr} calculated to {edge_probability_inter} > 1.0. '
                                      'Setting to 1.0')
                    edge_probability_inter = 1.0
                edge_probability_intra = min(1.0, edge_probability_inter * config.edge_probability_snr) # p
                if edge_probability_intra > 1.0:
                    get_logger().warn(f'Intra-community connection probability p for {config.num_classes} classes '
                                      f'and {config.num_nodes} nodes with an expected degree of {config.expected_degree} '
                                      f'and an SNR of {config.edge_probability_snr} and an inter-community connection '
                                      f'probability of {edge_probability_inter} calculated to {edge_probability_intra} > 1.0. '
                                      'Setting to 1.0')
                    edge_probability_intra = 1.0
                return min(1.0 - eps, max(eps, edge_probability_intra)), min(1.0 - eps, max(eps, edge_probability_inter))
            case type_:
                raise ValueError(f'Unsupported edge probability type {type_}')
    
    def _sample_class_means(self, config: RandomSBMConfig, torch_rng: torch.Generator) -> Float[Tensor, 'num_classes feature_dim']:
        num_classes, feature_dim = config.num_classes, config.feature_dim
        match config.feature_class_mean_sampling:
            case SBMClassMeanSampling.NORMAL:
                return torch.randn(num_classes, feature_dim, generator=torch_rng) * config.feature_class_mean_sigma
            case SBMClassMeanSampling.EQUIDISTANT:
                assert num_classes <= feature_dim
                means = torch.zeros(num_classes, feature_dim)
                means[torch.arange(num_classes), torch.arange(num_classes)] = config.feature_class_mean_distance / np.sqrt(2)
                # random rotation
                rotation = torch.from_numpy(special_ortho_group.rvs(feature_dim,
                    random_state=int(torch.randint(2**16, size=(1,), generator=torch_rng, dtype=torch.int).item()))).float()
                means = means @ rotation
                return means
            case sampling_type:
                raise ValueError(f'Unsupported sampling type {sampling_type}')
    