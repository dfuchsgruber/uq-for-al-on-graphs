from graph_al.data.config import DeterministicSBMConfig, RandomSBMConfig, SBMClassMeanSampling, SBMEdgeProbabilitiesType
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

class DeterministicSBMDataset(SBMDataset):
    """ Class with a fixed graph that is "sampled" from a CSBM generative process. 
    Used for toy graphs. """
    
    
    def __init__(self, config: DeterministicSBMConfig, torch_rng: torch.Generator):
        
        num_classes = config.num_classes
        if config.affiliation_matrix is None:
            raise ValueError(f'Need to supply an affiliation matrix')
        
        if config.class_prior is None:
            class_prior = torch.full((num_classes,), 1 / num_classes, dtype=torch.float)
        else:
            raise NotImplementedError(f'The ground truth oracle does not support class priors at the moment')
        
        super().__init__(config, torch_rng, class_prior,
                         np.array(config.affiliation_matrix),
                         torch.tensor(config.class_means, dtype=torch.float), 
                         config.feature_sigma,
                         torch.tensor(config.node_features, dtype=torch.float),
                         torch.tensor(config.labels),
                         torch.tensor(config.edge_index))
        