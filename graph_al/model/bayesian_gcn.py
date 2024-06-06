from typing import Tuple
from graph_al.model.base import BaseModelMultipleSamples
from graph_al.model.config import GCNConfig, BayesianGCNConfig
from graph_al.data.base import Dataset, Data
from graph_al.model.prediction import Prediction
from graph_al.utils.utils import apply_to_optional_tensors

import torch_geometric.nn as tgnn
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import torch
from torch import Tensor
from jaxtyping import Float, Int, jaxtyped
from typeguard import typechecked
import torch.nn as nn
import torch.nn.functional as F

@jaxtyped(typechecker=typechecked)
def sample_weight(mean: Float[Tensor, '...'], rho: Float[Tensor, '...']) -> Tuple[Float[Tensor, '...'], Float[Tensor, '...']]:
    """ Samples a weight using reparametrization. """
    sigma = (1.0 + rho.exp()).log()
    eps = torch.zeros_like(mean).normal_()
    return mean + (eps * sigma), sigma

@jaxtyped(typechecker=typechecked)
def kl_divergence_to_standard_normal(mean: Float[Tensor, '...'], std: Float[Tensor, '...']) -> Float[Tensor, '']:
    """ Computes the KL divergence of a diagonal normal to the standard normal prior. """
    kl = -0.5 * (std.log()*2 - mean.pow(2) - std.pow(2) + 1).sum()
    return kl

class BayesianLinear(nn.Module):
    
    def __init__(self, in_dim: int, out_dim: int, config: BayesianGCNConfig):
        super().__init__()
        # Parametrize sigma in [0, inf[ as log(1 + exp(rho))
        
        self.rho_init = config.rho_init
        
        self.w_mean = nn.Parameter(torch.Tensor(in_dim, out_dim))
        self.w_rho = nn.Parameter(torch.Tensor(in_dim, out_dim))
        
        self.b_mean = nn.Parameter(torch.Tensor(out_dim))
        self.b_rho = nn.Parameter(torch.Tensor(out_dim))
        self.reset_parameters()
    
    @property
    def num_kl_terms(self) -> int:
        return sum(self.w_mean.size()) + sum(self.b_mean.size())
    
    def reset_parameters(self, generator: torch.Generator | None=None):
        nn.init.normal_(self.w_mean, std=0.1)
        nn.init.constant_(self.w_rho, -3.0)
        nn.init.zeros_(self.b_mean)
        nn.init.constant_(self.b_rho, -3.0)
        
    @jaxtyped(typechecker=typechecked)
    def sample(self) -> Tuple[Float[Tensor, 'in_dim out_dim'], Float[Tensor, 'out_dim'], Float[Tensor, '']]:
        """ Samples the weight and bias of the transformation. """
        w, sigma_w = sample_weight(self.w_mean, self.w_rho)
        b, sigma_b = sample_weight(self.b_mean, self.b_rho)
        kl = kl_divergence_to_standard_normal(self.w_mean, sigma_w) + kl_divergence_to_standard_normal(self.b_mean, sigma_b)
        return w, b, kl
        
    @jaxtyped(typechecker=typechecked)
    def forward(self, x: Float[Tensor, 'n in_dim']) -> Tuple[Float[Tensor, 'n out_dim'], Float[Tensor, '']]:
        w, b, kl = self.sample()
        return torch.mm(x, w) + b, kl
        

class BayesianGCNConv(tgnn.GCNConv):
    
    def __init__(self, config: BayesianGCNConfig, in_dim: int, out_dim: int):
        super().__init__(in_dim, out_dim, improved=config.improved, cached=config.cached,
                         add_self_loops=config.add_self_loops)
        self.lin = BayesianLinear(in_dim, out_dim, config)
    
    def reset_cache(self):
        self._cached_adj_t = None
        self._cached_edge_index = None

    def reset_parameters(self, generator: torch.Generator | None = None):
        super().reset_parameters()
        self.lin.reset_parameters()
    
    @property
    def num_kl_terms(self) -> int:
        return self.lin.num_kl_terms
    
    @jaxtyped(typechecker=typechecked)
    def forward(self, x: Float[Tensor, 'num_nodes in_dim'], edge_index: Int[Tensor, '2 num_edges'], 
                edge_weight: Float[Tensor, 'num_edges ...'] | None) -> Tuple[Float[Tensor, 'num_nodes out_dim'], Float[Tensor, '']]:
        x, kl = self.lin(x)
        
        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, self.flow, x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)
        return out, kl

class BayesianGCN(BaseModelMultipleSamples):
    """ Bayesian GCN that samples the weights and biases of its linear transformations from a diagonal normal """
    
    def __init__(self, config: BayesianGCNConfig, dataset: Dataset):
        super().__init__(config, dataset)
        self.inplace = config.inplace
        in_dim = dataset.num_input_features
        self.layers = nn.ModuleList()
        for out_dim in list(config.hidden_dims) + [dataset.data.num_classes]:
            self.layers.append(BayesianGCNConv(config, in_dim, out_dim))
            in_dim = out_dim
            
    def reset_parameters(self, generator: torch.Generator):
        for layer in self.layers:
            layer.reset_parameters(generator) # type: ignore
            
    def reset_cache(self):
        for layer in self.layers:
            layer.reset_cache() # type: ignore
    
    @property
    def num_kl_terms(self) -> int:
        return sum(layer.num_kl_terms for layer in self.layers) # type: ignore
    
    @jaxtyped(typechecker=typechecked)
    def forward_impl(self, x: Float[Tensor, 'num_nodes num_input_features'],
                     edge_index: Int[Tensor, '2 num_edges'] | None,
                     edge_weight: Int[Tensor, 'num_edges 1'] | None = None,
                     acquisition: bool = False) -> Tuple[Float[Tensor, 'num_nodes num_classes'],
                                                         Float[Tensor, 'num_nodes embedding_dim'] | None,
                                                         Float[Tensor, '']]:
        kl_total = torch.tensor(0.0, device=x.device)
        embedding = None
        for layer_idx, layer in enumerate(self.layers):
            
            if edge_index is None:
                x, kl = layer.lin(x) # type: ignore
            else:
                x, kl = layer(x, edge_index, edge_weight)
            kl_total += kl
            
            if acquisition and layer_idx == len(self.layers) - 2: # only return an embedding when doing acquisition
                embedding = x
            if layer_idx != len(self.layers) - 1:
                x = F.relu(x, inplace=self.inplace and not acquisition)
        return x, embedding, kl_total
    
    @jaxtyped(typechecker=typechecked)  
    def forward(self, batch: Data, acquisition: bool=False) -> Tuple[Float[Tensor, 'num_nodes num_classes'], 
                                                                     Float[Tensor, 'num_nodes num_classes'] | None,
                                                                     Float[Tensor, 'num_nodes embedding_dim'] | None, 
                                                                     Float[Tensor, 'num_nodes embedding_dim'] | None,
                                                                     Float[Tensor, ''],
                                                                     Float[Tensor, ''] | None]:
        logits, embeddings, kl = self.forward_impl(batch.x, batch.edge_index, batch.edge_attr, acquisition=acquisition)
        if acquisition:
            logits_unpropagated, embeddings_unpropagated, kl_unpropagated = self.forward_impl(batch.x, edge_index=None, edge_weight=None, acquisition=acquisition)
        else:
            logits_unpropagated, embeddings_unpropagated, kl_unpropagated = None, None, None
        return logits, logits_unpropagated, embeddings, embeddings_unpropagated, kl, kl_unpropagated
    
    @typechecked
    def predict_multiple(self, batch: Data, num_samples: int, acquisition: bool = False) -> Prediction:
        logits, logits_unpropagted, embeddings, embeddings_unpropagated, kl, kl_unpropagated = map(lambda tensors: apply_to_optional_tensors(torch.stack, tensors), # type: ignore
                                                                                                   zip(*[self(batch, acquisition=acquisition) for _ in range(num_samples)]))
        assert kl is not None
        return Prediction(logits=logits, logits_unpropagated=logits_unpropagted, kl_divergence=kl,
                          num_kl_terms=torch.tensor(self.num_kl_terms), embeddings=embeddings, embeddings_unpropagated=embeddings_unpropagated)
    