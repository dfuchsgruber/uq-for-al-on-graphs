from graph_al.model.base import BaseModelMonteCarloDropout
from graph_al.model.config import APPNPConfig
from graph_al.data.base import Dataset, Data
from graph_al.model.prediction import Prediction
from graph_al.utils.utils import apply_to_optional_tensors

import torch_geometric.nn as tgnn
import torch
from torch import Tensor
from jaxtyping import Float, Int, jaxtyped
from typeguard import typechecked
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

import torch.nn.functional as F
from torch import Tensor


class APPNP(BaseModelMonteCarloDropout):
    """ APPNP model. """
    
    def __init__(self, config: APPNPConfig, dataset: Dataset):
        super().__init__(config, dataset)
        self.inplace = config.inplace
        self.dropout = config.dropout
        self.layers = nn.ModuleList()
        in_dim = dataset.num_input_features
        
        for out_dim in list(config.hidden_dims) + [dataset.data.num_classes]:
            self.layers.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim
            
        self.propagate = tgnn.APPNP(config.k, config.alpha, dropout=self.dropout,
                                    cached=config.cached, add_self_loops=config.add_self_loops)
    @property
    def prediction_changes_at_eval(self) -> bool:
        return self.dropout > 0

    def reset_cache(self):
        self.propagate._cached_edge_index = None
        self.propagate._cached_adj_t = None

    def reset_parameters(self, generator: torch.Generator):
        for layer in self.layers:
            conv: nn.Linear = layer # type: ignore
            conv.reset_parameters()
            self.propagate.reset_parameters()
        

    @jaxtyped(typechecker=typechecked)  
    def forward(self, batch: Data, acquisition: bool=False) -> Tuple[Float[Tensor, 'num_nodes num_embeddings'] | None,
                                                                     Float[Tensor, 'num_nodes num_embeddings'] | None,
                                                                     Float[Tensor, 'num_nodes num_classes'],
                                                                     Float[Tensor, 'num_nodes num_classes']]:
        x, edge_index, edge_weight = batch.x, batch.edge_index, batch.edge_weight
        embeddings, embeddings_unpropagated = None, None
        for layer_idx, layer in enumerate(self.layers):
            x = layer(x)
            if acquisition and layer_idx == len(self.layers) - 2:
                embeddings_unpropagated = x
            if layer_idx != len(self.layers) - 1:
                x = F.relu(x, inplace=self.inplace)
                if self.dropout:
                    x = F.dropout(x, p=self.dropout, inplace=self.inplace and not acquisition, 
                                  training=self.training or self.dropout_at_eval)
        if self.training or self.dropout_at_eval:
            self.propagate.train(True)
        x_prop = self.propagate(x, edge_index, edge_weight)
        if acquisition and embeddings_unpropagated is not None:
            embeddings = self.propagate(embeddings_unpropagated, edge_index, edge_weight)
        self.propagate.train(self.training)
        return embeddings, embeddings_unpropagated, x_prop, x
    
    def predict_multiple(self, batch: Data, num_samples: int, acquisition: bool = False) -> Prediction:
        embeddings, embeddings_unpropagated, logits, logits_unpropagted = map(lambda tensors: apply_to_optional_tensors(torch.stack, tensors), # type: ignore
                                                                              zip(*[self(batch, acquisition=acquisition) for _ in range(num_samples)]))
        return Prediction(logits=logits, logits_unpropagated=logits_unpropagted, 
                          embeddings=embeddings, embeddings_unpropagated=embeddings_unpropagated)
        
            
    