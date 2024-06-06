from typing import Tuple
from graph_al.model.base import BaseModelMonteCarloDropout
from graph_al.model.config import GCNConfig
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

class GCN(BaseModelMonteCarloDropout):
    """ GCN model. """
    
    def __init__(self, config: GCNConfig, dataset: Dataset):
        super().__init__(config, dataset)
        self.inplace = config.inplace
        self.layers = nn.ModuleList()
        in_dim = dataset.num_input_features
        for out_dim in list(config.hidden_dims) + [dataset.data.num_classes]:
            self.layers.append(tgnn.GCNConv(in_dim, out_dim, improved=config.improved,
                                            cached=config.cached, add_self_loops=config.add_self_loops))
            in_dim = out_dim
    
    def reset_cache(self):
        for layer in self.layers:
            layer._cached_edge_index = None
            layer._cached_adj_t = None

    def reset_parameters(self, generator: torch.Generator):
        for layer in self.layers:
            conv: tgnn.GCNConv = layer # type: ignore
            conv.reset_parameters()
    
    @jaxtyped(typechecker=typechecked)
    def forward_impl(self, x: Float[Tensor, 'num_nodes num_input_features'],
                     edge_index: Int[Tensor, '2 num_edges'] | None,
                     edge_weight: Int[Tensor, 'num_edges 1'] | None = None,
                     acquisition: bool=False) -> Tuple[
                         Float[Tensor, 'num_nodes embedding_dim'] | None,
                         Float[Tensor, 'num_nodes num_classes']]:
        embedding = None
        for layer_idx, layer in enumerate(self.layers):
            
            if edge_index is None:
                x = layer.lin(x) # type: ignore
            else:
                x = layer(x, edge_index, edge_weight)
            if acquisition and layer_idx == len(self.layers) - 2: # only return an embedding when doing acquisition
                embedding = x
            if layer_idx != len(self.layers) - 1:
                x = F.relu(x, inplace=self.inplace)
                if self.dropout:
                    x = F.dropout(x, p=self.dropout, inplace=self.inplace and not acquisition, 
                                  training=self.training or self.dropout_at_eval)
        return embedding, x

    @property
    def prediction_changes_at_eval(self):
        return self.dropout > 0

    @jaxtyped(typechecker=typechecked)  
    def forward(self, batch: Data, acquisition: bool=False) -> Tuple[Float[Tensor, 'num_nodes num_embeddings'] | None, 
                                                                     Float[Tensor, 'num_nodes num_embeddings'] | None,
                                                                     Float[Tensor, 'num_nodes num_classes'], 
                                                                     Float[Tensor, 'num_nodes num_classes'] | None]:
        embeddings, logits = self.forward_impl(batch.x, batch.edge_index, batch.edge_attr, acquisition=acquisition)
        if acquisition:
            embeddings_unpropagated, logits_unpropagated = self.forward_impl(batch.x, edge_index=None, edge_weight=None, 
                                                                             acquisition=acquisition)
        else:
            embeddings_unpropagated, logits_unpropagated = None, None
        return embeddings, embeddings_unpropagated, logits, logits_unpropagated
    
    @typechecked
    def predict_multiple_collate(self, batch: Data, num_samples: int, acquisition: bool = False) -> Prediction:
        """ Predicts multiple samples at once by collating into one large batch """
        batch_collated = self.collate_samples(batch, num_samples)
        embeddings, embeddings_unpropagated, logits, logits_unpropagted = self(batch_collated, acquisition=acquisition)
        embeddings = self.split_predicted_tensor(embeddings, num_samples)
        embeddings_unpropagated = self.split_predicted_tensor(embeddings_unpropagated, num_samples)
        logits = self.split_predicted_tensor(logits, num_samples)
        logits_unpropagted = self.split_predicted_tensor(logits, num_samples)
        return Prediction(logits=logits, logits_unpropagated=logits_unpropagted, 
                          embeddings=embeddings, embeddings_unpropagated=embeddings_unpropagated)
    
    @typechecked
    def predict_multiple_iterative(self, batch: Data, num_samples: int, acquisition: bool = False) -> Prediction:
        """ Predicts multiple samples by iteratively using `self.forward` """
        embeddings, embeddings_unpropagated, logits, logits_unpropagted = map(lambda tensors: apply_to_optional_tensors(torch.stack, tensors),  # type: ignore
                                                                              zip(*[self(batch, acquisition=acquisition) for _ in range(num_samples)]))
        return Prediction(logits=logits, logits_unpropagated=logits_unpropagted, 
                          embeddings=embeddings, embeddings_unpropagated=embeddings_unpropagated)  
    
    @typechecked
    def predict_multiple(self, batch: Data, num_samples: int, acquisition: bool = False) -> Prediction:
        if self._collate_samples and num_samples > 1:
            return self.predict_multiple_collate(batch, num_samples, acquisition=acquisition)
        else:
            return self.predict_multiple_iterative(batch, num_samples, acquisition=acquisition)
    
        
            
    