from typing import Tuple
from graph_al.data.enum import DatasetSplit
from graph_al.model.base import BaseModel
from graph_al.model.config import GCNConfig, SEALConfig
from graph_al.model.gcn import GCN
from graph_al.data.base import Dataset, Data
from graph_al.model.prediction import Prediction
from graph_al.utils.utils import apply_to_optional_tensors

import torch_geometric.nn as tgnn
import torch
from torch import Tensor
from jaxtyping import Float, Int, jaxtyped, Bool
from typeguard import typechecked
import torch.nn as nn
import torch.nn.functional as F

class SEAL(BaseModel):
    """ The SEAL [1] model using adversarial training.
    
    References:
    [1] https://arxiv.org/ftp/arxiv/papers/1908/1908.08169.pdf
    """
    
    def __init__(self, config: SEALConfig, dataset: Dataset):
        super().__init__(config, dataset)
        self.delta = config.delta
        self.gcn = GCN(config, dataset)
        self.discriminator_dropout = config.discriminator_dropout
        self.discriminator = nn.ModuleList()
        in_dim = config.hidden_dims[-1]
        for out_dim in config.hidden_dims_discriminator + [dataset.num_classes]:
            self.discriminator.append(nn.Linear(in_dim, out_dim))
        self._gcn_frozen = False
    
    def freeze_gcn(self):
        self._gcn_frozen = True
        
    def unfreeze_gcn(self):
        self._gcn_frozen = False
        
    def reset_cache(self):
        self.gcn.reset_cache()

    def reset_parameters(self, generator: torch.Generator):
        self.gcn.reset_parameters(generator)
        for linear in self.discriminator: # type: ignore
            linear: nn.Linear = linear
            linear.reset_parameters()
    
    @jaxtyped(typechecker=typechecked)
    def forward_discriminator(self, x: Float[Tensor, 'num_samples num_nodes num_input_features'],
                     acquisition: bool=False) -> Tuple[
                                                       Float[Tensor, 'num_samples num_nodes num_embeddings'] | None,
                                                       Float[Tensor, 'num_samples num_nodes num_classes']]:
        """ Computes the discriminator logits. We output k logits as the k+1-th logit is defined to be zero. """
        embedding = x
        latent = None
        for layer_idx, layer in enumerate(self.discriminator):
            embedding = layer(x)
            if layer_idx < len(self.discriminator) - 1:
                embedding = F.leaky_relu(embedding)
                if self.discriminator_dropout:
                    embedding = F.dropout(embedding, p=self.discriminator_dropout, training=self.training)
            if layer_idx == len(self.discriminator) - 2:
                latent = embedding
        return latent, embedding

    @property
    def prediction_changes_at_eval(self):
        return self.gcn.dropout > 0

    @jaxtyped(typechecker=typechecked)
    def predict(self, batch: Data, acquisition: bool = False) -> Prediction:
        with torch.set_grad_enabled(not self._gcn_frozen):
            prediction = self.gcn.predict(batch, acquisition=True)
        assert prediction.embeddings is not None
        prediction.discriminator_embeddings, prediction.discriminator_logits = self.forward_discriminator(prediction.embeddings)
        return prediction
            
    