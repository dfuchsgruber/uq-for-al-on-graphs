# Partially taken from: https://github.com/stadlmax/Graph-Posterior-Network

import math
from torch import Tensor
import torch.nn as nn
from graph_al.model.config import GPNEvidenceScale

from jaxtyping import jaxtyped, Float
from typeguard import typechecked
from torch import Tensor

class Evidence(nn.Module):
    """layer to transform density values into evidence representations according to a predefined scale"""

    def __init__(self,
                 scale: GPNEvidenceScale,
                 tau: float | None = None):
        super().__init__()
        self.tau = tau
        self.scale = scale

    def __repr__(self):
        return f'Evidence(tau={self.tau}, scale={self.scale})'

    @jaxtyped(typechecker=typechecked)
    def forward(self, log_q: Float[Tensor, 'num_nodes num_classes'], dim: int, further_scale: float=1.0) -> Float[Tensor, 'num_nodes num_classes']:

        scaled_log_q = log_q + self.log_scale(dim, further_scale=further_scale)
        if self.tau is not None:
            scaled_log_q = self.tau * (scaled_log_q / self.tau).tanh()
        scaled_log_q = scaled_log_q.clamp(min=-30.0, max=30.0)
        return scaled_log_q

    @typechecked
    def log_scale(self, dim: int, further_scale: float = 1.) -> float:
        scale = 0

        match self.scale:
            case GPNEvidenceScale.LATENT_OLD:
                scale = 0.5 * (dim * math.log(2 * math.pi) + math.log(dim + 1))
            case GPNEvidenceScale.LATENT_NEW | GPNEvidenceScale.LATENT_NEW_PLUS_CLASSES:
                scale = 0.5 * dim * math.log(4 * math.pi)
        scale = scale + math.log(further_scale)

        return scale
