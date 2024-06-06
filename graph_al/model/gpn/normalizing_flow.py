# Partially taken from: https://github.com/stadlmax/Graph-Posterior-Network

import torch.nn as nn
from pyro.distributions.util import copy_docs_from
from pyro.distributions.torch_transform import TransformModule
from torch.distributions import Transform, constraints
import torch.nn.functional as F
from torch import nn
import torch
import math

from jaxtyping import jaxtyped, Float
from typeguard import typechecked
from torch import Tensor

from graph_al.utils.seed import next_generator

@copy_docs_from(Transform)
class ConditionedRadial(Transform):
    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True
    event_dim = 1 # type: ignore

    def __init__(self, params):
        super().__init__(cache_size=1)
        self._params = params
        self._cached_logDetJ = None

    def _call(self, x):
        """
        :param x: the input into the bijection
        :type x: torch.Tensor
        Invokes the bijection x=>y; in the prototypical context of a
        :class:`~pyro.distributions.TransformedDistribution` `x` is a sample from the base distribution (or the output
        of a previous transform)
        """
        x0, alpha_prime, beta_prime = self._params() if callable(self._params) else self._params

        # Ensure invertibility using approach in appendix A.2
        alpha = F.softplus(alpha_prime)
        beta = -alpha + F.softplus(beta_prime)

        # Compute y and logDet using Equation 14.
        diff = x - x0[:, None, :]
        r = diff.norm(dim=-1, keepdim=True).squeeze()
        h = (alpha[:, None] + r).reciprocal()
        h_prime = - (h ** 2)
        beta_h = beta[:, None] * h

        self._cached_logDetJ = ((x0.size(-1) - 1) * torch.log1p(beta_h) +
                                torch.log1p(beta_h + beta[:, None] * h_prime * r))
        return x + beta_h[:, :, None] * diff

    def _inverse(self, y):
        """
        :param y: the output of the bijection
        :type y: torch.Tensor
        Inverts y => x. As noted above, this implementation is incapable of
        inverting arbitrary values `y`; rather it assumes `y` is the result of a
        previously computed application of the bijector to some `x` (which was
        cached on the forward call)
        """

        raise KeyError("ConditionedRadial object expected to find key in intermediates cache but didn't")

    def log_abs_det_jacobian(self, x, y):
        """
        Calculates the elementwise determinant of the log Jacobian
        """
        x_old, y_old = self._cached_x_y
        if x is not x_old or y is not y_old:
            # This call to the parent class Transform will update the cache
            # as well as calling self._call and recalculating y and log_detJ
            self(x)

        return self._cached_logDetJ


@copy_docs_from(ConditionedRadial)
class Radial(ConditionedRadial, TransformModule):

    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True
    event_dim = 1

    def __init__(self, c, input_dim):
        super().__init__(self._params)

        self.x0 = nn.Parameter(torch.Tensor(c, input_dim,))
        self.alpha_prime = nn.Parameter(torch.Tensor(c,))
        self.beta_prime = nn.Parameter(torch.Tensor(c,))
        self.c = c
        self.input_dim = input_dim
        self.reset_parameters()

    def _params(self):
        return self.x0, self.alpha_prime, self.beta_prime

    def reset_parameters(self, generator: torch.Generator | None = None):
        generator = None # Note that if we use the supplied generator, for some ungodly reason the parameters will correlate and the NF won't learn...
        stdv = 1. / math.sqrt(self.x0.size(1))
        self.alpha_prime.data.uniform_(-stdv, stdv, generator=generator)
        self.beta_prime.data.uniform_(-stdv, stdv, generator=generator)
        self.x0.data.uniform_(-stdv, stdv, generator=generator)


class BatchedNormalizingFlowDensity(nn.Module):
    """layer of normalizing flows density which calculates c densities in a batched fashion"""

    def __init__(self, c, dim, flow_length):
        super(BatchedNormalizingFlowDensity, self).__init__()
        self.c = c
        self.dim = dim
        self.flow_length = flow_length
        self.transforms = nn.Sequential(*(
            Radial(c, dim) for _ in range(flow_length)
        ))
        
    def reset_parameters(self, generator: torch.Generator | None = None):
        for transform in self.transforms:
            transform.reset_parameters(generator=None) # type: ignore

    def forward(self, z):
        sum_log_jacobians = 0
        z = z.repeat(self.c, 1, 1)
        for idx, transform in enumerate(self.transforms):
            z_next = transform(z)
            log_jacobian = transform.log_abs_det_jacobian(z, z_next) # type: ignore
            sum_log_jacobians = sum_log_jacobians + log_jacobian # type: ignore
            z = z_next

        return z, sum_log_jacobians

    @jaxtyped(typechecker=typechecked)
    def log_prob(self, x: Float[Tensor, 'num_nodes flow_dim']) -> Float[Tensor, 'num_nodes num_classes']:
        z, sum_log_jacobians = self.forward(x)
        # Use fast calculation of the standard normal pdf N(0, I)
        # log(N(0, I)) = -0.5 * (d * log(2pi) + x^T x)
        log_prob_z = -0.5 * (math.log(math.pi * 2) * z.size(-1) + z.norm(p=2, dim=-1)**2)
        log_prob_x = log_prob_z + sum_log_jacobians  # [batch_size]
        log_prob_x = log_prob_x.transpose(0, 1)
        
        if not self.training:
            # If we're evaluating and observe a NaN value, this is always caused by the
            # normalizing flow "diverging". We force these values to minus infinity.
            log_prob_x[torch.isnan(log_prob_x)] = float('-inf')

        return log_prob_x