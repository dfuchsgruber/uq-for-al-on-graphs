import torch
import torch.nn.functional as F
from torch import Tensor
import torch_scatter

import graph_al.model.gpn.distributions as UD
import torch.distributions as D

from jaxtyping import jaxtyped, Int, Float
from typeguard import typechecked

@torch.no_grad()
@jaxtyped(typechecker=typechecked)
def balanced_loss_weights(labels: Int[Tensor, "num_samples"], beta: float=0.999, normalize: bool = True,
                          num_classes: int | None = None) -> Float[Tensor, "num_samples"]:
    """ Computes the class balanced loss weights per sample. 

    Args:
        labels (Int[Tensor, &quot;num_samples&quot;]): The label
        beta (float, optional): Smoothing. 0 means no reweighting and -> 1 approaches inverse class frequency . Defaults to 0.9999.
        normalize (bool, optional): If given, the class weights sum to `num_classes`. Defaults to True.
        
    Returns:
        (Float[Tensor, &quot;num_samples&quot;]): The per-instance loss weights.
        
    References:
        https://arxiv.org/pdf/1901.05555.pdf
    """
    counts = torch_scatter.scatter_sum(torch.ones_like(labels), labels, dim_size=num_classes)
    num_classes = counts.size(0)
    effective_counts = 1 - beta / (1 - torch.exp(counts * torch.log(torch.tensor(beta, device=counts.device))))
    if normalize:
        effective_counts *= num_classes / effective_counts[torch.isfinite(effective_counts)].sum()
    weights = effective_counts[labels]
    return weights
    
@jaxtyped(typechecker=typechecked)
def uce_loss(alpha: Float[Tensor, 'num_nodes num_classes'], y: Int[Tensor, 'num_nodes']) -> Float[Tensor, 'num_nodes']:
    """Computes the uncertainty cross entropy loss (UCE loss) for Graph Posterior Network models """
    a_sum = alpha.sum(-1)
    a_true = alpha.gather(-1, y.view(-1, 1)).squeeze(-1)
    uce = a_sum.digamma() - a_true.digamma()
    return uce
    
def entropy_regularization(alpha: Float[Tensor, 'num_nodes num_classes'], approximate: bool=True) -> Float[Tensor, 'num_nodes']:
    """Computes the regularizer for the entropy of the predicted distribution. """
    if alpha.size(0) == 0:
        return torch.tensor([], dtype=torch.float, device=alpha.device)
    if approximate:
        loss = UD.Dirichlet(alpha).entropy()
    else:
        loss = D.Dirichlet(alpha).entropy()
    return -loss
    
    
