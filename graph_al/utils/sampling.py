import torch
from torch import Tensor
from jaxtyping import Bool

from graph_al.utils.logging import get_logger

def sample_from_mask(mask: Bool[Tensor, 'num'], size: int | float,
                     generator: torch.Generator | None = None) -> Bool[Tensor, 'num']:
    """ Samples indices from a mask of possible indices.
    
    Args:
        mask (Bool[Tensor, &#39;num_nodes&#39;]): All available indices 
        size (int | float): How many (absolute or as a fraction) to sample
        generator (torch.Generator | None): A random number generator for sampling
        
    Returns:
        (Bool[Tensor, &#39;num_nodes&#39;]): Sampled indices
    """
    mask_size = mask.sum().item()

    result = torch.zeros_like(mask)
    if isinstance(size, float):
        size = int(size * mask_size)
    elif not isinstance(size, int):
        raise ValueError(f'Size to sample must either be int or float, but got {type(size)}')
    if size == 0:
        get_logger().warn(f'Sampling {size} (type {type(size)}) from a mask of size {mask_size}')
        return result
    elif size > mask_size:
        raise RuntimeError(f'Trying to sample {size} indices from a mask containing {mask_size} elements')
    else:
        indices = torch.where(mask)[0]
        indices = indices[torch.randperm(indices.size(0), generator=generator)[:size]]
        result[indices] = True
        return result
    
    
        
    