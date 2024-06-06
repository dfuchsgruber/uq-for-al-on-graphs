import torch
import torch_scatter
import math
from tqdm import tqdm

from jaxtyping import jaxtyped, Float, Int
from typeguard import typechecked
from torch import Tensor
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize

def approximate_ppr_matrix(edge_index: Int[Tensor, '2 num_edges'], edge_weights: Float[Tensor, 'num_edges'],
                           teleport_probability: float = 0.2, num_iterations: int = 10, verbose: bool=True,
                           num_nodes: int | None=None) -> Float[Tensor, 'num_nodes num_nodes']:
    """ approximates the ppr matrix by doing power iteration
    """
    if num_nodes is None:
        num_nodes = int(edge_index.max().item()) + 1
        
    idx_src, idx_target = edge_index
    # Check for a stochastic matrix
    sums = torch_scatter.scatter_add(edge_weights, idx_src, dim=-1, dim_size=num_nodes)
    assert torch.allclose(sums, torch.tensor(1.0), atol=1e-4), \
        f'Expected stochastic matrix for PPR approximation but got {sums[~torch.isclose(sums, torch.tensor(1.0), atol=1e-4)]}' + \
            f' at indices {torch.where(~torch.isclose(sums, torch.tensor(1.0), atol=1e-4))[0]}'

    
    edge_idxs = edge_index.cpu().numpy()
    A = sp.coo_matrix((edge_weights.cpu().numpy(), edge_idxs), shape=(num_nodes, num_nodes))

    Pi = np.ones((num_nodes, num_nodes)) / num_nodes
    pbar = (range(num_iterations))
    if verbose:
        pbar = tqdm(pbar)
    for it in pbar:
        new = (1 - teleport_probability) * (A.T @ Pi) + (teleport_probability / num_nodes) * np.eye(num_nodes)
        diff = np.linalg.norm(new - Pi)
        if verbose:
            pbar.set_description(f'APPR residuals: {diff:.5f}') # type: ignore
        Pi = new
    return torch.tensor(Pi)

@torch.no_grad()
@jaxtyped(typechecker=typechecked)
def approximate_ppr_scores(edge_index: Int[Tensor, '2 num_edges'], edge_weights: Float[Tensor, 'num_edges'],
                           teleport_probability: float = 0.2, num_iterations: int = 10, verbose: bool=True,
                           num_nodes: int | None=None) -> Float[Tensor, 'num_nodes']:
    """ Computes (approximate) per-node ppr centrality scores """
    
    if num_nodes is None:
        num_nodes = int(edge_index.max().item()) + 1
        
    idx_src, idx_target = edge_index
    # Check for a stochastic matrix
    sums = torch_scatter.scatter_add(edge_weights, idx_src, dim=-1, dim_size=num_nodes)
    assert torch.allclose(sums, torch.tensor(1.0), atol=1e-4), \
        f'Expected stochastic matrix for PPR approximation but got {sums[~torch.isclose(sums, torch.tensor(1.0), atol=1e-4)]}' + \
            f' at indices {torch.where(~torch.isclose(sums, torch.tensor(1.0), atol=1e-4))[0]}'
        
    edge_idxs = edge_index.cpu().numpy()
    A = sp.coo_matrix((edge_weights.cpu().numpy(), edge_idxs), shape=(num_nodes, num_nodes))

    page_rank_scores = np.ones(num_nodes) / num_nodes
    pbar = (range(num_iterations))
    if verbose:
        pbar = tqdm(pbar)
    for it in pbar:
        prev = page_rank_scores.copy() # type: ignore
        page_rank_scores = (1 - teleport_probability) * (A.T @ page_rank_scores) + (teleport_probability) / num_nodes
        diff = np.linalg.norm(prev - page_rank_scores)
        if verbose:
            pbar.set_description(f'APPR residuals: {diff:.5f}')# type: ignore
        assert np.allclose(page_rank_scores.sum(), 1.0, atol=1e-4)
    return torch.tensor(page_rank_scores)