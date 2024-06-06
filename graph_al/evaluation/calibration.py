import torch
import numpy as np

from jaxtyping import jaxtyped, Int, Float, Bool
from typeguard import typechecked
from typing import Tuple, Dict

from graph_al.model.prediction import Prediction
from graph_al.data.base import Data


@jaxtyped(typechecker=typechecked)
def compute_ece(probs: Float[torch.Tensor, 'num_nodes num_classes'], y_true: Int[torch.Tensor, 'num_nodes'], bins: int=10, 
    eps: float=1e-12) -> float:
    """ Calculates the expected calibration error.
    
    Args:
        probs (Float[Tensor, 'num_nodes num_classes']): the predicted probabilities
        y_true (int[Tensor, 'num_nodes']): true labels
        bins (int): How many bins to use
        eps (float): Small value
    """
    _, bin_confidence, bin_accuracy, bin_weight = calibration_curve(probs, y_true, bins=bins, eps=eps)
    return (bin_weight * np.abs(bin_confidence - bin_accuracy)).sum().item()


@jaxtyped(typechecker=typechecked)
def calibration_curve(probs: Float[torch.Tensor, 'num_nodes num_classes'], y_true: Int[torch.Tensor, 'num_nodes'], bins: int=10, 
    eps: float=1e-12) ->Tuple[Float[np.ndarray, 'num_bins_plus_one'], Float[np.ndarray, 'num_bins'], Float[np.ndarray, 'bins'], Float[np.ndarray, 'bins']]:
    """ Calculates the calibration curve for predictions.
    
    Args:
        probs (Float[Tensor, 'num_nodes num_classes']): the predicted probabilities
        y_true (int[Tensor, 'num_nodes']): true labels
        bins (int): How many bins to use
        eps (float): Small value

    Returns:
        bin_edges (Float[np.ndarray, 'bins+1']): bin edges
        bin_confidence (Float[np.ndarray, 'bins']): average confidence in each bin
        bin_accuracy (Float[np.ndarray, 'bins']): average accuracy in each bin
        bin_weight (Float[np.ndarray, 'bins']): weight of each bin
    """
    n, c = probs.size()
    c = max(c, int(y_true.max().item() + 1))
    max_prob, hard = probs.detach().cpu().max(dim=-1)
    y_true_one_hot = np.eye(c)[y_true.detach().cpu().numpy()]
    
    bin_edges = np.linspace(0., 1., bins + 1)
    bin_width = 1 / bins
    digitized = np.digitize(max_prob.numpy(), bin_edges)
    digitized = np.maximum(np.minimum(digitized, bins), 1) - 1 # Push values outside the bins into the rightmost and leftmost bins
    
    bins_sum = np.bincount(digitized, minlength=bins, weights=max_prob.numpy())
    bins_size = np.bincount(digitized, minlength=bins)
    is_correct = y_true_one_hot[range(n), hard]
    
    bin_confidence = bins_sum / (bins_size + eps)
    bin_accuracy = np.bincount(digitized, minlength=bins, weights=is_correct) / (bins_size + eps)
    bin_weight = bins_size / bins_size.sum()
    
    return bin_edges, bin_confidence, bin_accuracy, bin_weight