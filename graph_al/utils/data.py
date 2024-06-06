import numpy as np
import scipy.sparse as sp
from jaxtyping import jaxtyped, Int, Bool
from typeguard import typechecked
from typing import Dict, Tuple

from graph_al.utils.logging import get_logger

@typechecked
def make_key_collatable(key: str) -> str:
    """ Makes a key collatable for tg. That is, it removes 
    substrings "batch", "index" and "face" (yes, stupid indeed...) """
    replaces = {
        'batch' : 'b#atch',
        'index' : 'i#ndex',
        'face' : 'f#ace'
    }
    for old, new in replaces.items():
        key = key.replace(old, new)
    return key

def make_mapping_collatable(d):
    """ Fixes the keys in a mapping so that they do not contain keys the tg collator is senstitive to.
    That includes the substrings 'batch', 'index', 'face' (yes, very stupid...)
    """
    result = {}
    for k, v in d.items():
        old_k, k = k, make_key_collatable(k)
        if k in result:
            raise RuntimeError(f'Duplicate keys {k} by sanitizing.')
        if k != old_k:
            get_logger().info(f'Making mapping collatable: Sanitized "{old_k}" to "{k}"')
        result[k] = v
    return result


@typechecked
def sparse_max(A: sp.spmatrix, B: sp.spmatrix) -> sp.spmatrix:
    """
    Return the element-wise maximum of sparse matrices `A` and `B`.

    References:
    -----------
    Taken from: https://stackoverflow.com/questions/19311353/element-wise-maximum-of-two-sparse-matrices
    """
    AgtB = (A > B).astype(int)
    M = AgtB.multiply(A - B) + B
    return M

@jaxtyped(typechecker=typechecked)
def compress_labels(labels: Int[np.ndarray, "n"], label_to_idx: Dict[str, int]) -> Tuple[Int[np.ndarray, "n"], Dict[str, int], Dict[int, int]]:
    """ Compresses the labels of a graph. For examples the labels {0, 1, 3, 4} will be re-mapped
    to {0, 1, 2, 3}. Labels will be compressed in ascending order. 
    
    
    Args:
        mask (Int[np.ndarray, &#39;n&#39;]): Labels to compress
        label_to_idx (Dict[str, int]): Mapping from label name to its index
        
    Returns:
        (Int[np.ndarray, &#39;n&#39;]): New labels
        (Dict[str, int]): New mapping from label name to index
        (Dict[int, int]): New mapping from old label index to compressed label index
    """
    label_set_sorted = np.sort(np.unique(labels))
    compression = {int(idx_old) : int(idx_new) for idx_new, idx_old in enumerate(label_set_sorted)}
    labels_new, label_to_idx_new = remap_labels(labels, label_to_idx, compression)
    return labels_new, label_to_idx_new, compression

def remap_labels(labels: Int[np.ndarray, "n"], label_to_idx: Dict[str, int], remapping: Dict[int, int], 
                 undefined_label: int=-1) -> Tuple[Int[np.ndarray, "n"], Dict[str, int]]:
    """ Remaps labels of a graph. 
    
    Args:
        mask (Int[np.ndarray, &#39;n&#39;]): Labels to remap
        label_to_idx (Dict[str, int]): Mapping from label name to its index
        remapping (Dict[int, int]): New mapping from old label index to new label index
        undefined_label (int): Labels that are not defined in the remapping will be assigned this value. Defaults to -1.
        
    Returns:
        (Int[np.ndarray, &#39;n&#39;]): New labels
        (Dict[str, int]): New mapping from label name to index
    """
    labels_new = np.ones_like(labels) * undefined_label
    label_to_idx_new = {}
    idx_to_label = {idx : label for label, idx in label_to_idx.items()}
    for label_old, label_new in remapping.items():
        labels_new[labels == label_old] = label_new
        label_to_idx_new[idx_to_label[label_old]] = label_new
    # assert not (labels_new == -1).any()
    return labels_new, label_to_idx_new

def sample_from_mask(mask: Bool[np.ndarray, "n"], num_samples: int, rng: np.random.RandomState | None=None) -> Bool[np.ndarray, "n"]:
    """ Samples a submask from a mask.
    
    Args:
        mask (Bool[np.ndarray, &#39;n&#39;]): the mask to sample from
        num_samples (int): How many to draw
        rng (np.random.RandomState | None): A random number generator, defaults to None.

    Returns:
        (Bool[np.ndarray, &#39;n&#39;]): The sampled submask
    """
    if rng is None:
        rng = np.random.RandomState()
    if mask.sum() < num_samples:
        raise RuntimeError(f'Can not sample {num_samples} vertices from a mask with {mask.sum()} vertices')
    idxs = np.where(mask)[0]
    rng.shuffle(idxs)
    sample_mask = np.zeros_like(mask, dtype=bool)
    sample_mask[idxs[ : num_samples]] = True
    return sample_mask
