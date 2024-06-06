from copy import copy

from typing import List
from jaxtyping import Float, jaxtyped
from typeguard import typechecked

import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data

from graph_al.data.config import FeatureNormalization

@jaxtyped(typechecker=typechecked)
def normalize_features(features: Float[torch.Tensor, '...'], normalization: FeatureNormalization, dim: int=-1) -> Float[torch.Tensor, '...']:
    """ Normalizes features

    Args:
        features (Float[torch.Tensor, '...']): the features to normalize
        normalization (FeatureNormalization): which normalization
        dim (int): along which axis

    Returns:
        Float[torch.Tensor, '...']: the normalized features
    """
    match normalization:
        case FeatureNormalization.NONE:
            pass
        case FeatureNormalization.L1:
            features /= torch.linalg.norm(features, dim=dim, ord=1, keepdim=True)
        case FeatureNormalization.L2:
            features /= torch.linalg.norm(features, dim=dim, ord=2, keepdim=True)
        case normalization:
            raise ValueError(f'Unsupported normalization {normalization}')
    return features

class ReorderLabels(T.BaseTransform):
    """Reorders the labels of a data instance

    Args:
        order (List[int]): The order in which old classes are mapped. That is, all nodes with label order[i] will be assigned label i
    """

    def __init__(self, order: List[int]) -> None:
        super().__init__()
        assert len(order) == len(set(order)), f'Mutiple defintions of classes in {order}'
        assert set(order) == set(range(max(order) + 1)), f'Order should specify class indices from 0, ..., num_classes - 1, not {order}'
        self.order = order

    def __call__(self, data: Data) -> Data:
        data = copy(data) # shallow copy
        y = torch.full_like(data.y, fill_value=-1)
        for idx, y_i in enumerate(self.order):
            y[data.y == y_i] = idx
        assert (y >= 0).all()
        data.y = y
        return data

class ReorderLeftOutClassLabels(ReorderLabels):
    """ Transformation that reorders the labels such that 

    Args:
        ReorderLabels (_type_): _description_
    """
    def __init__(self, left_out_class_labels: List[int], num_classes_all: int):
        super().__init__(list(sorted(range(num_classes_all), key=lambda x: x in left_out_class_labels)))

        