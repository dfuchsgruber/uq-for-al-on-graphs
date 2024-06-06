import torch
import numpy as np
from itertools import takewhile, zip_longest

from typing import Callable, Any, Tuple, Sequence, Dict
from jaxtyping import jaxtyped, Shaped

from typeguard import typechecked

def apply_to_nested(x: Any, fn: Callable, filter_: Callable=lambda _: True) -> Any:
    """ Applies a function to (nested) torch.Tensor instances. """
    if isinstance(x, Tuple):
        return tuple(*(apply_to_nested(t, fn, filter_) for t in x))
    elif isinstance(x, Sequence):
        return [apply_to_nested(t, fn, filter_) for t in x]
    elif isinstance(x, Dict):
        return {k : apply_to_nested(v, fn, filter_) for k, v in x.items()}
    elif filter_(x):
        return fn(x)
    return x

def apply_to_tensor(x: Any, fn: Callable) -> Any:
    if isinstance(x, torch.Tensor):
        return fn(x)
    else:
        return x
    
def apply_to_nested_tensors(x: Any, fn: Callable, filter_: Callable=lambda _: True) -> Any:
    return apply_to_nested(x, lambda t: apply_to_tensor(t, fn), filter_=filter_)

@typechecked
def apply_to_optional_tensors(fn: Callable, tensors: Sequence[torch.Tensor | None]) -> Any | None:
    """ Applies a function to a sequence of optional tensors. """
    if not (all(tensor is None for tensor in tensors) or all(tensor is not None for tensor in tensors)):
        raise ValueError(f'Sequence of optional tensors must be homogenous w.r.t. being None')
    if len(tensors) == 0 or tensors[0] is None:
        return None
    else:
        return fn(tensors)
    
def batched(n, iterable):
    '''batched(3, 'ABCDEFG') --> ('A', 'B', 'C'), ('D', 'E', 'F'),  ('G',)
    
    From: https://stackoverflow.com/questions/24527006/split-a-generator-into-chunks-without-pre-walking-it
    '''
    fillvalue = object()  # Anonymous sentinel object that can't possibly appear in input
    args = (iter(iterable),) * n
    for x in zip_longest(*args, fillvalue=fillvalue):
        if x[-1] is fillvalue:
            # takewhile optimizes a bit for when n is large and the final
            # group is small; at the cost of a little performance, you can
            # avoid the takewhile import and simplify to:
            # yield tuple(v for v in x if v is not fillvalue)
            yield tuple(takewhile(lambda v: v is not fillvalue, x))
        else:
            yield x

@jaxtyped(typechecker=typechecked)
def ndarray_to_tuple(x: Shaped[np.ndarray, '...']) -> Tuple:
    """ Transforms a ndarray to a tuple. """
    # do the actual transformation recursively without typechecks
    def inner(a):
        if isinstance(a, np.ndarray):
            return tuple(inner(i) for i in a)
        else:
            return a
    return inner(x)
    
    
    
    