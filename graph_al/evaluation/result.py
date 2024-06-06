from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Literal, Set

from jaxtyping import Int, Float, jaxtyped
from typeguard import typechecked
from torch import Tensor

from graph_al.evaluation.enum import MetricName, MetricTemplate
import torch

@dataclass
class Result:
    """ Storage class for the results of a single model training. """
    
    metrics: Dict[MetricTemplate, Any]
    
    acquired_class_counts: Int[Tensor, 'num_classes']
    acquisition_step: int
    acquisition_metrics: Dict[str, Any] = field(default_factory=dict)
    acquired_idxs: Int[Tensor, 'num_acquired'] | None = None
    
    @property
    @jaxtyped(typechecker=typechecked)
    def acquired_class_distribution(self) -> Float[Tensor, 'num_classes']:
        return self.acquired_class_counts / self.acquired_class_counts.sum()
    
    @property
    @typechecked
    def acquired_class_distribution_entropy(self) -> float:
        p = self.acquired_class_distribution
        return -(p * (p + 1e-12).log()).sum().item()
    
    @property
    @typechecked
    def num_acquired(self) -> int:
        return int(self.acquired_class_counts.sum().item())


@dataclass
class Results:
    """ Storage class for all results over an entire acquisition (i.e. active learning run) """
    
    results: List[Result]
    dataset_split_num: int | None = None
    model_initialization_num: int | None = None
    
    def __len__(self) -> int:
        return len(self.results)
    
    def __post_init__(self):
        # Check that the acquired class count dimensions match
        num_classes = set(result.acquired_class_counts.size(0) for result in self.results)
        assert len(num_classes) <= 1, f'Individual results have class counts for different number of classes {num_classes}'
    
    @property
    def metrics(self) -> Set[MetricTemplate]:
        return set.union(*[set(result.metrics.keys()) for result in self.results])
    
    def __contains__(self, key: MetricTemplate) -> bool:
        match key.name:
            case MetricName.NUM_ACQUIRED | MetricName.ACQUIRED_CLASS_DISTRIBUTION \
                | MetricName.ACQUIRED_CLASS_DISTRIBUTION_ENTROPY | MetricName.ACQUIRED_CLASS_COUNTS:
                return True
        return len(self.results) > 0 and any(key in r.metrics for r in self.results)
    
    @jaxtyped(typechecker=typechecked)
    def __getitem__(self, key: MetricTemplate) -> \
        Int[Tensor, 'num_acquisitions num_classes'] | Float[Tensor, 'num_acquisitions'] | Int[Tensor, 'num_acquisitions']:
        match key.name:
            case MetricName.NUM_ACQUIRED:
                return torch.tensor([result.num_acquired for result in self.results])
            case MetricName.ACQUIRED_CLASS_DISTRIBUTION:
                return torch.stack([result.acquired_class_distribution for result in self.results])
            case MetricName.ACQUIRED_CLASS_DISTRIBUTION_ENTROPY:
                return torch.tensor([result.acquired_class_distribution_entropy for result in self.results])
            case MetricName.ACQUIRED_CLASS_COUNTS:
                return torch.stack([result.acquired_class_counts for result in self.results])
        if len(self.results) == 0 or not any(key in r.metrics for r in self.results):
            raise IndexError(f'Key {key} not in {self.__class__}')
        else:
            return torch.tensor([result.metrics.get(key, torch.tensor(torch.nan)) for result in self.results])
        
        
        
        