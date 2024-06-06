from graph_al.acquisition.config import AcquisitionStrategyBestSplitConfig, AcquisitionStrategyBestOrderedSplitConfig
from graph_al.acquisition.fixed_sequence import BaseAcquisitionStrategyFixedSequence
from graph_al.acquisition.train_multiple_gcns import AcquisitionStrategyFromTrainMultipleGCNs
from graph_al.acquisition.random import AcquisitionStrategyRandom
from graph_al.data.base import Dataset
from graph_al.model.base import BaseModel

from graph_al.utils.logging import get_logger

from pathlib import Path
import os
import torch
from collections import defaultdict

from typeguard import typechecked
from jaxtyping import jaxtyped, Bool, Int
from torch import Tensor
from typing import Tuple, Dict

class AcquisitionStrategyBestSplit(AcquisitionStrategyFromTrainMultipleGCNs, AcquisitionStrategyRandom):
    """ Base for acquisition strategies that use results from `train_multiple_gcns.py`, i.e. results
    from several trainings of a model on random splits to find a good split / good pool by "brute-force" """
    
    
    def __init__(self, config: AcquisitionStrategyBestSplitConfig):
        super().__init__(config)
        self.metric = config.metric
        self.higher_is_better = config.higher_is_better
        
        metric = self.metrics[self.metric]
        if self.higher_is_better:
            metric = -metric
        best_split_idx = metric.argmin(0)
        self.mask_best_split = self.masks_train[best_split_idx]
        get_logger().info(f'Picked best split with metric {self.metric} of {self.metrics[self.metric][best_split_idx]:.3f}')
        get_logger().info(f'Split has {self.mask_best_split.numpy().sum()} nodes.')
        
    @property
    def mask_not_in_val(self) -> Bool[Tensor, 'num_nodes'] | None:
        return self.mask_best_split
        
    @jaxtyped(typechecker=typechecked)
    def base_sampling_mask(self, model: BaseModel, dataset: Dataset, generator: torch.Generator) -> Bool[Tensor, 'num_nodes']:
        return self.mask_best_split.to(dataset.data.mask_train_pool.device)
    
class AcquisitionStrategyBestOrderedSplit(BaseAcquisitionStrategyFixedSequence, AcquisitionStrategyFromTrainMultipleGCNs):
    """ Acquisition strategy that uses the best sequence of nodes from the results of `optimize_best_split_order.py` """

    def __init__(self, config: AcquisitionStrategyBestOrderedSplitConfig, num_nodes: int):
        self._load_results(config)
        self.metric = config.metric
        self.higher_is_better = config.higher_is_better
        self.delta_metric_penality = config.delta_metric_penality

        values = self.values
        if self.higher_is_better:
            values = -values

        average = values.mean(-1)
        # penalty term for increasing / decreasing metric if higher is not better / higher is better
        difference = values[:, 1:] - values[:, :-1]
        penality = (torch.exp(difference) * (difference > 0)).sum(-1)
        score = average + self.delta_metric_penality * penality

        best_order_idx = score.argmin()
        self.best_order = self.orders[best_order_idx]
        super().__init__(config, self.best_order.tolist(), num_nodes)

        get_logger().info(f'Picked best order with score {score[best_order_idx]}: Average {self.metric}: {average[best_order_idx]}. Penality term: {penality[best_order_idx]}.')
        get_logger().info(f'Metric curve is ' + ', '.join(f'{value:.2f}' for value in values[best_order_idx].tolist()))
        get_logger().info(f'Node sequence is {self.sequence}')
        get_logger().info(f'Split has {self.best_order.size(0)} nodes.')

    def _load_results(self, config: AcquisitionStrategyBestOrderedSplitConfig):
        orders, values = [], []
        results_dir = Path(config.results_directory)
        for file in os.listdir(results_dir):
            if file.endswith('.pt'):
                data = torch.load(results_dir / file)
                orders.append(data['orders'])
                values.append(data[str(config.metric)].mean(-1))
        
        self.orders = torch.cat(orders, dim=0)
        self.values = torch.cat(values, dim=0)
        get_logger().info('Acquisition strategy loaded best order')


        