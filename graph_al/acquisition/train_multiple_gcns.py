
from graph_al.acquisition.base import BaseAcquisitionStrategy
from graph_al.acquisition.config import AcquisitionStrategyFromTrainMultipleGCNsConfig
from graph_al.utils.logging import get_logger

from pathlib import Path
import os
import torch
from collections import defaultdict

class AcquisitionStrategyFromTrainMultipleGCNs(BaseAcquisitionStrategy):
    """ Base for acquisition strategies that use results from `train_multiple_gcns.py`, i.e. results
    from several trainings of a model on random splits to find a good split / good pool by "brute-force" """
    
    
    def __init__(self, config: AcquisitionStrategyFromTrainMultipleGCNsConfig):
        super().__init__(config)
        self._load_results(config)
        
    def _load_results(self, config: AcquisitionStrategyFromTrainMultipleGCNsConfig):
        """ Loads all results.

        Args:
            config (AcquisitionStrategyFromBestSplitsConfig): the configuration that specifies the directory in which results lie
        """
        masks_train = []
        masks_test = []
        metrics = defaultdict(list)
        
        results_dir = Path(config.results_directory)
        for file in os.listdir(config.results_directory):
            if file.endswith('.pt'):
                x = torch.load(results_dir / file)
                masks_train.append(x['masks_train'])
                masks_test.append(x['masks_test'])
                for key, value in x.items():
                    if key not in ('masks_train', 'masks_val', 'masks_test'):
                        metrics[key].append(value)
        
        self.masks_train = torch.cat(masks_train, dim=0)
        self.masks_test = torch.cat(masks_test, dim=0)
        self.metrics = {}
        for key, metric in metrics.items():
            metric = torch.tensor(sum(metric, start=[]))
            if len(metric.size()) == 2:
                metric = metric.mean(dim=-1) # Average over different model runs
            self.metrics[key] = metric
        
        get_logger().info('Acquisition strategy loaded best splits')