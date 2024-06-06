

from graph_al.acquisition.config import AcquisitionStrategyBestSplitConfig, AcquisitionStrategyConfig, AcquisitionStrategyFixedSequenceConfig
from graph_al.acquisition.train_multiple_gcns import AcquisitionStrategyFromTrainMultipleGCNs
from graph_al.acquisition.random import AcquisitionStrategyRandom
from graph_al.acquisition.base import BaseAcquisitionStrategy
from graph_al.data.base import Dataset
from graph_al.model.base import BaseModel
from graph_al.model.prediction import Prediction

import torch

from typeguard import typechecked
from jaxtyping import jaxtyped, Bool, Int
from torch import Tensor
from typing import List, Tuple, Dict

from graph_al.model.config import ModelConfig

class BaseAcquisitionStrategyFixedSequence(BaseAcquisitionStrategy):

    def __init__(self, config: AcquisitionStrategyConfig, sequence: List[int], num_nodes: int):
        super().__init__(config)
        self.sequence = sequence
        self.num_nodes = num_nodes

    @property
    def mask_not_in_val(self) -> Bool[Tensor, 'num_nodes']:
        mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        mask[self.sequence] = True
        return mask

    def acquire_one(self, mask_acquired: Bool[Tensor, 'num_nodes'], prediction: Prediction | None, model: BaseModel, dataset: Dataset, 
                        model_config: ModelConfig,  generator: torch.Generator) -> Tuple[int, Dict[str, Tensor | None]]:
        pool = self.pool(mask_acquired, model, dataset, generator)
        for node_idx in self.sequence:
            if pool[node_idx]:
                return node_idx, {} # type: ignore
        else:
            raise RuntimeError((f'Can not add from the best split sequence anymore. Already in train',
                                torch.where(dataset.data.mask_train | pool)[0]))

class AcquisitionStrategyFixedSequence(BaseAcquisitionStrategyFixedSequence):
    """ Selects the best"""
    
    def __init__(self, config: AcquisitionStrategyFixedSequenceConfig, num_nodes: int):
        if config.order is not None:
            sequence = list(config.order)
            num_nodes = num_nodes
        elif config.order_path is not None:
            data = torch.load(config.order_path)
            sequence = data['sequence'].tolist()
            assert num_nodes == data['num_nodes']
        else:
            raise ValueError(f'Either supply a sequence of nodes to acquire or a file that lists the sequence.')
        super().__init__(config, sequence, num_nodes)
        
    @property
    def mask_not_in_val(self) -> Bool[Tensor, 'num_nodes']:
        mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        mask[self.sequence] = True
        return mask

    def acquire_one(self, mask_acquired: Bool[Tensor, 'num_nodes'], prediction: Prediction | None, model: BaseModel, dataset: Dataset, 
                        model_config: ModelConfig,  generator: torch.Generator) -> Tuple[int, Dict[str, Tensor | None]]:
        pool = self.pool(mask_acquired, model, dataset, generator)
        for node_idx in self.sequence:
            if pool[node_idx]:
                return node_idx, {} # type: ignore
        else:
            raise RuntimeError((f'Can not add from the best split sequence anymore. Already in train',
                                torch.where(dataset.data.mask_train | pool)[0]))
 