from typing import Dict, Tuple
from jaxtyping import Bool, Int, jaxtyped
from typing import Any
from torch import Generator, Tensor
from typeguard import typechecked
from graph_al.acquisition.base import BaseAcquisitionStrategy
from graph_al.acquisition.config import AcquisitionStrategyGalaxyConfig
from graph_al.acquisition.galaxy.graph import MultiLinearGraph
from graph_al.acquisition.galaxy.linear_graph import create_linear_graphs
from graph_al.acquisition.galaxy.s2algorithm import bisection_query
from graph_al.data.base import Dataset
from graph_al.data.enum import DatasetSplit
from graph_al.model.base import BaseModel
from graph_al.model.config import ModelConfig
from graph_al.model.prediction import Prediction
import torch
import numpy as np
from graph_al.utils.logging import get_logger

class AcquisitionStrategyGalaxy(BaseAcquisitionStrategy):
    """Implementation of the GALAXY startegy.
    
    https://proceedings.mlr.press/v162/zhang22k.html
    """
    
    def __init__(self, config: AcquisitionStrategyGalaxyConfig):
        super().__init__(config)
        self.graphs: MultiLinearGraph | None = None
        self.idx_to_graphs_idx = None
        self.graph_idx_to_idx = None
        self.order = config.order
        
        
    def _build_graphs(self, prediction: Prediction, dataset: Dataset, generator: Generator):
        # In contrast to the original implementation, we keep a test and validation set
        # so we build the graph only on the train pool
        self.graph_idx_to_idx = []
        self.idx_to_graphs_idx = {}
        non_eval_idxs = torch.where(~(dataset.data.get_mask(DatasetSplit.VAL) |
                                dataset.data.get_mask(DatasetSplit.TEST)))[0].tolist()
        
        for graph_idx, idx in enumerate(non_eval_idxs):
            self.graph_idx_to_idx.append(idx)
            self.idx_to_graphs_idx[idx] = graph_idx
        
        scores = prediction.get_probabilities(propagated=True).mean(0).cpu().numpy()[np.array(self.graph_idx_to_idx)]
        labels = dataset.data.y.cpu().numpy()[np.array(self.graph_idx_to_idx)]
        
        self.graphs = create_linear_graphs(
            scores,
            labels,
            'graph',
            self.order,
        )
        
        for idx in torch.where(dataset.data.get_mask(DatasetSplit.TRAIN))[0]:
            self.graphs.label(self.idx_to_graphs_idx[int(idx.item())])
       
    @jaxtyped(typechecker=typechecked)
    def acquire(self, model: BaseModel, dataset: Dataset, num: int, model_config: ModelConfig, generator: Generator) -> Tuple[Int[Tensor, ' num'], Dict[str, Any]]:
        # rebuild the graphs that are used for acquisition
        prediction = model.predict(dataset.data, acquisition=True)
        self._build_graphs(prediction, dataset, generator)
        return super().acquire(model, dataset, num, model_config, generator)
        
    @jaxtyped(typechecker=typechecked)
    def acquire_one(self, mask_acquired: Bool[Tensor, 'num_nodes'], prediction: Prediction | None, model: BaseModel, dataset: Dataset, model_config: ModelConfig, 
            generator: Generator) -> Tuple[int, Dict[str, Tensor | None]]:
        assert self.graphs is not None, "Graphs are not initialized."
        assert self.graph_idx_to_idx is not None, "Graph idx to idx is not initialized."
        query_idx, alter_idx = bisection_query(self.graphs)
        if query_idx is None:
            query_idx = alter_idx
        self.graphs.label(query_idx)
        return self.graph_idx_to_idx[query_idx], {}
        
        
        