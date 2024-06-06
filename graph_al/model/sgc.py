from graph_al.data.base import Dataset
from graph_al.data.generative import GenerativeDataset
from graph_al.data.config import DatasetSplit
from graph_al.model.base import BaseModel
from graph_al.model.config import SGCConfig
from graph_al.model.prediction import Prediction
from graph_al.utils.logging import get_logger
from graph_al.data.base import Data
from graph_al.model.config import ApproximationType

import numpy as np
import torch
import torch_scatter
import torch_geometric.nn as tgnn
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import NotFittedError

import itertools
from scipy.special import logsumexp
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
import math
from copy import deepcopy

from jaxtyping import jaxtyped, Float, Int, Bool, UInt64
from typeguard import typechecked
from typing import Tuple, Any

from graph_al.utils.utils import batched


class SGC(BaseModel):
    """ Uses the simplified graph convolution framework: p = sigma(A^(k)XW) """

    def __init__(self, config: SGCConfig, dataset: Dataset, generator: torch.Generator):
        super().__init__(config, dataset)
        self.inverse_regularization_strength = config.inverse_regularization_strength
        self.cached = config.cached
        self.balanced = config.balanced
        self.normalize = True
        self.add_self_loops = config.add_self_loops
        self.improved = config.improved
        self.k = config.k
        self.solver = config.solver

        self.reset_cache()
        self._frozen_prediction = None
        self.reset_parameters(generator)
        
    def reset_cache(self):
        self._cached_node_features = None



    def reset_parameters(self, generator: torch.Generator):
        self.logistic_regression = LogisticRegression(C=self.inverse_regularization_strength, solver=self.solver,
            class_weight='balanced' if self.balanced else None)
        self._frozen_prediction = None

    @torch.no_grad()
    def predict(self, batch: Data, acquisition: bool = False) -> Prediction:
        if isinstance(self._frozen_prediction, int): # Prediction is frozen to this one class
            probs = np.zeros((batch.num_nodes, batch.num_classes), dtype=float) # type: ignore
            probs[:, self._frozen_prediction] = 1.0
            probs, probs_unpropagated = probs, probs
            logits, logits_unpropagated = probs, probs_unpropagated # a bit arbitrary...
        else:
            if self.logistic_regression is None:
                raise RuntimeError(f'No regression model was fitted for SGC')
            try:
                x = self.get_diffused_node_features(batch)
                probs = self.logistic_regression.predict_proba(x)
                probs_unpropagated = self.logistic_regression.predict_proba(batch.x.cpu().numpy())
                logits = self.logistic_regression.decision_function(x)
                logits_unpropagated = self.logistic_regression.decision_function(batch.x.cpu().numpy())
            except NotFittedError:
                get_logger().warn(f'Predictions with a non-fitted regression model: Fall back to uniform predictions')
                probs = np.ones((batch.num_nodes, batch.num_classes), dtype=float) / batch.num_classes # type: ignore
                probs, probs_unpropagated = probs, probs
                logits, logits_unpropagated = probs, probs_unpropagated # a bit arbitrary...
        return Prediction(probabilities=torch.from_numpy(probs[None, ...]), 
                          probabilities_unpropagated=torch.from_numpy(probs_unpropagated)[None, ...],
                          logits=torch.from_numpy(logits[None, ...]),
                          logits_unpropagated=torch.from_numpy(logits_unpropagated[None, ...]),
                          # they are not really embeddings, this is a bit iffy...
                          embeddings=torch.from_numpy(logits[None, ...]),
                          embeddings_unpropagated=torch.from_numpy(logits_unpropagated[None, ...])
                          )

    @torch.no_grad()
    @jaxtyped(typechecker=typechecked)
    def get_diffused_node_features(self, batch: Data) -> Float[torch.Tensor, 'num_nodes num_features']:
        """ Gets the diffused node features. """
        return batch.get_diffused_nodes_features(self.k, normalize=self.normalize, improved=self.improved,
                    add_self_loops=self.add_self_loops, cache=True)

    def freeze_predictions(self, class_idx: int):
        self._frozen_prediction = class_idx
    
    def unfreeze_predictions(self):
        self._frozen_prediction = None

                

