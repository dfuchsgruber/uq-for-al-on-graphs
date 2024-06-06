from copy import deepcopy
from graph_al.acquisition.config import AcquisitionStrategyGEEMConfig
from graph_al.data.base import Data, Dataset
from graph_al.data.config import DatasetSplit
from graph_al.model.base import BaseModel
from graph_al.model.prediction import Prediction
from graph_al.acquisition.base import BaseAcquisitionStrategy
from graph_al.model.config import ModelConfig

from jaxtyping import Shaped, jaxtyped, Bool
from typeguard import typechecked
from typing import Dict, List, Tuple
from torch import Tensor

import itertools
from functools import partial
import torch
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from graph_al.utils.logging import get_logger

from graph_al.utils.utils import batched
from graph_al.utils.timer import Timer

class AcquisitionStrategyGraphExpectedErrorMinimization(BaseAcquisitionStrategy):
    
    """ Acquisition strategy that uses expected error minimization on graphs [1] to acquire a node. 
    
    References:
    [1]: https://arxiv.org/pdf/2007.05003.pdf
    """
    
    def __init__(self, config: AcquisitionStrategyGEEMConfig):
        super().__init__(config)
        assert not self.balanced, f'GEEM does not support balanced acquisition currently'
        self.multiprocessing = config.multiprocessing
        self.num_workers = config.num_workers
        self.compute_risk_on_subset = config.compute_risk_on_subset
        self.subsample_pool = config.subsample_pool
        
    def acquire_one(self, mask_acquired: Bool[Tensor, 'num_nodes'], prediction: Prediction | None, model: BaseModel, dataset: Dataset,
            model_config: ModelConfig, generator: torch.Generator) -> Tuple[int, Dict[str, Tensor | None]]:

        assert prediction is not None, 'GEEM expects a prediction'
        probabilities = prediction.get_probabilities(propagated=True)
        assert probabilities is not None, 'GEEM expects probabilities'
        probabilities = probabilities.mean(0) # ensemble average
        # For each node and potential label calculate the risk
        mask_predict_nodes = np.ones(dataset.num_nodes, dtype=bool)
        mask_predict_nodes &= dataset.data.get_mask(DatasetSplit.TRAIN_POOL).numpy()
        if self.subsample_pool is not None:
            idxs_predict_nodes = np.where(mask_predict_nodes)[0]
            np.random.shuffle(idxs_predict_nodes)
            idxs_predict_nodes = idxs_predict_nodes[:self.subsample_pool]
            mask_predict_nodes = np.zeros_like(mask_predict_nodes)
            mask_predict_nodes[idxs_predict_nodes] = True

        risk = torch.full_like(probabilities, np.inf)

        if self.multiprocessing:
            args = list(itertools.product(np.where(mask_predict_nodes)[0], range(dataset.data.num_classes)))
            args = list(zip(args, [generator.seed() for _ in range(len(args))]))
            chunk_size = int(np.ceil(len(args) / (self.num_workers or cpu_count())))
            batched_args = [(x, deepcopy(model), deepcopy(dataset)) for x in batched(chunk_size, args)]
            job = partial(compute_risk_job, verbose=self.verbose, compute_risk_on_subset=self.compute_risk_on_subset)
            with Pool(processes=self.num_workers) as pool:
                results = pool.starmap(job, batched_args)
            for result in results:
                for (i, c), risk_i_c in result:
                    risk[i, c] = risk_i_c # type: ignore
            assert torch.isfinite(risk[mask_predict_nodes]).all()
        else:
            args = [((i, c), generator.seed()) for i, c in itertools.product(np.where(mask_predict_nodes)[0], range(dataset.data.num_classes))]
            result = compute_risk_job(args, deepcopy(model), deepcopy(dataset), verbose=self.verbose, compute_risk_on_subset=self.compute_risk_on_subset)
            for (i, c), risk_i_c in result:
                risk[i, c] = risk_i_c # type: ignore
            assert torch.isfinite(risk[mask_predict_nodes]).all()

        proxy = (risk * probabilities).sum(1)
        return int(proxy.argmin().item()), {'risk' : risk, 'probabilities' : probabilities}
    

def compute_risk_job(jobs: List[Tuple[Tuple[int, int], int]], model: BaseModel, dataset: Dataset, verbose: bool=False,
        compute_risk_on_subset: int | None = None) -> List[Tuple[Tuple[int, int], float]]:
    from graph_al.model.sgc import SGC
    from sklearn.linear_model import LogisticRegression

    assert isinstance(model, SGC), 'currently, we only support SGC for threaded GEEM'

    batch = dataset.data
    mask_train = batch.get_mask(DatasetSplit.TRAIN).numpy()
    x = model.get_diffused_node_features(batch).numpy()
    y = dataset.data.y.numpy()

    mask_risk = ~mask_train
    if compute_risk_on_subset is not None: # For efficiency reasons, compute the risk only on a subset
        idxs_risk = np.where(mask_risk)[0]
        np.random.shuffle(idxs_risk)
        idxs_risk = idxs_risk[:compute_risk_on_subset]
        mask_risk = np.zeros_like(mask_risk)
        mask_risk[idxs_risk] = True

    risks = []
    for (i, c), seed in tqdm(jobs, desc='Parallel computing expected risk', disable=not verbose):
        assert not mask_train[i]
        label_i_true, y[i], mask_train[i] = dataset.data.y[i], c, True
        if np.unique(y[mask_train]).shape[0] == 1:
            # Weird "bug" in LogisticRegression that it can not fit with only one class
            # The prediction of the logistic regression classifier should be 1.0 for this one class
            probabilities = np.zeros((dataset.num_nodes, dataset.data.num_classes))
            probabilities[:, c] = 1.0
        else:
            regression = LogisticRegression(C=model.inverse_regularization_strength, solver=model.solver, # type: ignore
                class_weight='balanced' if model.balanced else None)
            regression.fit(x[mask_train], y[mask_train])
            probabilities = regression.predict_proba(x)
            del regression

        risks.append(((i, c), 1 - probabilities[mask_risk].max(axis=1).mean()))
        y[i], mask_train[i] = label_i_true, False
    if verbose:
        print()
    return risks




