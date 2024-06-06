from graph_al.acquisition.base import BaseAcquisitionStrategy
from graph_al.acquisition.config import OracleConfig, OracleAcquisitionUncertaintyType
from graph_al.data.base import Dataset
from graph_al.data.sbm import SBMDataset
from graph_al.model.bayes_optimal import BayesOptimal
from graph_al.model.config import ModelConfig
from graph_al.model.prediction import Prediction
from graph_al.model.trainer.config import TrainerConfig
from graph_al.utils.logging import get_logger
from graph_al.data.config import DatasetSplit

from jaxtyping import Float, Int, jaxtyped, Bool, Shaped
from typeguard import typechecked
from torch import Tensor
from typing import Dict, Tuple, Any

import torch
import numpy as np

class AcquisitionStrategyOracle(BaseAcquisitionStrategy):
    """ Strategy that uses k-greedy approximation to Coreset [1] for sampling.
     
    When acquiring only one node at a time, this approximation is optimal
     
    References:
    [1] : https://openreview.net/pdf?id=H1aIuk-RW
    """
    
    def __init__(self, config: OracleConfig):
        super().__init__(config)
        self.uncertainty = config.uncertainty
        
    @torch.no_grad()
    @jaxtyped(typechecker=typechecked)
    def acquire(self, 
                model: BayesOptimal, 
                dataset: Dataset, 
                num: int,  
                model_config: ModelConfig,
                generator: torch.Generator
                ) -> Tuple[Int[Tensor, 'num'], Dict[str, Any]]:
        assert isinstance(dataset.base, SBMDataset), "Only works with SBM dataset"
        # update graph
        model.graph =  dataset.base
        
        mask_train, mask_train_pool = dataset.data.mask_train.clone(), dataset.data.mask_train_pool.clone()
        prediction = model.predict(dataset.data, acquisition=True, which=DatasetSplit.TRAIN_POOL)
        idxs_train_pool = torch.where(mask_train_pool)[0]

        match self.uncertainty:
            case OracleAcquisitionUncertaintyType.EPISTEMIC:
                if prediction.epistemic_confidence is not None:
                    uncertainty = 1 / prediction.epistemic_confidence
                else:
                    assert prediction.aleatoric_confidence is not None, "Epistemic confidence is not supplied and needs to be calculated as ratio of total / aleatoric"
                    assert prediction.total_confidence is not None, "Epistemic confidence is not supplied and needs to be calculated as ratio of total / aleatoric"
                    uncertainty = prediction.aleatoric_confidence / prediction.total_confidence
            case OracleAcquisitionUncertaintyType.TOTAL:
                assert prediction.total_confidence is not None, "Total confidence not supplied"
                uncertainty = 1 / prediction.total_confidence
            case OracleAcquisitionUncertaintyType.ALEATORIC:
                assert prediction.aleatoric_confidence is not None, "Aleatoric confidence not supplied"
                uncertainty = 1 / prediction.aleatoric_confidence
            case uncertainty_type:
                raise ValueError(f'Unsupported uncertainty type {uncertainty_type}')
        
        acquisition = uncertainty[torch.arange(model.graph.num_nodes), model.graph.labels]
        
        idxs_train_pool = torch.where(mask_train_pool)[0]
        possible_acq = acquisition[mask_train_pool].numpy()
        
        best_nodes = idxs_train_pool[np.argsort(-possible_acq)[:num]]
        
        mask_train[best_nodes] = True
        mask_train_pool[best_nodes] = False
        
        acquired_idxs = best_nodes
        assert (torch.unique(acquired_idxs, return_counts=True)[1] == 1).all()
        assert dataset.data.mask_train_pool[acquired_idxs].all()
        metrics = {}
        if prediction.approximation_error is not None:
            metrics['approximation_error'] = prediction.approximation_error
        if prediction.confidence_residual is not None:
            metrics['confidence_residual'] = prediction.confidence_residual
        
        return acquired_idxs, {
            'mask_train' : mask_train, 
            'mask_train_pool' : mask_train_pool,
            'alea_confidence': torch.tensor(prediction.aleatoric_confidence),
            'total_confidence': torch.tensor(prediction.total_confidence),} | metrics