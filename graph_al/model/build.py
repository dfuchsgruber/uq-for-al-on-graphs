from graph_al.model.config import ModelConfig, GCNConfig, APPNPConfig, BayesianGCNConfig, GPNConfig, BayesOptimalConfig, SGCConfig, SEALConfig
from graph_al.model.base import BaseModel, Ensemble
from graph_al.model.gcn import GCN
from graph_al.model.appnp import APPNP
from graph_al.model.bayesian_gcn import BayesianGCN
from graph_al.model.gpn import GraphPosteriorNetwork
from graph_al.model.bayes_optimal import BayesOptimal
from graph_al.model.sgc import SGC
from graph_al.data.base import Dataset
from graph_al.model.seal import SEAL

import torch

def _get_model(config: ModelConfig, dataset: Dataset, generator: torch.Generator) -> BaseModel:
    match config.type_:
        case GCNConfig.type_:
            return GCN(config, dataset) # type: ignore
        case APPNPConfig.type_:
            return APPNP(config, dataset) # type: ignore
        case BayesianGCNConfig.type_:
            return BayesianGCN(config, dataset) # type: ignore
        case GPNConfig.type_:
            return GraphPosteriorNetwork(config, dataset) # type: ignore
        case BayesOptimalConfig.type_:
            return BayesOptimal(config, dataset) # type: ignore
        case SGCConfig.type_:
            return SGC(config, dataset, generator) # type: ignore
        case SEALConfig.type_:
            return SEAL(config, dataset) # type: ignore
        case _:
            raise ValueError(f'Unsupported model type {config.type_}')
        
def get_model(config: ModelConfig, dataset: Dataset, generator: torch.Generator) -> BaseModel:
    if config.num_ensemble_members is not None and config.num_ensemble_members > 1:
        return Ensemble(config, dataset, [_get_model(config, dataset, generator) for _ in range(config.num_ensemble_members)])
    else:
        return _get_model(config, dataset, generator)