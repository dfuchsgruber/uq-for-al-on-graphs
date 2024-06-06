from typing import Dict, Tuple
from jaxtyping import Bool, jaxtyped
from torch import Generator, Tensor
from typeguard import typechecked
from graph_al.acquisition.base import BaseAcquisitionStrategy
from graph_al.acquisition.config import AcquisitionStrategyBadgeConfig
from graph_al.acquisition.galaxy.graph import MultiLinearGraph
from graph_al.data.base import Dataset
from graph_al.model.base import BaseModel
from graph_al.model.config import ModelConfig
from graph_al.model.prediction import Prediction
import torch
import numpy as np

from copy import copy as copy
from copy import deepcopy as deepcopy
from scipy import stats

def distance(X1, X2, mu):
    Y1, Y2 = mu
    X1_vec, X1_norm_square = X1
    X2_vec, X2_norm_square = X2
    Y1_vec, Y1_norm_square = Y1
    Y2_vec, Y2_norm_square = Y2
    dist = X1_norm_square * X2_norm_square + Y1_norm_square * Y2_norm_square - 2 * (X1_vec @ Y1_vec) * (X2_vec @ Y2_vec)
    # Numerical errors may cause the distance squared to be negative.
    assert np.min(dist) / np.max(dist) > -1e-4
    dist = np.sqrt(np.clip(dist, a_min=0, a_max=None))
    return dist

def init_centers(X1, X2, chosen, chosen_list,  mu, D2):
    if len(chosen) == 0:
        ind = np.argmax(X1[1] * X2[1])
        mu = [((X1[0][ind], X1[1][ind]), (X2[0][ind], X2[1][ind]))]
        D2 = distance(X1, X2, mu[0]).ravel().astype(float)
        D2[ind] = 0
    else:
        newD = distance(X1, X2, mu[-1]).ravel().astype(float)
        D2 = np.minimum(D2, newD)
        D2[chosen_list] = 0
        Ddist = (D2 ** 2) / sum(D2 ** 2)
        customDist = stats.rv_discrete(name='custm', values=(np.arange(len(Ddist)), Ddist))
        ind = customDist.rvs(size=1)[0]
        while ind in chosen: ind = customDist.rvs(size=1)[0]
        mu.append(((X1[0][ind], X1[1][ind]), (X2[0][ind], X2[1][ind])))
    chosen.add(ind)
    chosen_list.append(ind)
    print(str(len(mu)) + '\t' + str(sum(D2)), flush=True)
    return chosen, chosen_list, mu, D2


class AcquisitionStrategyBadge(BaseAcquisitionStrategy):
    """Implementation of the Badge startegy.
    
    https://arxiv.org/abs/1906.03671
    
    Based on the imlementation: https://github.com/JordanAsh/badge/
    """
    
    def __init__(self, config: AcquisitionStrategyBadgeConfig):
        super().__init__(config)
        
    @jaxtyped(typechecker=typechecked)
    def acquire_one(self, mask_acquired: Bool[Tensor, 'num_nodes'], prediction: Prediction | None, model: BaseModel, dataset: Dataset, model_config: ModelConfig, 
            generator: Generator) -> Tuple[int, Dict[str, Tensor | None]]:
        
        
        idxs_unlabeled = torch.where(self.pool(mask_acquired, model, dataset, generator))[0].cpu().numpy()
        
        assert prediction is not None
        assert prediction.embeddings is not None
        embs = prediction.embeddings.mean(0).cpu().numpy()[idxs_unlabeled]
        probs = prediction.get_probabilities(propagated=True).mean(0).cpu().numpy()[idxs_unlabeled]

        # the logic below reflects a speedup proposed by Zhang et al.
        # see Appendix D of https://arxiv.org/abs/2306.09910 for more details
        m = len(idxs_unlabeled)
        mu = None
        D2 = None
        chosen = set()
        chosen_list = []
        emb_norms_square = np.sum(embs ** 2, axis=-1)
        max_inds = np.argmax(probs, axis=-1)

        probs = -1 * probs
        probs[np.arange(m), max_inds] += 1
        prob_norms_square = np.sum(probs ** 2, axis=-1)
        chosen, chosen_list, mu, D2 = init_centers((probs, prob_norms_square), (embs, emb_norms_square), chosen, chosen_list, mu, D2)
        assert len(chosen_list) == 1
        return int(idxs_unlabeled[chosen_list][0]), {}
        
        
        