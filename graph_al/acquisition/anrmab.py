from graph_al.acquisition.config import AcquisitionStrategyANRMABConfig
from graph_al.data.base import BaseDataset, Dataset
from graph_al.model.base import BaseModel
from graph_al.model.prediction import Prediction
from graph_al.acquisition.age import AcquisitionStrategyAGELike
from graph_al.model.config import ModelConfig
from graph_al.utils.logging import get_logger

from jaxtyping import Float, Int, jaxtyped, Bool, Shaped
from typeguard import typechecked
from torch import Tensor
from typing import Dict, List, Tuple
import torch
import numpy as np

class AcquisitionStrategyANRMAB(AcquisitionStrategyAGELike):

    def __init__(self, config: AcquisitionStrategyANRMABConfig, num_nodes: int):
        super().__init__(config)
        self.reset()
        self.min_probability_strategy = config.min_probability_strategy # called p_min in the paper
        self.budget = config.num_to_acquire_per_step * config.num_steps
        self.num_nodes = num_nodes

    def reset(self):
        super().reset()
        self.weights = torch.ones(3).float()
        self.reward_terms = []

        self._cached_query_matrix = None
        self._cached_phi = None
        self._cached_probabilities = None

    @torch.no_grad()
    def acquire_one(self, mask_acquired: Bool[Tensor, 'num_nodes'], prediction: Prediction | None, model: BaseModel, dataset: Dataset, 
        trainer_config: ModelConfig, generator: torch.Generator) -> Tuple[int, Dict[str, Tensor | None]]:

        assert prediction is not None, f'Need a model prediction for ANRMAB acquisition'

        mask_train, mask_train_pool = dataset.data.mask_train.clone().cpu(), dataset.data.mask_train_pool.clone().cpu()
        idxs_train_pool = torch.tensor(range(len(mask_train_pool)))[mask_train_pool]

        if mask_train.sum() == 0:
            # No training instances, sample one index randomly from pool
            self._last_acquisition_was_random = True
            sampled_idx = int((idxs_train_pool[torch.randint(idxs_train_pool.size(0), (1,), generator=generator).item()]).item()) # type: ignore
            representativeness, entropy, centrality = None, None, None
        else:
            # Select the instance from the pool according to age
            self._last_acquisition_was_random = False
            representativeness = self._representativeness(
                mask_train=mask_train, 
                mask_train_pool=mask_train_pool, 
                prediction=prediction
                ).cpu()
            entropy = self._entropy(
                mask_train=mask_train, 
                mask_train_pool=mask_train_pool, 
                prediction=prediction
                ).cpu()
            centrality = self._centrality(
                mask_train=mask_train, 
                mask_train_pool=mask_train_pool, 
                dataset=dataset
                ).cpu()

            query_matrix = torch.stack((representativeness, entropy, centrality)).float() # 3 x num_train_pool, called Q in the paper
            query_matrix /= query_matrix.sum(1, keepdim=True)
            probabilities = self.weights * (1 - 3 * self.min_probability_strategy) / self.weights.sum() + self.min_probability_strategy 
            phi = probabilities @ query_matrix # num_train_pool
            assert torch.allclose(torch.tensor(1.0), phi.sum())

            # We have to use a numpy generator to sample, as torch.distributions.Categorical does not support a generator...
            numpy_rng = np.random.default_rng(generator.seed())
            sampled_unlabeled_idx = numpy_rng.choice(phi.size(0), size=1, p=phi.detach().cpu().numpy())
            sampled_idx = int((idxs_train_pool[sampled_unlabeled_idx]).item())
        
            # Cache values for updating the reward terms once the label has been acquired
            self._cached_phi = phi
            self._cached_probabilities = probabilities
            self._cached_query_matrix = query_matrix
            self._cached_idx_to_unlabled_idx = {idx : unlabeled_idx for unlabeled_idx, idx in enumerate(idxs_train_pool.tolist())}

        return sampled_idx, {'mask_train' : mask_train, 'mask_train_pool' : mask_train_pool,
            'representativeness' : representativeness, 'entropy' : entropy, 'centrality' : centrality, 'weights' : self.weights}

    @typechecked
    def update(self, idxs_acquired: List[int], prediction: Prediction | None, dataset: Dataset, model: BaseModel):

        if self._last_acquisition_was_random:
            return

        get_logger().info(f'Updating with acquired idxs {idxs_acquired}')
        assert self._cached_phi is not None
        assert self._cached_probabilities is not None
        assert self._cached_query_matrix is not None
        assert self._cached_idx_to_unlabled_idx is not None

        phi, probabilities, query_matrix = self._cached_phi, self._cached_probabilities, self._cached_query_matrix

        # Calculate the reward r^t(v_i; f^t, tau)
        for sampled_idx in idxs_acquired:
            assert sampled_idx in self._cached_idx_to_unlabled_idx, f'The sampled node {sampled_idx} was not in the pool'
            sampled_unlabeled_idx = self._cached_idx_to_unlabled_idx[sampled_idx]

            self.reward_terms.append(1 / (phi[sampled_unlabeled_idx] * phi.size(0)))
            reward = 1 / self.budget * sum(self.reward_terms)

            r_hat = reward * query_matrix[:, sampled_unlabeled_idx] / phi[sampled_unlabeled_idx]
            self.weights *= torch.exp(self.min_probability_strategy / 2 * (r_hat + (1 / probabilities) * np.sqrt(np.log(self.num_nodes / (0.1 * 3 * self.budget)))))

        self._cached_phi = None
        self._cached_probabilities = None
        self._cached_query_matrix = None
        self._cached_idx_to_unlabled_idx = None



