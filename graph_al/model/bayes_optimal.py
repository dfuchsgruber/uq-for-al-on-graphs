from graph_al.data.base import Dataset
from graph_al.data.generative import GenerativeDataset
from graph_al.data.config import DatasetSplit
from graph_al.data.sbm.base import SBMDataset
from graph_al.model.base import BaseModel
from graph_al.model.config import BayesOptimalConfig, BayesianPrediction, BayesianLikelihoodConfig
from graph_al.model.prediction import Prediction
from graph_al.utils.logging import get_logger
from graph_al.data.base import Data
from graph_al.model.config import ApproximationType
from graph_al.data.likelihood_cache import ConditionalLogLikelihoodRegistry
from graph_al.utils.utils import ndarray_to_tuple

import numpy as np
import torch
import itertools
from scipy.special import logsumexp
import scipy.sparse as sp
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
import math
from dataclasses import asdict
from copy import deepcopy

from jaxtyping import jaxtyped, Float, Int, Bool, UInt64
from typeguard import typechecked
from typing import Tuple, Any

from graph_al.utils.utils import batched

class BayesOptimal(BaseModel):

    def __init__(self, config: BayesOptimalConfig, graph: Dataset) -> None:
        super().__init__(config, dataset=graph)
        self.approximation_type = config.approximation_type
        if config.approximation_type and config.approximation_type != 'bruteforce':
            assert config.num_samples, "number of samples have to be specified when we use approximation"
        assert isinstance(graph.base, GenerativeDataset), "needs a dataset with generative probabilities"
        
        self.num_samples = config.num_samples
        self.predict_train = config.predict_train
        self.prediction = config.prediction
        self.graph = graph.base
        self.importance_probability = config.importance_prob
        self.beam_size = config.beam_size
        self.beam_fraction_random_assignments = config.beam_fraction_random_assignments
        self.beam_search_restarts_every = config.beam_search_restarts_every
        self.beam_search_num_steps = config.beam_search_num_steps
        self.normalized = config.normalized
        self.confidence_likelihood_config = BayesianLikelihoodConfig(**asdict(config.confidence_likelihood))
        self.prediction_likelihood_config = BayesianLikelihoodConfig(**asdict(config.prediction_likelihood))
        self.compute_approximation_error = config.compute_approximation_error
        self.verbose = config.verbose
        self.total_confidence_fix_samples_per_node = config.total_confidence_fix_samples_per_node
        self.cached = config.cached
        self.num_workers = config.num_workers
        self.multiprocessing = config.multiprocessing
        self.batch_size = config.batch_size
        self.variational_inference_max_num_steps = config.variational_inference_max_num_steps
        self.variational_inference_convergence_tolerance = config.variational_inference_convergence_tolerance
        self.likelihood_cache_in_memory = config.likelihood_cache_in_memory
        
        self.reset_cache()
    
    def reset_parameters(self, generator: torch.Generator):
        self.reset_cache()
        return super().reset_parameters(generator)

    def reset_cache(self):
        self._cache = None
        self._cached_mask_labeled = None
        self._variational_inference_residual = None
    
    @jaxtyped(typechecker=typechecked)
    def get_cached_prediction_if_mask_matches(self, mask_labeled: Bool[np.ndarray, 'num_nodes']) -> Prediction | None:
        if self.cached and self._cached_mask_labeled is not None:
            if (self._cached_mask_labeled == mask_labeled).all():
                get_logger().info('Using cache')
                return self._cache
        else:
            return None

    @jaxtyped(typechecker=typechecked)
    def cache(self, prediction: Prediction, mask_labeled: Bool[np.ndarray, 'num_nodes']):
        self._cache = prediction
        self._cached_mask_labeled = mask_labeled.copy()

    @jaxtyped(typechecker=typechecked)
    def combinations(self, mask_fixed: Bool[np.ndarray, 'num_nodes'], 
                     true_labels: Int[np.ndarray, 'num_nodes']) -> Any:
        """ Samples all combinations of possible class assignments by respecting a certain mask of fixed nodes.
        
        Args:
            mask_fixed (Bool[np.ndarray, 'num_nodes']): A mask of nodes that are labeled already and should not be sampled
            labels (Int[np.ndarray, 'num_nodes']): The labels of nodes that should not be sampled 
            
        Yields:
            Int[np.ndarray, 'num_nodes']: A random assignment of nodes
        """
        a = []
        for idx, elem in enumerate(mask_fixed):
            if elem == 0:
                a += [[j for j in range(self.graph.num_classes)]]
            else:
                a += [[true_labels[idx]]]
        return map(np.array, itertools.product(*a))
    
    @jaxtyped(typechecker=typechecked)
    def uniform_sampling(self, num_samples: int, mask_fixed: Bool[np.ndarray, 'num_nodes'],
                         true_labels: Int[np.ndarray, 'num_nodes'], generator: np.random.Generator) -> Any:
        """ Uniformly samples label assignments for a given set of non labeled nodes.
        
        Args:
            num_samples (int): How many samples to draw
            mask_fixed (Bool[np.ndarray, 'num_nodes']): A mask of nodes that are labeled already and should not be sampled
            labels (Int[np.ndarray, 'num_nodes']): The labels of nodes that should not be sampled
            generator (np.random.Generator): The generator to use for sampling
            
        Yields:
            Int[np.ndarray, 'num_nodes']: For each sample, a random assignment of nodes
        """
        for _ in range(num_samples):
            samples = generator.integers(low=0, high=self.graph.num_classes, size=(self.graph.num_nodes))
            samples[mask_fixed] = true_labels[mask_fixed]
            yield samples
    
    @jaxtyped(typechecker=typechecked)
    def importance_sampling(self, num_samples: int, mask_fixed: Bool[np.ndarray, 'num_nodes'], 
                            true_labels: Int[np.ndarray, 'num_nodes'],
                            generator: np.random.Generator) -> Tuple[
                                Int[np.ndarray, 'num_samples num_nodes'],
                                Float[np.ndarray, 'num_samples']
                            ]:
        nc = self.graph.num_classes
        p = self.importance_probability
        
        change_mask = generator.binomial(n=1, p=p, size=(num_samples, sum(~mask_fixed)))
        change_mask = change_mask.astype('bool')
        changes = generator.integers(low=0, high=nc, size=(change_mask.sum()))
        
        sample = np.array(true_labels).reshape(1, -1).repeat(num_samples, axis=0)
        relevant_values = sample[:, ~mask_fixed]
        relevant_values[change_mask] = changes.flatten()
        sample[:, ~mask_fixed] = relevant_values
        
        same_label = sample == self.graph.labels.numpy()
        p_sample = same_label * ((1 - p) + p / nc - p / nc)
        p_sample += p / nc 
        
        p_sample = np.log(p_sample)[:, ~mask_fixed].sum(1)
        return sample, p_sample
    
    @jaxtyped(typechecker=typechecked)
    def beam_search(self, mask_fixed: Bool[np.ndarray, 'num_nodes'], 
                            true_labels: Int[np.ndarray, 'num_nodes'],
                            x: Float[np.ndarray, 'num_nodes num_features'],
                            generator: np.random.Generator, 
                            likelihood_config: BayesianLikelihoodConfig) -> Tuple[
                                Int[np.ndarray, 'beam_size num_nodes'],
                                Float[np.ndarray, 'beam_size']]:
        """ Uses beam search to find the `beam_size` assignments with the highest density 
        
        Parameters:
        -----------
        mask_fixed (Bool[np.ndarray, 'num_nodes']): Which nodes are fixed.
        true_labels (Int[np.ndarray, 'num_nodes']): The true labels of (fixed) nodes
        x (Float[np.ndarray, 'num_nodes num_features']): Node features
        generator (torch.random.Generator): Generator to generate initial beam
        likelihood_config (BayesianLikelihoodConfig): How the likelihood should be computed

        Returns:
        --------
        Int[np.ndarray, 'beam_size num_nodes']: The top assignments of the beam
        Float[np.ndarray, 'beam_size']]: The conditional log likelihoods of the beam
        """
        idxs_not_fixed = np.where(~mask_fixed)[0]
        num_nodes, num_classes = self.graph.num_nodes, self.graph.num_classes
        num_unlabeled_nodes = len(idxs_not_fixed)

        beam = np.tile(true_labels, (self.beam_size, 1))
        beam[:, ~mask_fixed] = generator.integers(0, num_classes, (self.beam_size, num_unlabeled_nodes))

        for iteration in tqdm(range(self.beam_search_num_steps), disable=True):
            beam = np.tile(beam, (num_unlabeled_nodes * num_classes, 1))
            for idx, c in itertools.product(range(num_unlabeled_nodes), range(num_classes)):
                node_idx = idxs_not_fixed[idx]
                # set node `node_idx`'s label to `c` for all samples in one beam copy
                start_idx = self.beam_size * (idx * num_classes + c)
                beam[start_idx : start_idx + self.beam_size, node_idx] = c
            beam = np.unique(beam, axis=0)
            log_likelihood = self.graph.conditional_log_likelihood(beam, x[None, :], use_adjacency=likelihood_config.use_adjacency,
                                                                            use_features=likelihood_config.use_features,
                                                                            use_non_edges=likelihood_config.use_non_edges)
            # Get the most relevant log_likelihoods
            if self.beam_size < log_likelihood.shape[0]:
                next_beam_idxs = np.argpartition(log_likelihood, -self.beam_size)[-self.beam_size:]
                beam = beam[next_beam_idxs]
                log_likelihood = log_likelihood[next_beam_idxs]
            
        return beam, log_likelihood # type: ignore
            
    @jaxtyped(typechecker=typechecked)
    def _new_beam(self, beam_size: int, 
                            mask_fixed: Bool[np.ndarray, 'num_nodes'], 
                            true_labels: Int[np.ndarray, 'num_nodes'],
                            x: Float[np.ndarray, 'num_nodes num_features'], 
                            generator: np.random.Generator,
                            likelihood_config: BayesianLikelihoodConfig) -> Tuple[
                                Int[np.ndarray, 'beam_size num_nodes'], Float[np.ndarray, 'beam_size']
                            ]:
        beam = np.tile(true_labels, (beam_size, 1))
        beam[:, ~mask_fixed] = generator.integers(0, self.graph.num_classes, (beam_size, (~mask_fixed).sum()))
        marginal_log_likelihoods = self.graph.conditional_log_likelihood(beam, x[None, ...], 
                use_adjacency=likelihood_config.use_adjacency, 
                use_features=likelihood_config.use_features, use_non_edges=likelihood_config.use_non_edges)
        return beam, marginal_log_likelihoods

    @jaxtyped(typechecker=typechecked)
    def memory_efficient_cached_beam_search(self, mask_fixed: Bool[np.ndarray, 'num_nodes'], 
                            true_labels: Int[np.ndarray, 'num_nodes'],
                            x: Float[np.ndarray, 'num_nodes num_features'],
                            generator: np.random.Generator,
                            likelihood_config: BayesianLikelihoodConfig) -> Tuple[
                                Int[np.ndarray, 'beam_size num_nodes'],
                                Float[np.ndarray, 'beam_size']]:
        """ Uses beam search to find the `beam_size` assignments with the highest density. In each iteration, the
        density of the beam is averaged and stored. The next beam will never include samples of the previous beam.
        
        Parameters:
        -----------
        mask_fixed (Bool[np.ndarray, 'num_nodes']): Which nodes are fixed.
        true_labels (Int[np.ndarray, 'num_nodes']): The true labels of (fixed) nodes
        x (Float[np.ndarray, 'num_nodes num_features']): Node features
        generator (torch.random.Generator): Generator to generate initial beam

        Returns:
        --------
        Int[np.ndarray, 'beam_size num_nodes']: The top assignments of the beam
        Float[np.ndarray, 'beam_size']]: The conditional log likelihoods of the beam
        """
        idxs_not_fixed = np.where(~mask_fixed)[0]
        num_nodes, num_classes = self.graph.num_nodes, self.graph.num_classes
        num_unlabeled_nodes = len(idxs_not_fixed)

        # We restrict the beam to assignments for which we do not have computed marginal log likelihoods
        # Also, we cache the marginal log likelihoods for each assignment we have seen
        cache = {}
        cached_hashes = set()
        random_hashing_projection = generator.integers(0, 2**64, size=(num_nodes,), dtype=np.uint64) # this makes collisions of the hashes very unlikely
        
        for iteration in tqdm(range(self.beam_search_num_steps), disable=True):
            if iteration == 0 or (self.beam_search_restarts_every > 0 and (iteration % self.beam_search_restarts_every) == 0):
                beam, marginal_log_likelihoods = self._new_beam(self.beam_size, mask_fixed, true_labels, x, generator, likelihood_config)

            marginal_log_likelihoods = self.graph.conditional_log_likelihood_delta_with_one_label_changed(
                beam, x[None, ...], ~mask_fixed, labels_log_likelihood=marginal_log_likelihoods,  # type: ignore
                use_adjacency=likelihood_config.use_adjacency, 
                use_features=likelihood_config.use_features, use_non_edges=likelihood_config.use_non_edges,
            )
            beam = np.tile(beam, (num_unlabeled_nodes, num_classes, 1, 1)).transpose(2, 0, 1, 3) # beam_size, num_unlabeled, num_classes, num_nodes
            for unlabeled_idx, c in itertools.product(range(num_unlabeled_nodes), range(self.graph.num_classes)):
                beam[:, unlabeled_idx, c, idxs_not_fixed[unlabeled_idx]] = c

            marginal_log_likelihoods, beam = marginal_log_likelihoods.reshape(-1), beam.reshape(-1, beam.shape[-1])

            # Do not re-use values that are already cached
            beam_hashes = beam @ random_hashing_projection
            mask_not_in_cache = np.array([h not in cached_hashes for h in beam_hashes])
            marginal_log_likelihoods, beam, beam_hashes = marginal_log_likelihoods[mask_not_in_cache], beam[mask_not_in_cache], beam_hashes[mask_not_in_cache]

            # Only keep the `beam_size` most relevant assignments that were not considered before
            if beam.shape[0] > 0 and self.beam_size < (beam.shape[0] * (1 - self.beam_fraction_random_assignments)):
                next_beam_idxs = np.argpartition(marginal_log_likelihoods, -self.beam_size)[-self.beam_size:]
                beam = beam[next_beam_idxs]
                marginal_log_likelihoods = marginal_log_likelihoods[next_beam_idxs]

            if beam.shape[0] < self.beam_size: # Use random assignments to fill the beam
                new_beam, new_marginal_log_likelihoods = self._new_beam(self.beam_size - beam.shape[0], mask_fixed, true_labels, x, generator, likelihood_config)
                beam, marginal_log_likelihoods, beam_hashes = np.concatenate((beam, new_beam), axis=0), np.concatenate((marginal_log_likelihoods, new_marginal_log_likelihoods)), \
                    np.concatenate((beam_hashes, new_beam @ beam_hashes))

            # For memory efficiency, only cache up to `beam_size` assignments
            cache |= {tuple(assignment) : mll for assignment, mll in zip(beam, marginal_log_likelihoods)}
            cached_hashes |= set(beam_hashes)
            
        assignments, marginal_log_likelihoods = zip(*cache.items())
        return np.array(assignments), np.array(marginal_log_likelihoods)
    
    @jaxtyped(typechecker=typechecked)
    def get_likelihood_cache(self, x: Float[np.ndarray, 'num_nodes feature_dim'],
                         y: Int[np.ndarray, 'num_nodes'], likelihood_config: BayesianLikelihoodConfig) -> Tuple[Int[np.ndarray, 'num_assignments num_nodes'], Float[np.ndarray, 'num_assignments']]:
        if self.likelihood_cache_in_memory:
            cache = self.graph._in_memory_likelihood_cache
        else:
            cache = ConditionalLogLikelihoodRegistry(self.graph.likelihood_cache_database_path, self.graph.likelihood_cache_storage_path, # type: ignore
                self.graph.likelihood_cache_database_lockfile_path) # type: ignore
        key = (self.graph.key, (likelihood_config.use_adjacency, likelihood_config.use_non_edges, likelihood_config.use_features), ndarray_to_tuple(x.round(5)))
        get_logger().info('Using total confidence from log likelihood cache.')
        if key in cache:
            assignments, log_likelihoods = cache[key]
        else:
            get_logger().info(f'Computing log likelihoods for cache for key {key}...')
            num_assignments = int(self.graph.num_classes**self.graph.num_nodes)
            if self.multiprocessing:
                # distribute all assignments to the number of workers
                num_assignments_per_workers = max(1, num_assignments // (self.num_workers or cpu_count()))
                iterator = [(start, min(num_assignments_per_workers, num_assignments - start)) 
                            for start in range(0, num_assignments, num_assignments_per_workers)]
                job = partial(compute_all_conditional_log_likelihoods_job, y=y.copy(), x=x.copy(), classifier=deepcopy(self), likelihood_config=deepcopy(likelihood_config))
                with Pool(processes=self.num_workers) as pool:
                    results = pool.starmap(job, iterator)
                assignments, log_likelihoods = zip(*results)
                assignments, log_likelihoods = np.concatenate(assignments, axis=0), np.concatenate(log_likelihoods, axis=0)
            else:
                assignments, log_likelihoods = compute_all_conditional_log_likelihoods_job(0, num_assignments, y=y.copy(), x=x.copy(), 
                                                                                classifier=deepcopy(self), progress_bar=self.verbose,
                                                                                 likelihood_config=deepcopy(likelihood_config))
            cache[key] = (assignments, log_likelihoods)
            get_logger().info(f'Cached {assignments.shape[0]} assignments')
        return assignments, log_likelihoods

    @jaxtyped(typechecker=typechecked)
    def marginal_class_labels_from_cached_likelihood(self, mask_fixed: Bool[np.ndarray, 'num_nodes'],
                         x: Float[np.ndarray, 'num_nodes feature_dim'],
                         y: Int[np.ndarray, 'num_nodes'],
                         likelihood_config: BayesianLikelihoodConfig) -> Float[np.ndarray, 'num_nodes num_classes']:
        assignments, log_likelihoods = self.get_likelihood_cache(x, y, likelihood_config)
        # Filter all assignments 
        mask = (assignments[:, mask_fixed] == y[mask_fixed]).all(1)
        assignments, log_likelihoods = assignments[mask], log_likelihoods[mask]

        result = np.zeros((self.graph.num_nodes, self.graph.num_classes), dtype=float)
        result[np.arange(self.graph.num_nodes), y] = 1.0 # Total confidence is 1.0 for all labeled nodes
        for i, c in itertools.product(np.where(~mask_fixed)[0], range(self.graph.num_classes)):
            mask_i_c = assignments[:, i] == c
            result[i, c] = logsumexp(log_likelihoods[mask_i_c]) - np.log(mask_i_c.sum())
        return result
    
    @jaxtyped(typechecker=typechecked)
    def marginal_class_labels_variational_inference(self, mask_fixed: Bool[np.ndarray, 'num_nodes'],
                         x: Float[np.ndarray, 'num_nodes feature_dim'],
                         y: Int[np.ndarray, 'num_nodes'],
                         likelihood_config: BayesianLikelihoodConfig,
                         initial_confidence: Float[np.ndarray, 'num_nodes num_classes'] | None = None,
                         return_history: bool = False) -> Float[np.ndarray, 'num_steps num_nodes num_classes']:
        
        assert isinstance(self.graph, SBMDataset), f'VI estimation is currently only implemented for SBMs'
        
        edge_idxs = self.graph.edge_idxs.numpy()
        A = sp.coo_matrix((np.ones(edge_idxs.shape[1]), edge_idxs), shape=(self.graph.num_nodes, self.graph.num_nodes)).tocsr()
        A = np.array(A.todense()) # Materialize the non-edges as well, since we have to sum over them as well
        
        # log_probability_features[u, c] = p(X_u | y_u=c)
        log_probability_features = self.graph.feature_log_likelihood(x)
        # log_probability_adjacency[u, c1, v, c2] = p(A_uv | y_u=c1, y_v=c2)
        log_probability_adjacency = (A > 0)[:, None, :, None] * np.log(self.graph.affiliation_matrix[None, :, None, :])
        if likelihood_config.use_non_edges:
           log_probability_adjacency += (np.isclose(A, 0))[:, None, :, None] * np.log(1 - self.graph.affiliation_matrix[None, :, None, :])
        log_probability_adjacency[np.arange(self.graph.num_nodes), :, np.arange(self.graph.num_nodes), :] = 0 # We do not model self-loops
        
        if initial_confidence is None:
            confidence = np.random.uniform(0, 1, size=(self.graph.num_nodes, self.graph.num_classes)) #np.ones((self.graph.num_nodes, self.graph.num_classes))
            confidence /= confidence.sum(1, keepdims=True)
        else:
            confidence = initial_confidence.copy()
        confidence[mask_fixed] = 0.0
        confidence[np.where(mask_fixed), y[mask_fixed]] = 1.0
            
        history = [confidence]
        
        log_prior = np.log(self.graph.class_prior.numpy())
        iterator = tqdm(range(self.variational_inference_max_num_steps), desc='EM step', disable = not self.verbose)
        for iteration in iterator:
            # The confidence (probability distribution) for fixed nodes will also be fixed

            # Update the confidence for other nodes
            log_confidence_new = np.zeros_like(confidence)
            log_confidence_new += log_prior[None, :]
            if likelihood_config.use_adjacency:
                log_confidence_new += (log_probability_adjacency * confidence[None, None, ...]).sum(-1).sum(-1)
            if likelihood_config.use_features:
                log_confidence_new += log_probability_features
            confidence_new = np.exp(log_confidence_new - logsumexp(log_confidence_new, axis=1, keepdims=True))
            confidence_new[mask_fixed] = 0.0
            confidence_new[np.where(mask_fixed), y[mask_fixed]] = 1.0
            residual = np.abs(confidence - confidence_new)[~mask_fixed]
            confidence = confidence_new
            
            if return_history:
                history.append(confidence)
            else:
                history = [confidence]
            
            if (residual < self.variational_inference_convergence_tolerance).all():
                get_logger().info(f'Converged after {iteration + 1} iteration(s).')
                break
            else:
                iterator.set_description(f'Average residual {residual.mean():.5f}, Max residual {residual.max():.5f}')  
        return np.stack(history)    
    
    @jaxtyped(typechecker=typechecked)
    def total_confidence(self, mask_fixed: Bool[np.ndarray, 'num_nodes'],
                         x: Float[np.ndarray, 'num_nodes feature_dim'],
                         y: Int[np.ndarray, 'num_nodes'],
                         likelihood_config: BayesianLikelihoodConfig, 
                         unnormalized: bool =False,) -> Float[np.ndarray, 'num_nodes num_classes']:
        """ Computes the total confidence as a marginalization of aleatoric confidences over all non-fixed nodes.
        I.e. conf_total(i, c) = p(y_i = c| X, A, Y_{mask_fixed}) is proportional to
            p(A, X| y_i=c, Y_{mask_fixed}) = E_p(Y_not_fixed) [ p (A, X | y_i = c, Y_{mask_fixed}, Y_{not_fixed}) ] p(y_i = c)
        
        
        Args:
            x (Float[np.ndarray, 'num_nodes feature_dim']): Node features
            y (Int[np.ndarray, 'num_nodes']): Ground truth node labels
            likelihood_config (BayesianLikelihoodConfig): configuration for computing the likelihood
            
        Returns:
            Float[np.ndarray, 'num_classes num_nodes']: Indices [c, i] provide the total confidence of node i
                                                        at class c: p(y_i = c | X, A, Y_{mask_fixed}=y_{mask_fixed})
        """
        progress_bar = self.verbose
        iterator = list(itertools.product(range(self.graph.num_nodes), range(self.graph.num_classes)))
        if progress_bar:
            print()
            print()
            iterator = tqdm(iterator, f'Approximating total confidence')
        if self.approximation_type == ApproximationType.LIKELIHOOD_CACHE:
            log_likelihood = self.marginal_class_labels_from_cached_likelihood(mask_fixed.copy(), x.copy(), y.copy(), deepcopy(likelihood_config))
        elif self.approximation_type == ApproximationType.VARIATIONAL_INFERENCE:
            history = self.marginal_class_labels_variational_inference(mask_fixed.copy(), x.copy(), y.copy(), deepcopy(likelihood_config), return_history=True)
            self._variational_inference_residual = history[-2] - history[-1]
            return history[-1]
        else:
            log_likelihood = np.full((self.graph.num_nodes, self.graph.num_classes), 0.0, dtype=float)
            seeds = np.random.default_rng().integers(0, 0xFFFFFFFF, size=(self.graph.num_nodes, self.graph.num_classes))
            if self.total_confidence_fix_samples_per_node:
                seeds[:, :] = seeds[:, [0]] # Per node (row) use the same seed (-> same sampled assignments)
            
            if self.multiprocessing:
                job = partial(approximate_log_likelihood_job, y=y.copy(), mask_fixed=mask_fixed.copy(), seeds=seeds.copy(), x=x.copy(), classifier=deepcopy(self),
                    likelihood_config=deepcopy(likelihood_config))
                with Pool(processes=self.num_workers) as pool:
                    results = pool.starmap(job, iterator)
                    for i, c, log_likelihood_ic in results:
                        log_likelihood[i, c] = log_likelihood_ic
            else:
                for i, c in iterator: 
                    j, k, log_likelihood_jk = approximate_log_likelihood_job(i, c, classifier=self, mask_fixed=mask_fixed.copy(),
                                                                    x=x.copy(), y=y.copy(), seeds=seeds.copy(), likelihood_config=deepcopy(likelihood_config))
                    log_likelihood[j, k] = log_likelihood_jk
        
        if progress_bar:
            print()    
        
        log_posterior = log_likelihood # TODO: use a class prior
        # normalize the posterior
        if not unnormalized:
            log_posterior -= logsumexp(log_posterior, axis=1, keepdims=True)
        return np.exp(log_posterior)

    @jaxtyped(typechecker=typechecked)
    def aleatoric_confidence(self,
                             x: Float[np.ndarray, 'num_nodes feature_dim'],
                            y: Int[np.ndarray, 'num_nodes'],
                            likelihood_config: BayesianLikelihoodConfig) -> Float[np.ndarray, 'num_nodes num_classes']:
        """ Computes aleatoric confidence for all nodes, i.e. p(Y_i = c | Y_{-i}, X, A)
        
        Args:
            x (Float[np.ndarray, 'num_nodes feature_dim']): Node features
            y (Int[np.ndarray, 'num_nodes']): Ground truth node labels
            likelihood_config (BayesianLikelihoodConfig): configuration for likelihood computation
            
        Returns:
            Float[np.ndarray, 'num_classes num_nodes']: Indices [c, i] provide the aleatoric confidence of node i
                                                        at class c: p(y_i = c | X, A, Y_{-i})
        """
        # Do it inefficiently for now, TODO
        log_likelihood = np.zeros((self.graph.num_nodes, self.graph.num_classes))
        for i, c in itertools.product(range(self.graph.num_nodes), range(self.graph.num_classes)):
            y_ic = y.copy()
            y_ic[i] = c
            log_likelihood[i, c] = self.graph.conditional_log_likelihood(y_ic[None, :], x[None, :],
                                                                         use_adjacency=likelihood_config.use_adjacency,
                                                                         use_features=likelihood_config.use_features,
                                                                         use_non_edges=likelihood_config.use_non_edges,
                                                                         )
        log_likelihood: Float[np.ndarray, 'num_nodes num_classes'] = np.array(log_likelihood)
        log_posterior = log_likelihood # TODO: Use a prior
        log_posterior -= logsumexp(log_posterior, axis=1, keepdims=True)
        posterior = np.exp(log_posterior)
        assert np.allclose(posterior.sum(1), 1.0), posterior
        return posterior
    
    @jaxtyped(typechecker=typechecked)
    def maximum_likelihood_assignment(self, mask_fixed: Bool[np.ndarray, 'num_nodes'],
                         x: Float[np.ndarray, 'num_nodes feature_dim'],
                         y: Int[np.ndarray, 'num_nodes'],
                         likelihood_config: BayesianLikelihoodConfig) -> Int[np.ndarray, 'num_nodes']:
        """ Computes the maxmimum likelihood estimate for p(y_u | y_o, X, A) jointly.
        
        Args:
            mask_fixed (Bool[np.ndarray, 'num_nodes']): observed nodes
            x (Float[np.ndarray, 'num_nodes feature_dim']): node features
            y (Int[np.ndarray, 'num_nodes']): node labels (only relevant for observed nodes)
            likelihood_config (BayesianLikelihoodConfig): configuration for computing the likelihood
        Returns:
            Int[np.ndarray, 'num_nodes']: the assignment that maximizes the likelihood
        """
        num_assignments = int(self.graph.num_classes**(~mask_fixed).sum())
        if self.verbose:
            get_logger().info(f'Joint assignment: Testing {num_assignments} assignments')
        if self.multiprocessing:
            # distribute all assignments to the number of workers
            num_assignments_per_workers = max(1, num_assignments // (self.num_workers or cpu_count()))
            iterator = [(start, min(num_assignments_per_workers, num_assignments - start)) 
                        for start in range(0, num_assignments, num_assignments_per_workers)]
            job = partial(compute_argmax_joint_likelihood_job, y=y.copy(), mask_fixed=mask_fixed.copy(), x=x.copy(), classifier=deepcopy(self))
            with Pool(processes=self.num_workers) as pool:
                results = pool.starmap(job, iterator)
            log_likelihoods, assignments = zip(*results)
            assignment = assignments[np.array(log_likelihoods).argmax()]
        else:
            log_likelihood, assignment = compute_argmax_joint_likelihood_job(0, num_assignments, y=y.copy(), mask_fixed=mask_fixed.copy(), x=x.copy(), 
                                                                             classifier=deepcopy(self), likelihood_config=deepcopy(likelihood_config),
                                                                             progress_bar=self.verbose)
        return assignment # type: ignore    

    @jaxtyped(typechecker=typechecked)
    def maximum_likelihood_assignment_from_likelihood_cache(self, mask_fixed: Bool[np.ndarray, 'num_nodes'],
                         x: Float[np.ndarray, 'num_nodes feature_dim'],
                         y: Int[np.ndarray, 'num_nodes'],
                         likelihood_config: BayesianLikelihoodConfig) -> Int[np.ndarray, 'num_nodes']:
        """ Computes the maxmimum likelihood estimate for p(y_u | y_o, X, A) jointly by using the pre-computed likelihood cache.
        
        Args:
            mask_fixed (Bool[np.ndarray, 'num_nodes']): observed nodes
            x (Float[np.ndarray, 'num_nodes feature_dim']): node features
            y (Int[np.ndarray, 'num_nodes']): node labels (only relevant for observed nodes)
            likelihood_config (BayesianLikelihoodConfig): configuration for computing the likelihood
        Returns:
            Int[np.ndarray, 'num_nodes']: the assignment that maximizes the likelihood
        """
        assignments, log_likelihoods = self.get_likelihood_cache(x, y, likelihood_config)
        mask = (assignments[:, mask_fixed] == y[mask_fixed]).all(1)
        assignments, log_likelihoods = assignments[mask], log_likelihoods[mask]
        assignment = assignments[np.array(log_likelihoods).argmax()]
        return assignment

    @typechecked
    def predict(self, batch: Data, acquisition: bool=False, which: DatasetSplit | None=None) -> Prediction:
        if which: 
            mask_predict_nodes = batch.get_mask(which).numpy()
        else:
            # predict all but already labeled training nodes
            mask_predict_nodes = np.ones(self.graph.num_nodes, dtype=bool)
            mask_predict_nodes &= batch.get_mask(DatasetSplit.TRAIN_POOL).numpy()
        # Should we do something with this mask?
        # As long as we are doing it hackily and we use ONE assignment for all nodes within a sample, it does not really matter
        # Since we are computing likelihoods of shape [num_classes, num_samples, num_nodes] anyway
        # What is more sound: Predict likelihoods for each graph (which consists of num_nodes nodes) and only for num_nodes_to_predict nodes
        # i.e. [num_classes, num_samples, num_nodes_to_predict, num_nodes]
        # then the per-sample likelihoods (last axis) can be summed and we get independent MC approximations for each node to predict

        x = batch.x.numpy()
        y = batch.y.numpy()
        mask_labeled_nodes = batch.get_mask(DatasetSplit.TRAIN).numpy()
        
        prediction = self.get_cached_prediction_if_mask_matches(mask_labeled_nodes)
        if prediction is not None:
            return prediction
        
        if self.verbose:
            print(f'With approximation type', self.approximation_type)
        total_confidence = self.total_confidence(mask_labeled_nodes, x, y, self.confidence_likelihood_config)
        aleatoric_confidence = self.aleatoric_confidence(x, y, self.confidence_likelihood_config)
        
        match self.prediction:
            case BayesianPrediction.MARGINAL:
                if self.confidence_likelihood_config == self.prediction_likelihood_config:
                    probabilities = torch.from_numpy(total_confidence)
                else:
                    probabilities = torch.from_numpy(self.total_confidence(mask_labeled_nodes, x, y, self.prediction_likelihood_config))
            case BayesianPrediction.JOINT:
                predicted_assignment = self.maximum_likelihood_assignment(mask_labeled_nodes, x, y, self.prediction_likelihood_config)
                probabilities = np.zeros((self.graph.num_nodes, self.graph.num_classes), dtype=float)
                probabilities[np.arange(probabilities.shape[0]), predicted_assignment] = 1.0
                probabilities = torch.from_numpy(probabilities)
            case BayesianPrediction.JOINT_BEAM_SEARCH | BayesianPrediction.JOINT_CACHED_BEAM_SEARCH:
                match self.prediction:
                    case BayesianPrediction.JOINT_BEAM_SEARCH:
                        maximum_likelihood_assignments, log_likelihoods = self.beam_search(mask_labeled_nodes, y, x, np.random.default_rng(), 
                            self.prediction_likelihood_config)
                    case BayesianPrediction.JOINT_CACHED_BEAM_SEARCH:
                        maximum_likelihood_assignments, log_likelihoods = self.memory_efficient_cached_beam_search(mask_labeled_nodes, y, x, 
                            np.random.default_rng(), self.prediction_likelihood_config)
                predicted_assignment = maximum_likelihood_assignments[log_likelihoods.argmax()]
                # TODO: maybe sample from these assignments to get smoother probabilities (?)
                probabilities = np.zeros((self.graph.num_nodes, self.graph.num_classes), dtype=float)
                probabilities[np.arange(probabilities.shape[0]), predicted_assignment] = 1.0
                probabilities = torch.from_numpy(probabilities)
            case BayesianPrediction.JOINT_CACHED_LIKELIHOOD:
                predicted_assignment = self.maximum_likelihood_assignment_from_likelihood_cache(mask_labeled_nodes, x, y, 
                    self.prediction_likelihood_config)
                probabilities = np.zeros((self.graph.num_nodes, self.graph.num_classes), dtype=float)
                probabilities[np.arange(probabilities.shape[0]), predicted_assignment] = 1.0
                probabilities = torch.from_numpy(probabilities)            
            case _:
                raise ValueError(f'Invalid prediction type {self.prediction}')
        
        if self.compute_approximation_error:
            tmp = self.approximation_type
            self.approximation_type = ApproximationType.LIKELIHOOD_CACHE
            total_confidence_exact = self.total_confidence(mask_labeled_nodes, x, y, self.confidence_likelihood_config)
            self.approximation_type = tmp
            approximation_error = total_confidence_exact - total_confidence
        else:
            approximation_error = None
        
        # debugging
        if self.verbose:
            print(f'Labeled nodes: {torch.where(batch.mask_train)[0].tolist()}')
            print(total_confidence.round(3))
            if self.compute_approximation_error:
                print(f'Exact total confidence')
                print(total_confidence_exact.round(3))
                print('Approximation Error')
                print(approximation_error.round(3))
            print(f'probabilities')
            print(probabilities.numpy().round(2))

        if self.predict_train: # Predict 1.0 for the labeled nodes at the labeled class
            idxs_labeled = list(np.where(mask_labeled_nodes)[0])
            probabilities[idxs_labeled] = 0.0
            probabilities[idxs_labeled, batch.y[idxs_labeled]] = 1.0

        prediction = Prediction(
            probabilities=probabilities[None, :, :], 
            aleatoric_confidence=torch.from_numpy(aleatoric_confidence), 
            total_confidence=torch.from_numpy(total_confidence),
            approximation_error=torch.from_numpy(approximation_error) if approximation_error is not None else None,
            confidence_residual=torch.from_numpy(self._variational_inference_residual) if self._variational_inference_residual is not None else None,
            )
        if self.cached:
            self.cache(prediction, mask_labeled_nodes)
        return prediction
    
@jaxtyped(typechecker=typechecked)
def compute_all_conditional_log_likelihoods_job(start: int, num: int, classifier: BayesOptimal,
                                x: Float[np.ndarray, 'num_nodes feature_dim'],
                                y: Int[np.ndarray, 'num_nodes'],
                                likelihood_config: BayesianLikelihoodConfig, 
                                progress_bar: bool=False) -> Tuple[Int[np.ndarray, 'num_assignments num_nodes'], Float[np.ndarray, 'num_assignments']]:
    """ Computes maximum over joint likelihoods for different assignments y to the non-fixed nodes.      

    Args:
        start (int): at which assignment to start
        num (int): how many assignments to try
        classifier (BayesOptimal): the classifier
        x (Float[np.ndarray, &#39;num_nodes feature_dim&#39;]): features
        y (Int[np.ndarray, &#39;num_nodes&#39;]): labels (only relevant for where `mask_fixed` is true)
        likelihood_config (BayesianLikelihoodConfig): configuration for likelihood computation
        progress_bar (bool): Whether to print a progress bar
        
    Returns:
        log_likelihood_best (float): the maximal log likelihood computed in this job
        y_best (Int[np.ndarray, &#39;num_nodes&#39;]): assignment that corresponds to this log likelihood
    """
    num_nodes, num_classes = classifier.graph.num_nodes, classifier.graph.num_classes
    
    iterator = range(start, start + num, classifier.batch_size)
    if progress_bar:
        print()
        print()
        iterator = tqdm(iterator, desc=f'Computing log likelihoods of all assignments')
    
    all_assignments, all_log_likelihoods = [], []

    for start_idx in iterator:
        end_idx = min(start_idx + classifier.batch_size, start + num)
        
        # generate assignments by transforming indices into base `num_classes`
        assignments = np.repeat(y[None, :], end_idx - start_idx, axis=0)
        assignment = np.arange(start_idx, end_idx, 1) # "running" assignment in base 
        for node_idx in range(num_nodes):
            assignments[:, node_idx] = assignment % num_classes
            assignment //= num_classes
        
        log_likelihoods_batch = classifier.graph.conditional_log_likelihood(assignments, x[None, :], use_adjacency=likelihood_config.use_adjacency,
                                                                            use_features=likelihood_config.use_features,
                                                                            use_non_edges=likelihood_config.use_non_edges)
        all_assignments.append(assignments)
        all_log_likelihoods.append(log_likelihoods_batch)

    if len(all_assignments) > 0:
        return np.concatenate(all_assignments, axis=0), np.concatenate(all_log_likelihoods, axis=0) # type: ignore
    else:
        return np.empty((0, num_nodes), dtype=int), np.empty((0,), dtype=float)

@jaxtyped(typechecker=typechecked)
def compute_argmax_joint_likelihood_job(start: int, num: int, classifier: BayesOptimal, mask_fixed: Bool[np.ndarray, 'num_nodes'],
                                x: Float[np.ndarray, 'num_nodes feature_dim'],
                                y: Int[np.ndarray, 'num_nodes'], 
                                likelihood_config: BayesianLikelihoodConfig,
                                progress_bar: bool=False) -> Tuple[float, Int[np.ndarray, 'num_nodes']]:
    """ Computes maximum over joint likelihoods for different assignments y to the non-fixed nodes.      

    Args:
        start (int): at which assignment to start
        num (int): how many assignments to try
        classifier (BayesOptimal): the classifier
        mask_fixed (Bool[np.ndarray, &#39;num_nodes&#39;]): which nodes in y are fixed
        x (Float[np.ndarray, &#39;num_nodes feature_dim&#39;]): features
        y (Int[np.ndarray, &#39;num_nodes&#39;]): labels (only relevant for where `mask_fixed` is true)
        likelihood_config (BayesianLikelihoodConfig): configuration for the likelihood computation
        progress_bar (bool): Whether to print a progress bar
        
    Returns:
        log_likelihood_best (float): the maximal log likelihood computed in this job
        y_best (Int[np.ndarray, &#39;num_nodes&#39;]): assignment that corresponds to this log likelihood
    """
    num_nodes, num_classes = classifier.graph.num_nodes, classifier.graph.num_classes
    
    max_log_likelihood, best_assignment = -np.inf, None
    iterator = range(start, start + num, classifier.batch_size)
    if progress_bar:
        print()
        print()
        iterator = tqdm(iterator, desc=f'Sequentially computing argmax over joint likelihoods')
    
    for start_idx in iterator:
        end_idx = min(start_idx + classifier.batch_size, start + num)
        
        # generate assignments by transforming indices into base `num_classes`
        assignments = np.repeat(y[None, :], end_idx - start_idx, axis=0)
        assignment = np.arange(start_idx, end_idx, 1) # "running" assignment in base 
        for node_idx in np.where(~mask_fixed)[0]:
            assignments[:, node_idx] = assignment % num_classes
            assignment //= num_classes
        
        log_likelihoods_batch = classifier.graph.conditional_log_likelihood(assignments, x[None, :], use_adjacency=likelihood_config.use_adjacency,
                                                                            use_features=likelihood_config.use_features,
                                                                            use_non_edges=likelihood_config.use_non_edges)
        max_log_likelihood_batch_idx = log_likelihoods_batch.argmax()
        max_log_likelihood_batch = log_likelihoods_batch[max_log_likelihood_batch_idx]
        if max_log_likelihood_batch > max_log_likelihood:
            max_log_likelihood, best_assignment = max_log_likelihood_batch, assignments[max_log_likelihood_batch_idx]
    
    return max_log_likelihood, best_assignment # type: ignore
     
@jaxtyped(typechecker=typechecked)
def approximate_log_likelihood_job(i: int, c: int, 
                               classifier: BayesOptimal, 
                                mask_fixed: Bool[np.ndarray, 'num_nodes'],
                                x: Float[np.ndarray, 'num_nodes feature_dim'],
                                y: Int[np.ndarray, 'num_nodes'],
                                seeds: Int[np.ndarray, 'num_nodes num_classes'],
                                likelihood_config: BayesianLikelihoodConfig) -> Tuple[int, int, float]:
    """ Approximates the log likelihood log(p(A, X | Y_{mask_fixed})) """
    # Compute log(p(y_i = c | A, X, Y_{mask_fixed})) separately for each node i and class c
    # Fix node i to class c
    y_ic, mask_fixed_ic = y.copy(), mask_fixed.copy()
    y_ic[i] = c
    mask_fixed_ic[i] = True
    seed = int(seeds[i, c])
    
    match classifier.approximation_type:
        case ApproximationType.BRUTE_FORCE | None:
            log_likelihoods = []
            for samples in batched(classifier.batch_size, classifier.combinations(mask_fixed_ic, y_ic)):
                log_likelihoods.append(classifier.graph.conditional_log_likelihood(np.stack(samples), x[None, :], use_adjacency=likelihood_config.use_adjacency,
                                                                            use_features=likelihood_config.use_features,
                                                                            use_non_edges=likelihood_config.use_non_edges))
            log_likelihoods = np.concatenate(log_likelihoods, axis=0)
            return i, c, logsumexp(log_likelihoods) - np.log(log_likelihoods.shape[0])
        case ApproximationType.MONTE_CARLO:
            log_likelihoods = []
            for samples in batched(classifier.batch_size, 
                                   classifier.uniform_sampling(classifier.num_samples, mask_fixed_ic, y_ic, generator=np.random.default_rng(seed=seed))):
                log_likelihoods.append(classifier.graph.conditional_log_likelihood(np.stack(samples), x[None, :], use_adjacency=likelihood_config.use_adjacency,
                                                                            use_features=likelihood_config.use_features,
                                                                            use_non_edges=likelihood_config.use_non_edges))
            log_likelihoods = np.concatenate(log_likelihoods, axis=0)
            return i, c, logsumexp(log_likelihoods) - np.log(log_likelihoods.shape[0])
        case ApproximationType.IMPORTANCE:
            # TODO: batched fashion of importance sampling: `classifier.importance_sampling` needs to yield instead of returning
            samples, log_q_samples = classifier.importance_sampling(classifier.num_samples, mask_fixed_ic, y_ic, generator=np.random.default_rng(seed=seed)) # samping from q(y)
            log_likelihoods = classifier.graph.conditional_log_likelihood(samples, x[None, :], use_adjacency=likelihood_config.use_adjacency,
                                                                            use_features=likelihood_config.use_features,
                                                                            use_non_edges=likelihood_config.use_non_edges)
            # p(y) / q(y)
            norm_term: Float[np.ndarray, 'num_samples'] = (~mask_fixed_ic).sum() * np.log(1 / classifier.graph.num_classes) - log_q_samples # TODO: use a class prior
            log_likelihoods += norm_term
            return i, c, logsumexp(log_likelihoods) - np.log(log_likelihoods.shape[0])
        case ApproximationType.BEAM_SEARCH:
            # We find the most relevant assignments in terms of conditional log likelihood using beam search
            # Then we "sample" from them uniformly and treat it as importance sampling: Effectively we just average them and hope for the best
            assignments, log_likelihoods = classifier.beam_search(mask_fixed_ic, y_ic, x, generator=np.random.default_rng(seed=seed), likelihood_config=likelihood_config)
            return i, c, logsumexp(log_likelihoods) - np.log(assignments.shape[0])
        case ApproximationType.CACHED_BEAM_SEARCH:
            # We find the most relevant assignments in terms of conditional log likelihood using beam search
            # Then we "sample" from them uniformly and treat it as importance sampling: Effectively we just average them and hope for the best
            assignments, log_likelihoods = classifier.memory_efficient_cached_beam_search(mask_fixed_ic, y_ic, x, generator=np.random.default_rng(seed=seed), likelihood_config=likelihood_config)
            return i, c, logsumexp(log_likelihoods) - np.log(assignments.shape[0])
        case type_:
            raise ValueError(f'Unsupported approximation type {type_}')