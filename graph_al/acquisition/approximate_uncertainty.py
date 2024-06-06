from graph_al.acquisition.config import AcquisitionStrategyApproximateUncertaintyConfig
from graph_al.data.base import Data, Dataset
from graph_al.data.config import DatasetSplit
from graph_al.model.base import BaseModel
from graph_al.model.config import ModelConfig
from graph_al.model.prediction import Prediction
from graph_al.acquisition.attribute import AcquisitionStrategyByAttribute
from graph_al.model.sgc import SGC
from graph_al.model.build import get_model

from jaxtyping import jaxtyped, Bool, Int, Float
from typeguard import typechecked
from torch import Generator, Tensor

import itertools
import torch
import numpy as np
from tqdm import tqdm
from graph_al.utils.logging import get_logger
from sklearn.linear_model import LogisticRegression
import scipy.special

from graph_al.utils.timer import Timer

class AcquisitionStrategyApproximateUncertainty(AcquisitionStrategyByAttribute):
    
    """ Acquisition strategy that uses approximation to the ground truth uncertainty. 
    
    """
    
    def __init__(self, config: AcquisitionStrategyApproximateUncertaintyConfig):
        super().__init__(config)
        assert not self.balanced, f'Model does not support balanced acquisition currently'
        self.multiprocessing = config.multiprocessing
        self.num_workers = config.num_workers
        self.subsample_pool = config.subsample_pool
        self.aleatoric_confidence_with_left_out_node = config.aleatoric_confidence_with_left_out_node
        self.aleatoric_confidence_labels_num_samples = config.aleatoric_confidence_labels_num_samples
        self.compute_as_ratio = config.compute_as_ratio
        self.features_only = config.features_only


    @jaxtyped(typechecker=typechecked)
    def aleatoric_confidence(self, mask_predict: Bool[Tensor, 'n'], mask_train: Bool[Tensor, 'n'], 
                             labels: Int[Tensor, 'n'], model: BaseModel, 
                             dataset: Dataset, model_config: ModelConfig,
                             features_only: bool) -> Float[Tensor, 'n']:
        """ Computes (approximate) aleatoric confidence for all nodes in `mask_predict` for a "true" label
            assignment `labels`, i.e. it treats `labels` as ground_truth. """
        if isinstance(model, SGC):
            #  For SGC, re-training can be done cheaper
            return self.aleatoric_confidence_sgc(mask_predict, mask_train, labels, model, dataset.data, features_only)

        # We need to re-train a model from scratch
        from graph_al.active_learning import train_model
        labels = labels.clone()
        labels[mask_train] = dataset.data.y[mask_train]

        assert not self.aleatoric_confidence_with_left_out_node, 'Re-training a model for each node is too expensive'
        model_aleatoric = get_model(model_config, dataset, torch.default_generator)
        get_logger().info('Re-training model for aleatoric uncertainty...')
        with torch.enable_grad():
            train_mask_backup, train_labels_backup = dataset.data.mask_train.clone(), dataset.data.y.clone()
            dataset.data.mask_train[:], dataset.data.y = True, labels
            train_model(model_config.trainer, model_aleatoric, dataset, torch.default_generator)
            dataset.data.mask_train[:], dataset.data.y[:] = train_mask_backup, train_labels_backup
            
        prediction_aleatoric = model_aleatoric.predict(dataset.data, acquisition=True)
        probabilites = prediction_aleatoric.get_probabilities(propagated=not features_only)
        if probabilites is None:
            raise ValueError(f'Got no probabilities')
        aleatoric_confidence = probabilites.mean(0)[torch.arange(probabilites.size(0)), labels]
        return aleatoric_confidence

    @jaxtyped(typechecker=typechecked)
    def aleatoric_confidence_sgc(self, mask_predict: Bool[Tensor, 'n'], mask_train: Bool[Tensor, 'n'], 
                             labels: Int[Tensor, 'n'], model: SGC, 
                             batch: Data, features_only: bool) -> Float[Tensor, 'n']:
        """ Computes (approximate) aleatoric confidence for all nodes in `mask_predict` for a "true" label
            assignment `labels`, i.e. it treats `labels` as ground_truth. """
        labels = labels.clone()
        # replace the ground truth in labels when it is available
        labels[mask_train] = batch.y[mask_train]
        labels_np = labels.cpu().numpy()
        if features_only:
            x = batch.x.cpu().numpy()
        else:
            x = model.get_diffused_node_features(batch).cpu().numpy()
        aleatoric_confidence = torch.full((mask_train.size(0),), float('nan'), dtype=torch.float32)
        
        if self.aleatoric_confidence_with_left_out_node:
            # compute aleatoric confidence from one model trained on leaving out a node for each node
            iterator = tqdm( torch.where(mask_predict)[0]) if self.verbose else torch.where(mask_predict)[0]
            for idx in iterator:
                mask_aleatoric = np.ones_like(mask_train)
                mask_aleatoric[idx] = False
                aleatoric_confidence_idx = _probabilities_from_logistic_regression(x, 
                                                                        labels_np, mask_aleatoric, model, batch.num_classes)
                aleatoric_confidence[idx] = float(aleatoric_confidence_idx[idx, labels[idx]])
        else:
            # use one base model for aleatoric confidence, cheaper
            idxs = torch.where(mask_predict)[0]
            aleatoric_confidence[idxs] = torch.from_numpy(_probabilities_from_logistic_regression(x, 
                                                                        labels_np, np.ones_like(mask_train), 
                                                                        model, batch.num_classes)[idxs, labels[idxs]]).type(torch.float32)
            
        return aleatoric_confidence
    
    
    @jaxtyped(typechecker=typechecked)
    def epistemic_uncertainty_mp(self, mask_predict: Bool[Tensor, ' n'], mask_train: Bool[Tensor, ' n'], 
                                     total_confidence: Float[Tensor, 'n c'], model: BaseModel, 
                                     dataset: Dataset, model_config: ModelConfig,
                                     features_only: bool) -> Float[Tensor, 'n']:
        """ Computes approximate epistemic uncertainty as the ratio aleatoric confidence / total confidence."""
        idxs_train = torch.where(mask_train)[0]
        if self.aleatoric_confidence_labels_num_samples is None:
            aleatoric_samples = total_confidence.argmax(1)[..., None]
        else:
            aleatoric_samples = torch.multinomial(total_confidence, self.aleatoric_confidence_labels_num_samples, replacement=True)
            
        # For the observed labels, we do not sample
        aleatoric_samples[idxs_train] = dataset.data.y[idxs_train][..., None]
        
        aleatoric_confidences = torch.stack([
            self.aleatoric_confidence(mask_predict, mask_train, aleatoric_samples[:, i], model, dataset, model_config,
                                      features_only) 
            for i in range(aleatoric_samples.size(1))], dim=0)
        total_confidences = total_confidence[torch.arange(total_confidence.size(0)), aleatoric_samples.T]
        
        # print(total_confidence.size(), aleatoric_confidences.size())
        
        epistemic_uncertainties = aleatoric_confidences / (total_confidences + 1e-12)
        epistemic_uncertainty = epistemic_uncertainties.mean(0)
        epistemic_uncertainty[~mask_predict] = -float('inf')
        print('alea conf', torch.nanmean(aleatoric_confidences), aleatoric_confidences)
        print('epi uncertainty', torch.nanmean(epistemic_uncertainty), epistemic_uncertainty)
        print('class counts', torch.unique(aleatoric_samples.flatten(), return_counts=True))
        
        return epistemic_uncertainty
    
    @jaxtyped(typechecker=typechecked)
    def epistemic_uncertainty_argmax(self, mask_predict: Bool[Tensor, 'n'], mask_train: Bool[Tensor, 'n'], 
                                     total_confidence: Float[Tensor, 'n c'],
                                     labels: Int[Tensor, 'n'], model: SGC, 
                                     batch: Data, ) -> Float[Tensor, 'n']:
        
        labels = labels.clone()
        labels[mask_train] = batch.y[mask_train]
        labels_np = labels.cpu().numpy()
        x = model.get_diffused_node_features(batch).cpu().numpy()
        aleatoric_confidence = torch.full((mask_train.size(0),), float('nan'), dtype=torch.float32)
        
        iterator = tqdm( torch.where(mask_predict)[0])
        
        mask_aleatoric = np.ones_like(mask_train)
        prev_idx = None
        for idx in iterator:
            mask_aleatoric[idx] = False
            if prev_idx is not None:
                mask_aleatoric[prev_idx] = True
            prev_idx = idx
            with Timer() as timer:
                aleatoric_confidence_idx = _probabilities_from_logistic_regression(x, 
                                                                        labels_np, mask_aleatoric, model, batch.num_classes)
                aleatoric_confidence[idx] = aleatoric_confidence_idx[idx, labels[idx]]
        
        epistemic_uncertainty = aleatoric_confidence / total_confidence[torch.arange(total_confidence.size(0)), labels]
        epistemic_uncertainty[~mask_predict] = -float('inf')
        return epistemic_uncertainty

    @jaxtyped(typechecker=typechecked)
    def epistemic_uncertainty_esp(self, mask_predict: Bool[Tensor, 'n'], mask_train: Bool[Tensor, 'n'], 
                                     total_confidence: Float[Tensor, 'n c'],
                                     model: SGC, 
                                     batch: Data, 
                                     features_only: bool) -> Float[Tensor, 'n']:
        """ Computes approximate epistemic uncertainty as the gain in confidence in correct (=pseudo labels) predictions.
        
        Args:
            mask_predict: Bool[Tensor, 'n']: Mask of nodes to predict
            mask_train: Bool[Tensor, 'n']: Mask of training nodes
            total_confidence: Float[Tensor, 'n c']: Total confidence
            model: SGC: Model
            batch: Data: Batch
            features_only: bool: Whether to use only the features
            
        Returns:
            Float[Tensor, 'n']: The epistemic uncertainty
        """

        n, c = total_confidence.size()
        if features_only:
            x = batch.x.cpu().numpy()
        else:
            x = model.get_diffused_node_features(batch).cpu().numpy()
        mask_train_np: Bool[np.ndarray, 'n']  = mask_train.cpu().numpy()
        total_confidence_np = total_confidence.cpu().numpy()
        labels_np = total_confidence_np.argmax(1)

        joint_log_probs = torch.full_like(total_confidence, -float('inf'))

        if self.verbose: 
            print()
            
        # We simplify the computation of epistemic uncertainty up to a constant factor using
        # equation (28) in our paper, (see appendix G.2)
        for i, c in tqdm(list(itertools.product(torch.where(mask_predict)[0], range(batch.num_classes))), disable=not self.verbose):
            label_i_true, labels_np[i], mask_train_np[i] = batch.y[i], c, True
            # compute the log probability p(y_u-i = y_u-i^label | y_O, y_i=c)
            
            probs = _probabilities_from_logistic_regression(x, labels_np, mask_train_np, model, batch.num_classes)
            denominator_pseudo_labels = probs.argmax(1)

            joint_log_probs[i, c] = np.log(probs[np.arange(n), denominator_pseudo_labels][~mask_train_np]).sum()
            joint_log_probs[i, c] += np.log(total_confidence_np[i, int(labels_np[i])])
            labels_np[i], mask_train_np[i] = label_i_true, False

        # Now we take the expectation over the label for y_i = c weighted by the total confidence
        expected_ratios = joint_log_probs + np.log(total_confidence_np)
        expected_ratios = scipy.special.logsumexp(expected_ratios, axis=-1)
        assert not np.isnan(expected_ratios).any()
        return torch.from_numpy(expected_ratios).float()
            
    @jaxtyped(typechecker=typechecked)
    def get_attribute(self, prediction: Prediction | None, model: BaseModel, dataset: Dataset, generator: Generator,
                        model_config: ModelConfig) -> Tensor:
        """Computes approximate epistemic uncertainty.

        Args:
            prediction (Prediction | None): The prediction of the model
            model (BaseModel): The model
            dataset (Dataset): The dataset
            generator (Generator): The random number generator
            model_config (ModelConfig): The model configuration

        Returns:
            Tensor: The approximate epistemic uncertainty
        """
        assert prediction is not None, 'Approximate uncertainty expects a prediction'
        total_confidence = prediction.get_probabilities(propagated=not self.features_only)
        assert total_confidence is not None, 'Approximate uncertainty expects probabilities'
        total_confidence = total_confidence.mean(0) # ensemble average
        
        # For each node and potential label calculate the epistemic uncertainty
        mask_predict_nodes = np.ones(dataset.num_nodes, dtype=bool)
        mask_predict_nodes &= dataset.data.get_mask(DatasetSplit.TRAIN_POOL).cpu().numpy()
        if self.subsample_pool is not None:
            idxs_predict_nodes = np.where(mask_predict_nodes)[0]
            np.random.shuffle(idxs_predict_nodes)
            idxs_predict_nodes = idxs_predict_nodes[:self.subsample_pool]
            mask_predict_nodes = np.zeros_like(mask_predict_nodes)
            mask_predict_nodes[idxs_predict_nodes] = True
        
        if self.compute_as_ratio:
            # Compute the epistemic uncertainty as the ratio of aleatoric confidence / total confidence
            # In the paper, this is called MP (multiple pseudo labels)
            return self.epistemic_uncertainty_mp(torch.from_numpy(mask_predict_nodes), dataset.data.mask_train,
                                                     total_confidence, model, dataset, model_config, self.features_only)
        else:
            # Compute the epistemic uncertainty as the gain in confidence in correct (=pseudo labels) predictions
            assert isinstance(model, SGC)
            return self.epistemic_uncertainty_esp(torch.from_numpy(mask_predict_nodes), dataset.data.mask_train,
                                                    total_confidence, model, dataset.data, self.features_only)

@jaxtyped(typechecker=typechecked)
def _probabilities_from_logistic_regression(x: Float[np.ndarray, 'num_nodes num_features'],
                                            y: Int[np.ndarray, 'num_nodes'],
                                            mask_train: Bool[np.ndarray, 'num_nodes'],
                                            model: SGC, num_classes: int) -> Float[np.ndarray, 'num_nodes num_classes']:
    """Computes probabilities (i.e. confidence) from fitting a logistic regression model.
    
    Args:
        x: Float[np.ndarray, 'num_nodes num_features']: Node features
        y: Int[np.ndarray, 'num_nodes']: Labels
        mask_train: Bool[np.ndarray, 'num_nodes']: Mask of training nodes
        model: SGC: Model
        num_classes: int: Number of classes
        
    Returns:
        Float[np.ndarray, 'num_nodes num_classes']: Probabilities in the predictions fit to this data.
    """
    num_nodes = y.shape[0]
    if num_classes == 1:
        # Weird "bug" in LogisticRegression that it can not fit with only one class
        # The prediction of the logistic regression classifier should be 1.0 for this one class
        probabilities = np.zeros((num_nodes, num_classes))
        probabilities[:, y[mask_train][0]] = 1.0
    else:
        # automatically determine a good solver
        solver = model.solver if mask_train.sum() <= 10000 else 'saga'
        regression = LogisticRegression(C=model.inverse_regularization_strength, 
                                        solver=solver, # type: ignore
            class_weight='balanced' if model.balanced else None)
        regression.fit(x[mask_train], y[mask_train])
        probabilities = regression.predict_proba(x)
        del regression
    return probabilities




