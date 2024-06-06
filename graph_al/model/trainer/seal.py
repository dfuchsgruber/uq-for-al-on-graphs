from graph_al.data.base import Data, Dataset
from graph_al.data.config import DatasetSplit
from graph_al.evaluation.enum import MetricName
from graph_al.evaluation.result import Result
from graph_al.model.prediction import Prediction
from graph_al.model.trainer.loss import entropy_regularization, uce_loss
from graph_al.model.trainer.optimizer.build import get_optimizer
from graph_al.model.trainer.sgd import SGDTrainer
from graph_al.model.trainer.config import SEALTrainerConfig
from graph_al.model.seal import SEAL
from graph_al.evaluation.config import MetricTemplate

from hydra.utils import instantiate
from tqdm import tqdm
import torch
import torch.nn.functional as F

from typeguard import typechecked
from jaxtyping import jaxtyped, Shaped, Float, Int
from typing import Iterable, Dict, Tuple
from torch import Tensor

from graph_al.utils.logging import get_logger

class SEALTrainer(SGDTrainer):
    """ Trainer to train a GPN model """
    
    @typechecked
    def __init__(self, config: SEALTrainerConfig, model: SEAL, dataset: Dataset, generator: torch.Generator):
        super().__init__(config, model, dataset, generator)
        self.discriminator_optimizer_config = config.discriminator_optimizer
        self.num_discriminator_epochs = config.num_discriminator_epochs
        self.num_samples = config.num_samples
        self.discriminator_supervised_loss_weight = config.discriminator_supervised_loss_weight
    
    @typechecked
    def get_optimizer(self, model: SEAL) -> torch.optim.Optimizer:
        """ Gets the optimizer(s) for the GCN

        Args:
            model (SEAL): the model

        Returns:
            torch.optim.Optimizer: the optimizer
        """
        return get_optimizer(self.optimizer_config, model.gcn.parameters())
    
    @typechecked
    def get_discriminator_optimizer(self, model: SEAL) -> torch.optim.Optimizer:
        """Gets the optimizer for the discriminator component of the model

        Args:
            model (SEAL): the model

        Returns:
            torch.optim.Optimizer: the optimizer
        """
        return get_optimizer(self.discriminator_optimizer_config, model.discriminator.parameters())
        
    def discriminator_step(self, batch: Data, prediction: Prediction, epoch_idx: int, which: DatasetSplit=DatasetSplit.TRAIN, 
                    loss_weights: Float[Tensor, 'mask_size'] | None = None) -> Dict[MetricTemplate, float | int | Shaped[Tensor, '']]:
        """ One step of warmup training"""
        logits = prediction.discriminator_logits
        assert logits is not None
        assert logits.size(0) == 1
        logits = logits.mean(0)
        mask = batch.get_mask(which) & batch.y < prediction.num_classes
        cross_entropy_loss = F.cross_entropy(logits[mask], batch.y[mask], reduction='none')
        if loss_weights is not None:
            cross_entropy_loss *= loss_weights
        cross_entropy_loss = cross_entropy_loss.mean()
        
        # Adversarial loss
        idxs_pseudo_labeled, idxs_pseudo_unlabeled = self.sample_adversial_idxs(batch, prediction, generator=None)
        
        adversarial_probabilities = prediction.get_adversarial_is_labeled_probabilities()
        assert adversarial_probabilities is not None
        adversarial_probabilities = adversarial_probabilities.mean(0) # ensemble average
        adversial_loss = -adversarial_probabilities[idxs_pseudo_labeled].log().mean(0) + \
            (1 - adversarial_probabilities[idxs_pseudo_unlabeled]).log().mean(0)
        
        return {
            MetricTemplate(name=MetricName.LOSS, dataset_split=which) : self.discriminator_supervised_loss_weight * cross_entropy_loss + adversial_loss,
            MetricTemplate(name=MetricName.ADVERSIAL_LOSS, dataset_split=which) : adversial_loss,
        }
        
    @jaxtyped(typechecker=typechecked)
    def discriminator_epoch_loop(self, model: SEAL, dataset: Dataset, generator: torch.Generator, epoch_idx: int,
                   optimizer: torch.optim.Optimizer) -> Dict[MetricTemplate, float | int | Shaped[Tensor, '']]:
        """ Performs one epoch of warm-up training"""
        discriminator_metrics = dict()
        model = model.train()
        
        batch = dataset.data
        optimizer.zero_grad()
        prediction: Prediction = model.predict(batch)
        discriminator_metrics |= self.discriminator_step(batch, prediction, epoch_idx)
        loss = discriminator_metrics[MetricTemplate(name=MetricName.LOSS, dataset_split=DatasetSplit.TRAIN)]
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            model = model.eval()
            discriminator_metrics |= self.discriminator_step(batch, prediction, epoch_idx, which=DatasetSplit.VAL)
        
        return discriminator_metrics
    
    @typechecked
    def setup_discriminator_epoch_iterator(self) -> Iterable[int]:
        iterator = range(self.num_discriminator_epochs)
        if self.has_progress_bar:
            self.progress_bar = tqdm(iterator, desc='Discriminator')
            iterator = self.progress_bar
        return iterator
        
    def sample_adversial_idxs(self, batch: Data, prediction: Prediction, generator: torch.Generator | None = None) -> Tuple[
        Int[Tensor, 'num_samples_labeled'],
        Int[Tensor, 'num_samples_unlabeled']
    ]:
        """ Gets idxs for computing the expectations in adversarial loss terms. """
        probabilities = prediction.get_probabilities(propagated=True)
        assert probabilities is not None
        is_pseudo_labeled = batch.is_pseudo_labeled(probabilities.mean(0))
        idxs_pseudo_labeled = torch.where(is_pseudo_labeled)[0]
        idxs_pseudo_labeled = idxs_pseudo_labeled[torch.randperm(len(idxs_pseudo_labeled), generator=generator)]
        idxs_pseudo_unlabeled = torch.where(~is_pseudo_labeled)[0]
        idxs_pseudo_unlabeled = idxs_pseudo_unlabeled[torch.randperm(len(idxs_pseudo_unlabeled), generator=generator)]
        
        if self.num_samples is not None:
            idxs_pseudo_labeled = idxs_pseudo_labeled[:self.num_samples]
            idxs_pseudo_unlabeled = idxs_pseudo_labeled[:self.num_samples]
            
        return idxs_pseudo_labeled, idxs_pseudo_unlabeled
        
        
    def loss(self, batch: Data, prediction: Prediction, epoch_idx: int, which: DatasetSplit, loss_weights: Tensor | None = None) -> Dict[MetricTemplate, float | int | Tensor]:
        losses = super().loss(batch, prediction, epoch_idx, which, loss_weights)
        
        # In addition to the GCN loss, we force the distributions of the embeddings of unlabeled and labeled data to be similar
        # via the mean feature discrepancy loss term
        idxs_pseudo_labeled, idxs_pseudo_unlabeled = self.sample_adversial_idxs(batch, prediction, generator=None)
        assert prediction.discriminator_embeddings is not None
        discriminator_embeddings = prediction.discriminator_embeddings.mean(0)
        embeddings_pseudo_labeled: Float[Tensor, 'num_features'] = discriminator_embeddings[idxs_pseudo_labeled].mean(0)
        embeddings_pseudo_unlabeled: Float[Tensor, 'num_features'] = discriminator_embeddings[idxs_pseudo_unlabeled].mean(0)
        mean_feature_discrepancy = torch.norm(embeddings_pseudo_labeled - embeddings_pseudo_unlabeled, p=2, dim=0)
        losses[MetricTemplate(name=MetricName.MEAN_FEATURE_DISCREPANCY, dataset_split=which, propagated=self.logits_propagated)] = mean_feature_discrepancy
        losses[MetricTemplate(name=MetricName.LOSS, dataset_split=which, propagated=self.logits_propagated)] += mean_feature_discrepancy
        return losses
        
    def fit(self, model: SEAL, dataset: Dataset, generator: torch.Generator, acquisition_step: int):
        """ Performs training of a model with SGD over different epochs.

        Args:
            batch (Data): The data on which to train on
            training_config (SGDTrainingConfig): Configuration for training
            generator (torch.Generator): A random number generator
            acquisition_step (int): Which acquisition step it is
        """
        model, dataset = self.transfer_model_to_device(model), self.transfer_dataset_to_device(dataset) # type: ignore
        self.model = model
        
        model.unfreeze_gcn()
        super().fit(model, dataset, generator, acquisition_step) # Fits the GCN
        
        # Fit the discriminator
        model.freeze_gcn()
        self.setup_early_stopping()
        discriminator_optimizer = self.get_discriminator_optimizer(model)
        for discriminator_epoch in self.setup_discriminator_epoch_iterator():
            self.discriminator_epoch = discriminator_epoch
            discriminator_metrics = self.discriminator_epoch_loop(model, dataset, generator, discriminator_epoch, discriminator_optimizer)
            
            discriminator_metrics_to_log = {f'discriminator/{key}' : value for key, value in discriminator_metrics.items()}
            self.commit_metrics_to_wandb(discriminator_metrics_to_log, self.discriminator_epoch,)
            self.early_stopping_step(discriminator_metrics, self.discriminator_epoch, model) # type: ignore
            if self.should_stop(self.discriminator_epoch):
                if self.verbose:
                    get_logger().info(f'Early stopping discriminator after {self.discriminator_epoch} epochs.')
                break
            
        model.unfreeze_gcn()