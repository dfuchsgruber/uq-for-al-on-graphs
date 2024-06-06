from graph_al.data.base import Data, Dataset
from graph_al.data.config import DatasetSplit
from graph_al.evaluation.enum import MetricName
from graph_al.evaluation.result import Result
from graph_al.model.prediction import Prediction
from graph_al.model.trainer.loss import entropy_regularization, uce_loss
from graph_al.model.trainer.optimizer.build import get_optimizer
from graph_al.model.trainer.sgd import SGDTrainer
from graph_al.model.trainer.config import GPNTrainerConfig, GPNWarmup
from graph_al.model.gpn import GraphPosteriorNetwork
from graph_al.evaluation.config import MetricTemplate

from hydra.utils import instantiate
from tqdm import tqdm
import torch
import torch.nn.functional as F

from typeguard import typechecked
from jaxtyping import jaxtyped, Shaped, Float
from typing import Iterable, Dict
from torch import Tensor

class GPNTrainer(SGDTrainer):
    """ Trainer to train a GPN model """
    
    def __init__(self, config: GPNTrainerConfig, model: GraphPosteriorNetwork, dataset: Dataset, generator: torch.Generator):
        super().__init__(config, model, dataset, generator)
        self.warmup = config.warmup
        self.num_warmup_epochs = config.num_warmup_epochs
        self.flow_lr = config.flow_lr
        self.flow_weight_decay = config.flow_weight_decay
        self.entropy_regularization_loss_weight = config.entropy_regularization_loss_weight
        self.warmup_epoch = -1
        self.warmup_optimizer_config = config.warmup_optimizer
    
    @typechecked
    def get_optimizer(self, model: GraphPosteriorNetwork) -> torch.optim.Optimizer:
        """ Gets the optimizer(s) for the flow and non-flow components of the model.

        Args:
            model (GraphPosteriorNetwork): the model

        Returns:
            torch.optim.Optimizer: the optimizer
        """
            # return get_optimizer(self.optimizer_config, model.parameters())
        return get_optimizer(self.optimizer_config,                     
            [
                {'params' : model.flow_parameters, 'lr' : self.flow_lr, 'weight_decay' : self.flow_weight_decay},
                {'params' : model.non_flow_parameters}
            ])
    
    @typechecked
    def get_warmup_flow_optimizer(self, model: GraphPosteriorNetwork) -> torch.optim.Optimizer:
        """Gets the optimizer for the flow component of the model

        Args:
            model (GraphPosteriorNetwork): the model

        Returns:
            torch.optim.Optimizer: the optimizer
        """
        return get_optimizer(self.warmup_optimizer_config, model.flow_parameters)
        
        
    def entropy_regularization(self, batch: Data, prediction: Prediction, which: DatasetSplit,
                               loss_weights: Float[Tensor, 'mask_size'] | None = None, 
                               approximate: bool=True) -> Float[Tensor, '']:
        """Computes the regularizer for the entropy of the outputted distribution. """
        mask = batch.get_mask(which) & (batch.y < prediction.num_classes)
        alpha = prediction.alpha
        assert alpha is not None and alpha.size(0) == 1
        alpha = alpha.mean(0)
        reg = entropy_regularization(alpha[mask], approximate=approximate)
        if loss_weights is not None:
            reg *= loss_weights
        return reg.sum()
        
    def uce_loss(self, batch: Data, prediction: Prediction, epoch_idx: int, which: DatasetSplit, 
                    loss_weights: Float[Tensor, 'mask_size'] | None = None) -> Float[Tensor, '']:
        """Computes the uncertainty cross entropy loss (UCE loss) """
        alpha = prediction.alpha
        assert alpha is not None and alpha.size(0) == 1
        alpha = alpha.mean(0)
        mask = batch.get_mask(which) & (batch.y < prediction.num_classes)
        uce = uce_loss(alpha[mask], batch.y[mask])
        if loss_weights is not None:
            uce *= loss_weights
        return uce.sum()
        
    def loss(self, batch: Data, prediction: Prediction, epoch_idx: int, which: DatasetSplit, 
                    loss_weights: Float[Tensor, 'mask_size'] | None = None, approximate: bool=True) -> Dict[MetricTemplate, float | int | Shaped[Tensor, '']]:
        """Computes the entire training objective, i.e. UCE loss and entropy regularization. """
        uce_loss = self.uce_loss(batch, prediction, epoch_idx, which, loss_weights=loss_weights)
        entropy_reg = self.entropy_regularization(batch, prediction, which, loss_weights=loss_weights, approximate=approximate)
        total_loss = uce_loss + self.entropy_regularization_loss_weight * entropy_reg

        # with torch.no_grad():
        #     if which == DatasetSplit.TRAIN:
        #         beta_id = prediction.beta.mean(0).sum(-1)[batch.y < 4]
        #         beta_ood = prediction.beta.mean(0).sum(-1)[batch.y >= 4]

        #         from torchmetrics.functional.classification.auroc import binary_auroc       

        #         print('beta id', list(sorted(beta_id.tolist()))[::100])
        #         print('beta ood', list(sorted(beta_ood.tolist()))[::100])
        #         print('aucroc', binary_auroc(prediction.beta.mean(0).sum(-1).log(), (batch.y < 4).long()))

        #         torch.save({
        #             'id' : beta_id.cpu(),
        #             'ood' : beta_ood.cpu(),
        #             'all' : prediction.beta.mean(0).sum(-1).cpu(),
        #             'mask' : (batch.y < 4).long().cpu(),
        #         }, 'results.pt')



        return {
            MetricTemplate(name=MetricName.LOSS, dataset_split=which) : total_loss,
            MetricTemplate(name=MetricName.UCE_LOSS, dataset_split=which) : uce_loss,
            MetricTemplate(name=MetricName.ENTROPY_REGULARIZATION_LOSS, dataset_split=which) : entropy_reg,
        }
      
    def cross_entropy_loss(self, batch: Data, prediction: Prediction, epoch_idx: int, which: DatasetSplit, 
                    loss_weights: Float[Tensor, 'mask_size'] | None = None) -> Dict[MetricTemplate, float | int | Shaped[Tensor, '']]:
        """Computes the normal cross entropy loss when only warming up the encoder"""
        probabilities = prediction.get_probabilities(propagated=True)
        assert probabilities is not None and probabilities.size(0) == 1
        log_probs = probabilities.mean(0).log() # type: ignore
        mask = batch.get_mask(which) & batch.y < prediction.num_classes
        ce_loss = F.nll_loss(log_probs[mask], batch.y[mask], reduction='none')
        if loss_weights is not None:
            ce_loss *= loss_weights
        ce_loss = ce_loss.mean()
        return {
            MetricTemplate(name=MetricName.LOSS, dataset_split=which) : ce_loss,
        }
        
    def warmup_step(self, batch: Data, prediction: Prediction, epoch_idx: int, which: DatasetSplit=DatasetSplit.TRAIN, 
                    loss_weights: Float[Tensor, 'mask_size'] | None = None) -> Dict[MetricTemplate, float | int | Shaped[Tensor, '']]:
        """ One step of warmup training"""
        match self.warmup:
            case GPNWarmup.ENCODER:
                return self.cross_entropy_loss(batch, prediction, epoch_idx, which=which, loss_weights=loss_weights)
            case GPNWarmup.FLOW:
                return self.loss(batch, prediction, epoch_idx, which=which, loss_weights=loss_weights)
            case _:
                raise ValueError(self.warmup)
        
    @jaxtyped(typechecker=typechecked)
    def warmup_epoch_loop(self, model: GraphPosteriorNetwork, dataset: Dataset, generator: torch.Generator, epoch_idx: int,
                   optimizer: torch.optim.Optimizer) -> Dict[MetricTemplate, float | int | Shaped[Tensor, '']]:
        """ Performs one epoch of warm-up training"""
        warmup_metrics = dict()
        model = model.train()
        
        batch = dataset.data
        optimizer.zero_grad()
        prediction: Prediction = model.predict(batch)
        warmup_metrics |= self.warmup_step(batch, prediction, epoch_idx)
        loss = warmup_metrics[MetricTemplate(name=MetricName.LOSS, dataset_split=DatasetSplit.TRAIN)]
        loss.backward()
        optimizer.step()
        
        model = model.eval()
        warmup_metrics |= self.warmup_step(batch, prediction, epoch_idx, which=DatasetSplit.VAL)
        
        return warmup_metrics
    
    @typechecked
    def setup_warmup_epoch_iterator(self) -> Iterable[int]:
        iterator = range(self.num_warmup_epochs)
        if self.has_progress_bar:
            self.progress_bar = tqdm(iterator)
            iterator = self.progress_bar
        return iterator
        
    def fit(self, model: GraphPosteriorNetwork, dataset: Dataset, generator: torch.Generator, acquisition_step: int):
        """ Performs training of a model with SGD over different epochs.

        Args:
            batch (Data): The data on which to train on
            training_config (SGDTrainingConfig): Configuration for training
            generator (torch.Generator): A random number generator
            acquisition_step (int): Which acquisition step it is
        """
        model, dataset = self.transfer_model_to_device(model), self.transfer_dataset_to_device(dataset) # type: ignore
        
        # Warmup training for the flow
        match self.warmup:
            case GPNWarmup.FLOW:
                warmup_optimizer = self.get_warmup_flow_optimizer(model)
            case GPNWarmup.ENCODER:
                warmup_optimizer = self.get_optimizer(model)
            case _:
                raise ValueError(f'Warmup type {self.warmup} not supported.')
        
        for warmup_epoch in self.setup_warmup_epoch_iterator():
            self.warmup_epoch = warmup_epoch
            warmup_metrics = self.warmup_epoch_loop(model, dataset, generator, warmup_epoch, warmup_optimizer)
            loss = warmup_metrics.get(MetricTemplate(name=MetricName.LOSS, dataset_split=DatasetSplit.TRAIN), None)
            if loss:
                self.update_progress_bar('Warmup: '  + ', '.join(f'{key} : {value:.3f}' for key, value in {
                    'train loss' : warmup_metrics.get(MetricTemplate(name=MetricName.LOSS, dataset_split=DatasetSplit.TRAIN), None),
                    'val loss' : warmup_metrics.get(MetricTemplate(name=MetricName.LOSS, dataset_split=DatasetSplit.VAL), None),
                }.items()))
            
            warmup_metrics_to_log = {f'warmup/{key}' : value for key, value in warmup_metrics.items()}
            self.commit_metrics_to_wandb(warmup_metrics_to_log, self.warmup_epoch,)
        
        # Normal training using SGD: `self.loss` is overridden with the UCE loss that is used to train GPN
        super().fit(model, dataset, generator, acquisition_step)
    