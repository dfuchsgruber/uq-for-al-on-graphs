
import torch
import torch.nn.functional as F
from tqdm import tqdm
from hydra.utils import instantiate
import wandb
from collections import defaultdict

from typeguard import typechecked
from jaxtyping import Float, jaxtyped, Int, Shaped
from torch import Tensor
from typing import Dict, Iterable, List, Set

from graph_al.model.prediction import Prediction
from graph_al.data.base import Data, Dataset
from graph_al.model.base import BaseModel
from graph_al.model.trainer.config import SGDTrainerConfig, LossFunction
from graph_al.model.trainer.base import BaseTrainer
from graph_al.model.trainer.early_stopping import EarlyStopping
from graph_al.model.trainer.loss import balanced_loss_weights
from graph_al.evaluation.result import Result
from graph_al.model.trainer.optimizer.build import get_optimizer
from graph_al.utils.utils import apply_to_nested_tensors
from graph_al.utils.logging import get_logger
from graph_al.data.config import DatasetSplit
from graph_al.evaluation.enum import MetricName, MetricTemplate

class SGDTrainer(BaseTrainer):

    def __init__(self, config: SGDTrainerConfig, model: BaseModel, dataset: Dataset, generator: torch.Generator):
        super().__init__(config, model, dataset, generator)
        self.optimizer_config = config.optimizer
        self.early_stopping_config = config.early_stopping
        self.loss_function_type = config.loss
        self.balanced_loss = config.balanced_loss
        self.balanced_loss_beta = config.balanced_loss_beta
        self.balanced_loss_normalize = config.balanced_loss_normalize
        self.max_epochs = config.max_epochs
        self.min_epochs = config.min_epochs
        self.has_progress_bar = config.progress_bar
        self.commit_to_wandb_every_epoch = config.commit_to_wandb_every_epoch
        self.use_gpu = config.use_gpu
        self.kl_divergence_loss_weight = config.kl_divergence_loss_weight
        self.log_every_epoch = config.log_every_epoch
        self.logits_propagated = config.logits_propagated
        self.summary_metrics = config.summary_metrics

    @typechecked
    def get_optimizer(self, model: BaseModel) -> torch.optim.Optimizer:
        return get_optimizer(self.optimizer_config, model.parameters())
    
    def setup_early_stopping(self):
        if self.early_stopping_config is not None:
            self.early_stopping = EarlyStopping(self.early_stopping_config)
        else:
            self.early_stopping = None
            
    @typechecked
    def should_stop(self, epoch_idx: int) -> bool:
        if self.current_epoch < self.min_epochs:
            return False
        if self.early_stopping is not None:
            return self.early_stopping.should_stop
        else:
            return self.current_epoch >= self.max_epochs
        
    @typechecked
    def early_stopping_step(self, metrics: Dict[MetricTemplate, float | int | Shaped[Tensor, '']], epoch_idx: int,
                            model: BaseModel):
        if self.early_stopping is not None:
            self.early_stopping.step(metrics, epoch_idx, model=model)
            
    @property
    @typechecked
    def best_epoch(self) -> int:
        if self.early_stopping is not None:
            return self.early_stopping.best_epoch
        else:
            return -1
        
    @property
    @typechecked
    def best_state(self) -> Dict | None:
        if self.early_stopping is not None and self.early_stopping.save_model_state:
            return self.early_stopping.best_state
        else:
            return None
        

    @jaxtyped(typechecker=typechecked)
    def get_loss_weights(self, labels: Int[Tensor, 'num_nodes']) -> Float[Tensor, 'num_nodes']:
        if self.balanced_loss:
            return balanced_loss_weights(labels, self.balanced_loss_beta,
                                                       normalize=self.balanced_loss_normalize)
        else:
            return torch.ones_like(labels, dtype=torch.float)
    
    @typechecked
    def setup_epoch_iterator(self) -> Iterable[int]:
        iterator = range(self.max_epochs)
        if self.has_progress_bar:
            self.progress_bar = tqdm(iterator)
            iterator = self.progress_bar
        return iterator
    
    @typechecked
    def update_progress_bar(self, message: str):
        if self.has_progress_bar:
            self.progress_bar.set_description(message)
            
    def transfer_dataset_to_device(self, dataset: Dataset) -> Dataset:
        if self.use_gpu and torch.cuda.is_available():
            dataset = dataset.cuda()
        return dataset
    
    def transfer_model_to_device(self, model: BaseModel) -> BaseModel:
        if self.use_gpu and torch.cuda.is_available():
            model = model.cuda()
        return model

    @typechecked
    def should_log_in_epoch(self, epoch_idx: int) -> bool:
        log_every_epoch = self.log_every_epoch
        if log_every_epoch is None or log_every_epoch <= 0:
            return False
        else:
            return (epoch_idx % log_every_epoch) == 0

    @jaxtyped(typechecker=typechecked)
    def commit_metrics_to_wandb(self, metrics: Dict[str, float | int | Shaped[Tensor, '']], epoch_idx: int):
        if wandb.run is not None:
            if self.commit_to_wandb_every_epoch is None or self.commit_to_wandb_every_epoch < 1:
                commit = False
            else:
                commit = epoch_idx % self.commit_to_wandb_every_epoch == 0
            # Note that the values that are not commited will never be logged to wandb
            if wandb.run is not None and self.should_log_in_epoch(epoch_idx):
                wandb.run.log({f'trainer/{key}' : metrics[key] for key in metrics.keys()}, step=epoch_idx, commit=commit)
        
    @jaxtyped(typechecker=typechecked)
    def loss(self, batch: Data, prediction: Prediction, epoch_idx: int, which: DatasetSplit, 
                    loss_weights: Float[Tensor, 'mask_size'] | None = None) -> Dict[MetricTemplate, float | int | Shaped[Tensor, '']]:
        """ Computes the entire training objective """
        logits = prediction.get_logits(propagated=bool(self.logits_propagated))
        assert logits is not None, f'Can not compute loss without logits'
        if logits.size(0) > 1 and logits.grad is not None:
            get_logger().warn(f'Using average logits (over {logits.size(0)} samples) for loss function. This is not a good idea...')
        logits = logits.mean(0) # average over samples
        mask = batch.get_mask(which) & (batch.y < prediction.num_classes)
        match self.loss_function_type:
            case LossFunction.CROSS_ENTROPY:
                loss = F.cross_entropy(logits[mask], batch.y[mask], reduction='none')
                if loss_weights is not None:
                    loss *= loss_weights
                loss = loss.mean()
            case LossFunction.CROSS_ENTROPY_AND_KL_DIVERGENCE:
                loss = F.cross_entropy(logits[mask], batch.y[mask], reduction='none')
                if loss_weights is not None:
                    loss *= loss_weights
                loss = loss.mean()
                if prediction.kl_divergence is None or prediction.num_kl_terms is None:
                    raise RuntimeError(f'No KL divergence term in prediction')
                loss += prediction.kl_divergence.sum() / prediction.num_kl_terms.sum() * self.kl_divergence_loss_weight
            case _:
                raise ValueError(f'Unsupported loss function {self.loss_function_type}')
        
        return {MetricTemplate(name=MetricName.LOSS, dataset_split=which, propagated=self.logits_propagated) : loss}
        
    @jaxtyped(typechecker=typechecked)
    def any_loss_step(self, batch: Data, prediction: Prediction, epoch_idx: int, which: DatasetSplit,
                      loss_weights: Float[Tensor, 'mask_size'] | None = None) -> Dict[MetricTemplate, float | int | Shaped[Tensor, '']]:
        """ Calculates the loss for a prediction. """
        return self.loss(batch, prediction, epoch_idx, which, loss_weights=loss_weights)
    

    @jaxtyped(typechecker=typechecked)
    def any_steps(self, which: Iterable[DatasetSplit], batch: Data, prediction: Prediction, epoch_idx: int) -> Dict[MetricTemplate, float | int | Shaped[torch.Tensor, '']]:
        """Performs a step on any dataset split

        Args:
            which (DatasetSplit): on which split to perform the step
            batch (Data): the batch to predict for
            prediction (Prediction): the model predictions made on this batch
            epoch_idx (int): which epoch

        Returns:
            Dict[MetricTemplate, float | int | Shaped[torch.Tensor, '']]: train metrics, including the loss
        """
        metrics = super().any_steps(which, batch, prediction, epoch_idx)
        for split in which:
            loss_weights = self.get_loss_weights(batch.y[batch.get_mask(split) & (batch.y < prediction.num_classes)])
            with torch.set_grad_enabled(split == DatasetSplit.TRAIN):
                metrics |= self.any_loss_step(batch, prediction, epoch_idx, split, loss_weights=loss_weights)
        return metrics
  
    @jaxtyped(typechecker=typechecked)
    def epoch_loop(self, model: BaseModel, dataset: Dataset, generator: torch.Generator, epoch_idx: int,
                   optimizer: torch.optim.Optimizer) -> Dict[MetricTemplate, float | int | Shaped[Tensor, '']]:
        """ Overridable iteration of one training epoch.      

        Args:
            model (BaseModel): the model to train
            dataset (Dataset): the dataset on which to train
            generator (torch.Generator): a random number generator 
            epoch_idx (int): in which epoch
            optimizer (torch.optim.Optimizer): the optimizer

        Returns:
            Dict[MetricTemplate, float | int | Shaped[Tensor, '']]: Metrics for this epoch
        """
        epoch_metrics = dict()
        model = model.train()
        
        batch = dataset.data
        optimizer.zero_grad()
        prediction: Prediction = model.predict(batch)
        epoch_metrics |= self.any_steps([DatasetSplit.TRAIN], batch, prediction, epoch_idx)
        loss = epoch_metrics[MetricTemplate(name=MetricName.LOSS, dataset_split=DatasetSplit.TRAIN)]
        loss.backward()
        optimizer.step()
        
        model = model.eval()
        if model.prediction_changes_at_eval:
            with torch.no_grad():
                prediction = model.predict(batch)
        epoch_metrics |= self.any_steps([split for split in DatasetSplit if split != DatasetSplit.TRAIN], batch, prediction, epoch_idx)
        return apply_to_nested_tensors(epoch_metrics, lambda tensor: tensor.detach().cpu())

    def fit(self, model: BaseModel, dataset: Dataset, generator: torch.Generator, acquisition_step: int):
        """ Performs training of a model with SGD over different epochs.

        Args:
            batch (Data): The data on which to train on
            training_config (SGDTrainingConfig): Configuration for training
            generator (torch.Generator): A random number generator
            acquisition_step (int): Which acquisition step it is

        Returns:
            Dict[str, List]: Metrics logged over training.
        """
        model, dataset = self.transfer_model_to_device(model), self.transfer_dataset_to_device(dataset)
        
        optimizer = self.get_optimizer(model)
        self.setup_early_stopping()
        
        epoch_metrics = defaultdict(list)
        epoch_iterator = self.setup_epoch_iterator()
         
        for epoch_idx in epoch_iterator:
            self.current_epoch = epoch_idx
            metrics = self.epoch_loop(model, dataset, generator, self.current_epoch, optimizer)
            for name, value in metrics.items():
                epoch_metrics[name].append(value)
            loss = metrics.get(MetricTemplate(name=MetricName.LOSS, dataset_split=DatasetSplit.TRAIN), None)
            if loss:
                self.update_progress_bar(', '.join(f'{key} : {value:.3f}' for key, value in {
                    'train loss' : metrics.get(MetricTemplate(name=MetricName.LOSS, dataset_split=DatasetSplit.TRAIN), None),
                    'val loss' : metrics.get(MetricTemplate(name=MetricName.LOSS, dataset_split=DatasetSplit.VAL), None),
                    'train acc' : metrics.get(MetricTemplate(name=MetricName.ACCURACY, dataset_split=DatasetSplit.TRAIN), None),
                    'val acc' : metrics.get(MetricTemplate(name=MetricName.ACCURACY, dataset_split=DatasetSplit.VAL), None),
                }.items()))
            self.commit_metrics_to_wandb({str(k) : v for k, v in metrics.items()}, self.current_epoch)
            self.early_stopping_step(metrics, self.current_epoch, model) # type: ignore
            if self.should_stop(self.current_epoch):
                if self.verbose:
                    get_logger().info(f'Early stopping after {self.current_epoch} epochs.')
                break
        
        # Restore the best model
        if self.best_state is not None: # type: ignore
            model.load_state_dict(self.best_state)

        if self.verbose:
            for metric in self.summary_metrics:
                values = torch.tensor([v.item() if isinstance(v, torch.Tensor) else float(v) for v in epoch_metrics[metric]])
                print(metric, values.numpy().round(2)[::len(values) // 20])





    