from abc import abstractmethod

from typing import Any, Dict, Iterable
from typeguard import typechecked
from jaxtyping import jaxtyped, Shaped

from graph_al.data.config import DatasetSplit
from graph_al.model.base import BaseModel
from graph_al.model.prediction import Prediction
from graph_al.model.trainer.config import TrainerConfig
from graph_al.data.base import Data, Dataset
from graph_al.evaluation.result import Result
from graph_al.evaluation.enum import MetricName, MetricTemplate
from graph_al.evaluation.calibration import compute_ece

import torch
import torch_scatter
from torchmetrics.functional import accuracy, f1_score


from graph_al.utils.utils import apply_to_nested_tensors

class BaseTrainer:
    """ Base class for model training. """
    
    def __init__(self, config: TrainerConfig, model: BaseModel, dataset: Dataset, generator: torch.Generator):
        self.verbose = config.verbose

    @torch.no_grad()
    @jaxtyped(typechecker=typechecked)
    def any_steps(self, which: Iterable[DatasetSplit], batch: Data, prediction: Prediction, epoch_idx: int) -> Dict[MetricTemplate, float | int | Shaped[torch.Tensor, '']]:
        """ Computes metrics on all of the dataset (not considering splits)

        Args:
            batch (Data): the batch to predict for
            prediction (Prediction): the model predictions made on this batch
            epoch_idx (int): which epoch

        Returns:
            Dict[MetricTemplate, float | int | Shaped[torch.Tensor, '']]: test metrics
        """

        metrics = dict()
        for split in which:
            mask = batch.get_mask(split) & (batch.y < batch.num_classes)
            for propagated in (True, False):
                labels = prediction.get_predictions(propagated=bool(propagated))
                if labels is not None and (mask.sum() > 0):
                    metrics |= {
                        MetricTemplate(name=MetricName.ACCURACY, dataset_split=split, propagated=propagated) : accuracy(
                            labels[mask], batch.y[mask], task='multiclass', 
                                                        num_classes=batch.num_classes).item(),
                        MetricTemplate(name=MetricName.F1, dataset_split=split, propagated=propagated) : f1_score(
                            labels[mask], batch.y[mask], task='multiclass', 
                                                        num_classes=batch.num_classes).item()
                        }
                else:
                    metrics |= {
                        MetricTemplate(name=MetricName.ACCURACY, dataset_split=split, propagated=propagated) : float('nan'),
                        MetricTemplate(name=MetricName.F1, dataset_split=split, propagated=propagated) : float('nan'),
                        }
        return metrics

    def fit(self, model: BaseModel, dataset: Dataset, generator: torch.Generator, acquisition_step: int) -> Result:
        """ Fits the model to a dataset.

        Args:
            model (BaseModel): The model to fit
            dataset (Dataset): The dataset to fit to
            generator (torch.Generator): A random number generator
            acquisition_step (int): Which acquisition step it is

        Returns:
            Result: Metrics, etc. for this model fit
        """
        ...
    
    def transfer_model_to_device(self, model: BaseModel) -> BaseModel:
        return model
    
    def transfer_dataset_to_device(self, dataset: Dataset) -> Dataset:
        return dataset
    