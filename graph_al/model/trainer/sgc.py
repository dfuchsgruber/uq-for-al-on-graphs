import torch

from typeguard import typechecked
from typing import Dict, Iterable
from jaxtyping import jaxtyped, Shaped

from graph_al.data.config import DatasetSplit
from graph_al.evaluation.enum import MetricName
from graph_al.model.prediction import Prediction
from graph_al.data.base import Data, Dataset
from graph_al.model.sgc import SGC
from graph_al.model.trainer.config import SGCTrainerConfig
from graph_al.model.trainer.base import BaseTrainer
from graph_al.evaluation.result import Result
from graph_al.evaluation.config import MetricTemplate

class SGCTrainer(BaseTrainer):

    def __init__(self, config: SGCTrainerConfig, model: SGC, dataset: Dataset, generator: torch.Generator):
        super().__init__(config, model, dataset, generator)

    @jaxtyped(typechecker=typechecked)
    def any_steps(self, which: Iterable[DatasetSplit], batch: Data, prediction: Prediction, epoch_idx: int) -> Dict[MetricTemplate, float | int | Shaped[torch.Tensor, '']]:
        metrics = super().any_steps(which, batch, prediction, epoch_idx)
        for split in which:
            metrics[MetricTemplate(name=MetricName.LOSS, dataset_split=split)] = float('nan')
        return metrics

    @typechecked
    def fit(self, model: SGC, dataset: Dataset, generator: torch.Generator, acquisition_step: int):
        """ Fits the model to a dataset.

        Args:
            model (BaseModel): The model to fit
            dataset (Dataset): The dataset to fit to
            generator (torch.Generator): A random number generator
            acquisition_step (int): Which acquisition step it is

        Returns:
            Result: Metrics, etc. for this model fit
        """
        batch: Data = dataset.data
        mask_train = batch.get_mask(DatasetSplit.TRAIN)

        x = model.get_diffused_node_features(batch)[mask_train]
        labels_in_mask_train = torch.unique(batch.y[mask_train])
        if labels_in_mask_train.size(0) == 1:
            # "Bug" in sklearn's LogisticRegressionClassifier: It can not fit with only one class in the training set
            # Therefore we "hardcode" a constant prediction of 1.0 for this class into the classifier
            # via its 'frozen_prediction' feature
            model.freeze_predictions(labels_in_mask_train[0].item())
        else:
            model.unfreeze_predictions()
            model.logistic_regression.fit(x.numpy(), batch.y[mask_train].numpy())
            