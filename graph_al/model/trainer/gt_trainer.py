import torch
from typing import Dict, Iterable

from jaxtyping import jaxtyped, Shaped
from typeguard import typechecked

from graph_al.model.prediction import Prediction
from graph_al.data.base import Data, Dataset
from graph_al.model.base import BaseModel
from graph_al.model.trainer.config import OracleTrainerConfig
from graph_al.model.trainer.base import BaseTrainer
from graph_al.evaluation.result import Result
from graph_al.evaluation.config import MetricTemplate
from graph_al.evaluation.enum import MetricName
from graph_al.data.config import DatasetSplit

class OracleTrainer(BaseTrainer):

    def __init__(self, config: OracleTrainerConfig, model: BaseModel, dataset: Dataset, generator: torch.Generator):
        super().__init__(config, model, dataset, generator)

    @torch.no_grad()
    @jaxtyped(typechecker=typechecked)
    def any_steps(self, which: Iterable[DatasetSplit], batch: Data, prediction: Prediction, epoch_idx: int) -> Dict[MetricTemplate, float | int | Shaped[torch.Tensor, '']]:
        metrics = super().any_steps(which, batch, prediction, epoch_idx)
        for split in which:
            metrics[MetricTemplate(name=MetricName.LOSS, dataset_split=split)] = float('nan')
        return metrics
    
    def fit(self, model: BaseModel, dataset: Dataset, generator: torch.Generator, acquisition_step: int):
        """ Fits the model to a dataset.

        Args:
            model (BaseModel): The model to fit
            dataset (Dataset): The dataset to fit to
            generator (torch.Generator): A random number generator
            acquisition_step (int): Which acquisition step it is

        Returns:
            Result: Metrics, etc. for this model fit
        """
        super().fit(model, dataset, generator, acquisition_step)