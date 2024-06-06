import torch

from typeguard import typechecked 
from typing import Dict

from graph_al.data.base import Data, Dataset
from graph_al.data.enum import DatasetSplit
from graph_al.model.trainer.base import BaseTrainer
from graph_al.model.trainer.evaluation.config import TrainerEvaluationConfig
from graph_al.model.base import BaseModel
from graph_al.model.prediction import Prediction
from graph_al.utils.logging import get_logger
from graph_al.evaluation.enum import MetricTemplate
from graph_al.model.trainer.evaluation.calibration import evaluate_ece

@torch.no_grad()
@typechecked
def evaluate(config: TrainerEvaluationConfig, model: BaseModel, trainer: BaseTrainer, dataset: Dataset, generator: torch.Generator) -> Dict[MetricTemplate, float]:
    """Evaluates the result of one single model training by computing final metrics over the training.

    Args:
        config (TrainerEvaluationConfig): the config according to which to evaluate
        model (BaseModel): the model with the final weights already loaded
        trainer (BaseTrainer): the trainer that was used for training
        dataset (Dataset): the dataset
        generator (torch.Generator): an rng

    Returns:
        Dict[MetricTemplate, float]: metrics for the model training
    """
    model, dataset = trainer.transfer_model_to_device(model), trainer.transfer_dataset_to_device(dataset)
    metrics = {}
    prediction = model.predict(dataset.data, acquisition=True) # We want all attributes possible
    if prediction is not None:
        metrics |= evaluate_trainer_step(config, prediction, model, trainer, dataset.data, generator)
        if config.ece is not None:
            metrics |= evaluate_ece(config.ece, prediction, dataset.data, None)
    else:
        get_logger().warn('Model does not make predictions')

    return metrics


@torch.no_grad()
@typechecked
def evaluate_trainer_step(config: TrainerEvaluationConfig, prediction: Prediction, model: BaseModel, trainer: BaseTrainer, data: Data, generator: torch.Generator) -> Dict[MetricTemplate, float]:
    """Evaluate the metrics logged by the trainer (e.g. losses) by running the corresponding step methods.

    Args:
        config (TrainerEvaluationConfig): the config for evaluation
        prediction (Prediction): the prediction for which to evaluate
        model (BaseModel): the model
        trainer (BaseTrainer): the trainer
        data (Data): the data for which the prediction was made
        generator (torch.Generator): an rng

    Returns:
        Dict[MetricTemplate, float]: metrics logged by the trainer
    """
    metrics = trainer.any_steps(DatasetSplit, data, prediction, -1)
    return {metric : value.item() if isinstance(value, torch.Tensor) else float(value) for metric, value in metrics.items()}
