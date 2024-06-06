import torch

from jaxtyping import jaxtyped, Bool
from typeguard import typechecked 
from typing import Dict

from graph_al.data.base import Data
from graph_al.data.enum import DatasetSplit
from graph_al.evaluation.calibration import compute_ece
from graph_al.model.trainer.evaluation.config import TrainerEvaluationECEConfig
from graph_al.model.prediction import Prediction
from graph_al.evaluation.enum import MetricTemplate, MetricName

@jaxtyped(typechecker=typechecked)
def evaluate_ece(config: TrainerEvaluationECEConfig | None, prediction: Prediction, data: Data, 
    mask: Bool[torch.Tensor, 'num_nodes'] | None) -> Dict[MetricTemplate, float]:
    """Computes the exepected calibration error for a prediction for all splits.

    Args:
        config (TrainerEvaluationECEConfig): configuration according to which to compute the ece
        prediction (Prediction): the prediction for which to compute the ECE
        data (Data): the data bach
        mask (Bool[torch.Tensor, &#39;num_nodes&#39;] | None): an optional mask for the instances that are considered

    Returns:
        Dict[MetricTemplate, float]: ECE for all splits in `config.splits`
    """
    if config is None:
        return {}
    eces = {}
    for split in DatasetSplit:
        evaluation_mask = mask & data.get_mask(split) if mask is not None else data.get_mask(split)
        if (evaluation_mask).sum().item() == 0:
            continue
        for propagated in (True, False):
            probabilities = prediction.get_probabilities(propagated=bool(propagated))
            if probabilities is not None:
                values = probabilities.mean(0) # average over multiple samples
                eces[MetricTemplate(name=MetricName.ECE, dataset_split=split, propagated=propagated)] = compute_ece(values[evaluation_mask], 
                    data.y[evaluation_mask], bins=config.num_bins, eps=config.eps)
    return eces
