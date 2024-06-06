from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from graph_al.evaluation.enum import MetricTemplate

from typing import List, Dict

from graph_al.evaluation.enum import *

@dataclass
class AcquisitionPlotConfig:
    """ Config for generating a plot over the entire acquisition run and logs it to wandb """
    # Which metrics to show in this plot
    metrics: List[MetricTemplate] = MISSING
    # The name of the plot for wandb
    name: str | None = None
    
    # keyword arguments for `plt.figure`
    figure_kwargs: Dict | None = None
    # whether to create a legend
    legend: bool = True
    # x, y labels of the axis
    x_label: str | None = None
    y_label: str | None = None

@dataclass
class EvaluationConfig:
    
    # Metrics that will be logged against the entire acquisition process, i.e. after each model training for the best model
    # They will be averaged over all runs and logged to wandb
    # Note that the non averaged values of all metrics will be saved to the logdir anyway
    acquisition_metrics: List[MetricTemplate] = field(default_factory=lambda: [
        MetricTemplate(name=MetricName.ACCURACY, dataset_split=DatasetSplit.TRAIN),
        MetricTemplate(name=MetricName.ACCURACY, dataset_split=DatasetSplit.VAL),
        MetricTemplate(name=MetricName.LOSS, dataset_split=DatasetSplit.TRAIN),
        MetricTemplate(name=MetricName.LOSS, dataset_split=DatasetSplit.VAL),
        MetricTemplate(name=MetricName.F1, dataset_split=DatasetSplit.TRAIN),
        MetricTemplate(name=MetricName.F1, dataset_split=DatasetSplit.VAL),
        MetricTemplate(name=MetricName.ACQUIRED_CLASS_DISTRIBUTION_ENTROPY),
        MetricTemplate(name=MetricName.ACQUIRED_CLASS_DISTRIBUTION),
        ])
    
    # Which plots to make in wandb: Each plot is metric(s) over the entire acquisition
    # where the values are averaged
    acquisition_plots: List[AcquisitionPlotConfig] = field(default_factory=lambda: [])
    
    # For which metrics logged over the entire acquisition to compute an area under the logged curve
    # logs also a value normalized to [0, 1]
    log_acquisition_area_under_the_curve: List[MetricTemplate] = field(default_factory=lambda: [])

    evaluate_ece: bool = True # whether to log the ece after each model training
    

cs = ConfigStore.instance()
cs.store(name="base_evaluation", node=EvaluationConfig, group='evaluation')