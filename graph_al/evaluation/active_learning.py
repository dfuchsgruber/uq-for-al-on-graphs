from os import PathLike
from pathlib import Path
from graph_al.evaluation.config import EvaluationConfig, MetricTemplate
from graph_al.evaluation.enum import MetricName
from graph_al.evaluation.result import Results
from graph_al.evaluation.acquisition_curve import AcquisitionCurve
from graph_al.utils.logging import get_logger

from typing import Any, Dict, List
from typeguard import typechecked

import wandb
import numpy as np
import torch
from graph_al.utils.utils import apply_to_nested_tensors


def print_summary(results: List[Results]):
    """ Prints a pretty summary of the results """
    from rich.table import Table
    import rich
    
    metrics = set.union(*(set(r.results[-1].metrics.keys()) for r in results))
    table = Table('Metric', 'Mean', 'Std', title='Summary after budget exhausted')
    for metric in sorted(metrics, key=str):
        values = [r.results[-1].metrics.get(metric, None) for r in results]
        values = [v.item() if isinstance(v, torch.Tensor) and len(v.size()) == 0 else v for v in values]
        values = [float(value) for value in values if isinstance(value, float)]
        table.add_row(str(metric), f'{np.mean(values) if len(values) > 0 else np.nan:.2f}', 
                      f'{np.std(values) if len(values) > 1 else np.nan:.2f}')
    rich.print(table)

def evaluate_active_learning(config: EvaluationConfig, results: List[Results]) -> Dict[str, Any]:
    """ Runs final evaluation for the entire active learning prodcedure. 

    Args:
        config (EvaluationConfig): according to which config to run evaluation
        results (List[Results]): the results of active learning over multiple runs
    """
    metrics_keys = set.union(*(r.metrics for r in results))
    # Add metrics about the acquisition that every model should log
    curve = AcquisitionCurve.from_results(results, [metric for metric in metrics_keys | {
        MetricTemplate(name=MetricName.NUM_ACQUIRED),
        MetricTemplate(name=MetricName.ACQUIRED_CLASS_DISTRIBUTION_ENTROPY),
        MetricTemplate(name=MetricName.ACQUIRED_CLASS_DISTRIBUTION),}])
    
    summary_metrics = {}
    summary_metrics |= {str(metric) : [r.results[-1].metrics.get(metric, None) for r in results] for metric in metrics_keys}
    
    for plot in config.acquisition_plots:
        curve.plot(plot.metrics,
                   name=plot.name, figure_kwargs=plot.figure_kwargs, legend=plot.legend,
                   x_label=plot.x_label, y_label=plot.y_label)
        metrics_unnormalized = curve.area_under_the_curve(config.log_acquisition_area_under_the_curve,
                                normalized=False)
        metrics_normalized = curve.area_under_the_curve(config.log_acquisition_area_under_the_curve,
                                normalized=True)
        metrics = {f'{key}/unnormalized' : value for key, value in metrics_unnormalized.items()} | \
            {f'{key}/normalized' : value for key, value in metrics_normalized.items()}
        metrics = {f'{key}/auc' : auc for key, (auc, _) in metrics.items()} | \
            {f'{key}/final' : final for key, (_, final) in metrics.items()}
        summary_metrics |= metrics
        
        if wandb.run is not None:
            wandb.run.log({f'{key}_auc/mean' : values.float().mean() for key, values in metrics.items()} | \
                {f'{key}/std' : values.float().std() for key, values in metrics.items()})
            
    return summary_metrics
            
@typechecked
def aggregate_results(results: List[Results]) -> Dict[MetricTemplate | str, Any]:
    output = {}
    # Log all metrics obtained by training
    keys = set.union(*(set.union(*(set(result.metrics.keys()) for result in rs.results)) for rs in results))
    output |= {key : [
        [apply_to_nested_tensors(result.metrics.get(key, None), lambda tensor: tensor.detach().cpu()) 
         for result in rs.results]
            for rs in results
             ] for key in keys}
    # Log all metrics obtained by acquisition
    acquisition_keys = set.union(*(set.union(*(set(result.acquisition_metrics.keys()) for result in rs.results)) for rs in results))
    output |= {key : [
        [apply_to_nested_tensors(result.acquisition_metrics.get(key, None), lambda tensor: tensor.detach().cpu()) 
         for result in rs.results]
            for rs in results
             ] for key in acquisition_keys}
    # Log some "meta" data
    output |= {MetricTemplate(name=MetricName.ACQUIRED_CLASS_DISTRIBUTION) : [
        [result.acquired_class_distribution.cpu() for result in rs.results]
        for rs in results
        ]}
    output |= {'acquisition_step' : [
        [result.acquisition_step for result in rs.results]
        for rs in results
        ]}
    output |= {'acquired_idxs' : [
        [result.acquired_idxs for result in rs.results]
        for rs in results
        ]}
    return output
            
@typechecked           
def save_results(results: List[Results], outdir: Path) -> PathLike:
    """ Saves all results to a pytorch file. In contrast to the AcquisitionCurve, it does not require
    the metrics to be homogenous. """
    output = aggregate_results(results)
    # Save the entire result (will be synced later on `wandb.finish`)
    outfile = outdir / 'acquisition_curve_metrics.pt' # type: ignore
    get_logger().info(f'Saved results to {outfile}')
    torch.save({str(k) : v for k, v in output.items()}, outfile)
    return outfile