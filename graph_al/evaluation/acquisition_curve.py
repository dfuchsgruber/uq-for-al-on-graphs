from graph_al.evaluation.enum import MetricName, MetricTemplate
from graph_al.evaluation.result import Results
from graph_al.utils.logging import get_logger

import torch
from dataclasses import dataclass

from jaxtyping import Shaped, Int, Float
from typeguard import typechecked
from typing import List, Dict, Literal, Iterable, Tuple
from torch import Tensor
import matplotlib.pyplot as plt

import os
import wandb

@dataclass
class AcquisitionCurve:
    """ Class for a curve of values that evolves over acquisition. """
    
    values: Dict[MetricTemplate, Shaped[Tensor, 'num_runs num_acquisition_steps']]
    num_acquired: Int[Tensor, 'num_acquisition_steps']
    
    def _parse_keys(self, keys: Literal['all'] | Iterable[MetricTemplate]) -> Iterable[MetricTemplate]:
        if keys == 'all':
            keys = self.values.keys()
        return keys
    
    @typechecked
    def area_under_the_curve(self, keys: Literal['all'] | Iterable[MetricTemplate], normalized: bool=True
                             ) -> Dict[MetricTemplate, Tuple[Float[Tensor, 'num_runs'], Float[Tensor, 'num_runs']]]:
        """ Computes area under the acquisition curves. 

        Args:
            keys (Literal[&#39;all&#39;] | Iterable[str]): The values to calculate the area under
            normalized (bool, optional): Normalizes the x axis to [0, 1]. Defaults to True.

        Returns:
            Dict[MetricTemplate, Tuple[Float[Tensor, 'num_runs'], Float[Tensor, 'num_runs']]: Area under the curves for each run and the final value
        """
        keys = list(self._parse_keys(keys))
        order = torch.argsort(self.num_acquired) # should be sorted anyway, just be sure
        result = {} 
        for key in keys:
            values = self.values[key].float() # num_runs, num_acquisition_steps
            if normalized:
                x = self.num_acquired.float() - self.num_acquired.min()
                x /= x.max()
                y = values
            else:
                x, y = self.num_acquired, values
            x, y = x[order], y[:, order]
            areas, final_values = [], []
            for run_idx in range(y.size(0)):
                mask_finite = torch.isfinite(y[run_idx])
                if not mask_finite.all():
                    get_logger().warn(f'Metric {key} does have non-finite values in run {run_idx}, which will be ignored...')
                areas.append(torch.trapezoid(y[run_idx][mask_finite], x[mask_finite], dim=-1)) # num_runs
                final_values.append(y[run_idx][mask_finite][-1] if mask_finite.any() else torch.tensor(torch.nan))
            result[key] = torch.stack(areas), torch.stack(final_values)
        return result
    
    @property
    def num_runs(self) -> int:
        if len(self.values) == 0:
            raise RuntimeError(f'Can not infer the number of runs from no values')
        return self.values[list(self.values.keys())[0]].size(0)
    
    @typechecked
    def log(self, keys: Literal['all'] | Iterable[MetricTemplate]):
        """ Logs the values in the acquisition function with the number of acquired samples as step. """
        if wandb.run is None:
            return
        keys = list(self._parse_keys(keys))
        
        wandb.define_metric('num_acquired')
        for key in keys:
            wandb.define_metric(f'{key}/mean', step_metric='num_acquired')
            wandb.define_metric(f'{key}/std', step_metric='num_acquired')
        
        # Log means to wandb
        steps = self.num_acquired.tolist()
        for step_idx, step in enumerate(steps):
            metrics = {f'{key}/mean' : self.values[key][:, step_idx].float().mean() for key in keys} | \
                {f'{key}/std' : self.values[key][:, step_idx].float().std() for key in keys} 
            metrics['num_acquired'] = step
            if wandb.run is not None:
                wandb.run.log(metrics, commit=True)
    
    
    @typechecked
    def plot(self, keys: Literal['all'] | Iterable[MetricTemplate], name: str | None, figure_kwargs: Dict | None = None,
             legend: bool = True, x_label: str | None = None, y_label: str | None = None):
        """ Plots multiple acquisition curves in a seperate matplotlib instance and logs it to wandb . """
        keys = list(self._parse_keys(keys))
        
        if name is None:
            name = ', '.join(map(str, keys))
        if x_label is None:
            x_label = 'Num acquired nodes'
        
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        if len(color_cycle) < len(keys):
            get_logger().warn(f'Trying to plot {len(keys)} curves, but the color cycle has only {len(color_cycle)} '
                              'distinct colors. There will be duplicates.')
        
        if figure_kwargs is None:
            figure_kwargs = dict()
        fig, ax = plt.subplots(**figure_kwargs) # type: ignore
        ax: plt.Axes = ax 
        
        x = self.num_acquired.numpy()
        for key_idx, key in enumerate(keys):
            values = self.values[key].float() # num_runs, num_acquisition_steps
            mean = values.mean(0)
            std = values.std(0)
            color = color_cycle[key_idx % len(color_cycle)]
            ax.plot(x, mean.numpy(), label=key, c=color)
            ax.fill_between(x, (mean - std).numpy(), (mean + std).numpy(), color=color, alpha=0.1)
            
        if legend:
            fig.legend()
        if x_label:
            ax.set_xlabel(x_label)
        if y_label:
            ax.set_ylabel(y_label)
            
        ax.margins(0)
        
        if wandb.run is not None:
            wandb.run.log({f'acquisition_plots/{name}' : wandb.Image(fig)})
        plt.close(fig)
    
    @classmethod
    def from_results(cls, results: List[Results], keys: List[MetricTemplate]) -> 'AcquisitionCurve':
        values = {}
        for key in keys:
            value = torch.stack([result[key] if key in result else torch.tensor(torch.nan) for result in results])
            match len(value.size()):
                case 2: # num runs, num_acquisition_steps
                    values[key] = value
                case 3: # num runs, num_acquisition_steps, d
                    # TODO: one could nest this, to support > 3 dims
                    for idx in range(value.size(2)):
                        values[f'{key}[{idx}]'] = value[..., idx]
                case _:
                    raise RuntimeError(f'Acquisition curves are supposed to be built from 0d or 1d metrics collected after '
                                       f'model training but {key} is {len(value.size()) - 2}d')
        
        num_acquired = torch.stack([result[MetricTemplate(name=MetricName.NUM_ACQUIRED)] for result in results])
        # We require the number of acquired nodes to be exactly the same at each acquisition step over all
        # results (runs), otherwise we would need to bin, potentially a TODO
        assert (num_acquired == num_acquired[0]).all(), f'Currently it is not supported if different runs have variable number of acquired nodes'
        num_acquired = num_acquired[0]
        
        return AcquisitionCurve(values, num_acquired)
        
        
        
        


