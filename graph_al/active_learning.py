import torch
from graph_al.acquisition.base import BaseAcquisitionStrategy

from graph_al.config import Config
from graph_al.data.base import Dataset
from graph_al.model.base import Ensemble, BaseModel
from graph_al.model.trainer.config import TrainerConfig
from graph_al.utils.logging import get_logger
from graph_al.model.trainer.build import get_trainer
from graph_al.acquisition.build import get_acquisition_strategy
from graph_al.evaluation.result import Result
from graph_al.model.trainer.evaluation.evaluate import evaluate

from jaxtyping import jaxtyped, Int
from torch import Tensor

def initial_acquisition(acquisition_strategy: BaseAcquisitionStrategy, config: Config, model: BaseModel, dataset: Dataset, 
        generator: torch.Generator) -> Int[Tensor, 'num_acquired']:
    """ Initially acquires some nodes before model training. It is recommended to use a model-free acquisition strategy for this
    e.g. balanced. """
    
    all_aquired_idxs = []
    acquisition_strategy.reset()
    for acquisition_step in range(1, 1 + config.initial_acquisition_strategy.num_steps):
        model = model.eval()
        with torch.no_grad():
            acquired_idxs, _ = acquisition_strategy.acquire(model, dataset, config.initial_acquisition_strategy.num_to_acquire_per_step, config.model.trainer, generator)
        dataset.add_to_train_idxs(acquired_idxs)
        all_aquired_idxs.append(acquired_idxs)
    get_logger().info(f'Starting run with initial class counts {dataset.data.class_counts_train.tolist()}')
    if len(all_aquired_idxs) > 0:
        return torch.cat(all_aquired_idxs)
    else:
        return torch.tensor([], dtype=torch.int)

def train_model(config: TrainerConfig, model: BaseModel, dataset: Dataset, generator: torch.Generator, acquisition_step: int=-1) -> Result:
    """ Trains the model according to a training configuration. If no instances are in the train mask, the current model
    is simply evaluated.

    Args:
        config (TrainerConfig): the config according to which to train
        model (BaseModel): the model to train
        dataset (Dataset): the data on which to train
        generator (torch.Generator): a rng
        acquisition_step (int, optional): in which acquisition step to train. Defaults to -1.

    Returns:
        Result: the result
    """
    if dataset.data.num_train == 0:
        trainer = get_trainer(config, model, dataset, generator)
    elif isinstance(model, Ensemble):
        for _member in model.models:
            member: BaseModel = _member # type: ignore
            trainer = get_trainer(config, member, dataset, generator)
            trainer.fit(member, dataset, generator, acquisition_step)
        trainer = get_trainer(config, model, dataset, generator)
    else:
        trainer = get_trainer(config, model, dataset, generator)
        trainer.fit(model, dataset, generator, acquisition_step)
    # Run evaluation after fitting
    metrics = evaluate(config.evaluation, model, trainer, dataset, generator)
    return Result(metrics, dataset.data.class_counts_train.cpu(), acquisition_step=acquisition_step)