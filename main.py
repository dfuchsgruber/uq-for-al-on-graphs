#!/usr/bin/env python

from omegaconf import OmegaConf, DictConfig
import hydra

from graph_al.config import Config
from graph_al.utils.exceptions import print_exceptions
from graph_al.utils.logging import print_table
from graph_al.utils.wandb import wandb_initialize
from graph_al.utils.logging import get_logger, print_config
from graph_al.utils.seed import set_seed
from graph_al.data.build import get_dataset
from graph_al.model.build import get_model
from graph_al.acquisition.build import get_acquisition_strategy
from graph_al.evaluation.active_learning import evaluate_active_learning, save_results
from graph_al.evaluation.result import Results
from graph_al.utils.wandb import wandb_get_metrics_dir
from graph_al.active_learning import initial_acquisition, train_model
from graph_al.acquisition.base import mask_not_in_val
from graph_al.data.enum import DatasetSplit
import wandb
import tqdm

import numpy as np # needed for using the eval resolver

OmegaConf.register_new_resolver("eval", lambda expression: eval(expression, globals(), locals()))

def setup_environment():
    import os
    import warnings
    os.environ['WANDB__SERVICE_WAIT'] = '300'

    # Stop pytorch-lightning from pestering us about things we already know
    warnings.filterwarnings(
        "ignore",
        "There is a wandb run already in progress",
        module="pytorch_lightning.loggers.wandb",
    )

    # Fixes a weird pl bug with dataloading and multiprocessing
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')
    

@hydra.main(config_path='config', config_name='main', version_base=None)
@print_exceptions
def main(config_dict: DictConfig) -> None:
    
    # delayed imports for a fast import of the main module
    
    import torch
    setup_environment()
    
    OmegaConf.resolve(config_dict)
    config: Config = hydra.utils.instantiate(config_dict, _convert_='object')
    rng = set_seed(config)
    generator = torch.random.manual_seed(rng.integers(2**31))
    get_logger().info(f'Big seed is {config.seed}')
    print_config(config) # type: ignore


    if not config.wandb.disable:
        wandb_initialize(config.wandb)
    outdir = wandb_get_metrics_dir(config)
    assert outdir is not None
    
    dataset = get_dataset(config.data, generator)
    model = get_model(config.model, dataset, generator)
    acquisition_strategy = get_acquisition_strategy(config.acquisition_strategy, model, dataset, generator)
    initial_acquisition_strategy = get_acquisition_strategy(config.initial_acquisition_strategy, model, dataset, generator)

    num_splits = config.data.num_splits
    if not dataset.has_multiple_splits and num_splits > 1:
        get_logger().warn(f'Dataset only supports one split, but requested {num_splits}. Only doing one.')
        num_splits = 1
    
    results = []
    for split_idx in range(num_splits):
        dataset.split(generator=generator, mask_not_in_val=mask_not_in_val(acquisition_strategy, initial_acquisition_strategy))

        for init_idx in range(config.model.num_inits):
            get_logger().info(f'Dataset split {split_idx}, Model initialization {init_idx}')
            acquisition_metrics_init = []

            model = get_model(config.model, dataset, generator)
            
            acquisition_step = 0
            acquisition_results = []
            dataset.reset_train_idxs()
            
            # 0. Initial aqcuisition: Usually randomly select nodes
            # If no nodes are selected, the model is also not trained
            # and the first actual acquisition uses an untrained model
            initial_train_idxs = initial_acquisition(initial_acquisition_strategy, config, model, dataset, generator)
            get_logger().info(f'Acquired the following initial pool: {initial_train_idxs.tolist()}')
            get_logger().info(f'Initial pool class counts: {dataset.data.class_counts_train.tolist()}')
            model.reset_cache()
            result = train_model(config.model.trainer, model, dataset, generator, acquisition_step=0)
            result.acquired_idxs = initial_train_idxs.cpu()
            acquisition_results.append(result)
                
            iterator = range(1, 1 + config.acquisition_strategy.num_steps)
            if config.progress_bar:
                iterator = tqdm.tqdm(iterator)
            acquisition_strategy.reset()
            for acquisition_step in iterator:
                model = model.eval()
                
                if dataset.data.mask_train_pool.sum().item() <= 0:
                    get_logger().info(f'Acquisition ends early because the entire pool was acquired: {dataset.data.class_counts_train.tolist()}')
                    break
                
                # 1. Aquire new instances
                with torch.no_grad():
                    acquired_idxs, acquisition_metrics = acquisition_strategy.acquire(model, dataset, config.acquisition_strategy.num_to_acquire_per_step, config.model, generator)
                acquisition_metrics_init.append(acquisition_metrics)
                dataset.add_to_train_idxs(acquired_idxs)
                # 2. Retrain the model
                if config.retrain_after_acquisition and acquisition_strategy.retrain_after_each_acquisition is not False:
                    model.reset_parameters(generator=generator)
                result = train_model(config.model.trainer, model, dataset, generator, acquisition_step=acquisition_step)
                # 3. Collect results
                result.acquired_idxs = acquired_idxs.cpu()
                acquisition_results.append(result)
                get_logger().info(f'Acquired node(s): {acquired_idxs.tolist()}')
                get_logger().info(f'Class counts after acquisition: {dataset.data.class_counts_train.tolist()}')
                
                if config.progress_bar:
                    message = f'Run {split_idx},{init_idx}: Num acquired: {dataset.data.num_train}, ' + ', '.join(f'{name} : {result.metrics[name]:.3f}' for name in config.progress_bar_metrics)
                    iterator.set_description(message) # type: ignore
                    
            # After the budget is exhausted
            run_results = Results(acquisition_results, dataset_split_num=split_idx, model_initialization_num=init_idx)
            results.append(run_results)
            # Checkpoint this model
            torch.save(model.state_dict(), outdir / f'model-{split_idx}-{init_idx}-{acquisition_step}.ckpt')
            torch.save({'mask_train' : dataset.data.get_mask(DatasetSplit.TRAIN).cpu(),
                        'mask_val' : dataset.data.get_mask(DatasetSplit.VAL).cpu(),
                        'mask_test' : dataset.data.get_mask(DatasetSplit.TEST).cpu(),
                        'mask_train_pool' : dataset.data.get_mask(DatasetSplit.TRAIN_POOL).cpu()}, outdir / f'masks-{split_idx}-{init_idx}-{acquisition_step}.ckpt')
            torch.save(acquisition_metrics_init, outdir / f'acquisition_metrics-{split_idx}-{init_idx}-{acquisition_step}.pt')
    
    summary_metrics = evaluate_active_learning(config.evaluation, results)
    if config.print_summary:
        print_table(summary_metrics, title='Summary over all splits and initializations')
    save_results(results, outdir)
    
    if wandb.run is not None:  
        wandb.run.log({}) # Ensures a final commit to the wandb server
        wandb.finish()
        
if __name__ == '__main__':
    main()