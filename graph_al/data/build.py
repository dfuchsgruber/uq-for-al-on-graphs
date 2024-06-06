
from graph_al.data.config import DataConfig, NpzConfig, SBMConfig, TorchGeometricDataConfig, DeterministicSBMConfig, RandomSBMConfig
from graph_al.data.base import Dataset, BaseDataset

from graph_al.data.sbm import RandomSBMDataset, DeterministicSBMDataset
from graph_al.data.npz import NpzDataset
from graph_al.data.torch_geometric import TorchGeometricDataset

import torch

def get_base_dataset(config: DataConfig, generator: torch.Generator) -> BaseDataset:
    """ Builds the base dataset. """
    match config.type_:
        case RandomSBMConfig.type_:
            dataset = RandomSBMDataset(config, generator) # type: ignore
        case DeterministicSBMConfig.type_:
            dataset = DeterministicSBMDataset(config, generator) # type: ignore
        case NpzConfig.type_:
            dataset =  NpzDataset(config) # type: ignore
        case TorchGeometricDataConfig.type_:
            dataset =  TorchGeometricDataset(config) # type: ignore
        case _:
            raise ValueError(f'Unsupported base dataset type {config.type_}')
    assert dataset.num_classes == config.num_classes, f'Config specifies {config.num_classes} but dataset has {dataset.num_classes}'
    return dataset

def get_dataset(config: DataConfig, generator: torch.Generator) -> Dataset:
    """ Builds the base dataset. """
    base_dataset = get_base_dataset(config, generator)
    return Dataset(config, base_dataset)