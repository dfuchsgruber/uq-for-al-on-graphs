from graph_al.data.base import BaseDataset
from graph_al.utils.data import *
from graph_al.data.config import TorchGeometricDataConfig, TorchGeometricDatasetType
import torch_geometric.datasets as D
from torch_geometric.transforms import LargestConnectedComponents, Compose, ToUndirected

class TorchGeometricDataset(BaseDataset):
    
    """ Dataset loaded  from pytorch geometrics dataset library. """

    def __init__(self, config: TorchGeometricDataConfig):
        
        transforms = []
        if config.undirected:
            transforms.append(ToUndirected())
        if config.largest_connected_component:
            transforms.append(LargestConnectedComponents())
        transform = Compose(transforms)
        match config.torch_geometric_dataset:
            case TorchGeometricDatasetType.CORA_ML:
                tg_dataset = D.CitationFull(config.root, 'Cora_ML', transform=transform)[0]
            case TorchGeometricDatasetType.CITESEER:
                tg_dataset = D.CitationFull(config.root, 'CiteSeer', transform=transform)[0]
            case TorchGeometricDatasetType.PUBMED:
                tg_dataset = D.CitationFull(config.root, 'PubMed', transform=transform)[0]
            case TorchGeometricDatasetType.AMAZON_COMPUTERS:
                tg_dataset = D.Amazon(config.root, 'computers', transform=transform)[0]
            case TorchGeometricDatasetType.AMAZON_PHOTOS:
                tg_dataset = D.Amazon(config.root, 'photo', transform=transform)[0]
            case TorchGeometricDatasetType.REDDIT:
                tg_dataset = D.Reddit(root=config.root, transform=transform)[0]
            case TorchGeometricDatasetType.OGBN_ARXIV:
                raise NotImplementedError('We have not yet implemented ogbn-arxiv. What would be the split-pool?')
            case _:
                raise ValueError(f'Unsupported torch geometric dataset {config.torch_geometric_dataset}')
        
        x = tg_dataset.x # type: ignore
        y = tg_dataset.y # type: ignore
        edge_index = tg_dataset.edge_index # type: ignore
        
        node_to_idx = {f'node_{idx}' : idx for idx in range(x.size(0))}
        label_to_idx = {f'label_{idx}' : idx for idx in range(y.max().item() + 1)}
        feature_to_idx = {f'feature_{idx}' : idx for idx in range(x.size(1))}

        super().__init__(
            node_features = x,
            edge_idxs = edge_index,
            labels = y,
            node_to_idx = make_mapping_collatable(node_to_idx),
            label_to_idx = make_mapping_collatable(label_to_idx),
            feature_to_idx = make_mapping_collatable(feature_to_idx),
        )
        
        
        
        