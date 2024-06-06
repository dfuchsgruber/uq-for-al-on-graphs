from typing import Callable, Dict
import wandb
import torch
from torch import Tensor
from torch.utils.data import Dataset as TorchDataset
from torch_geometric.data import Data as TorchGeometricData
import torch_geometric.nn as tgnn
import torch_scatter
from copy import copy, deepcopy
import itertools

from graph_al.data.config import DataConfig, DatasetSplit
from graph_al.utils.sampling import sample_from_mask
from graph_al.utils.ppr import approximate_ppr_matrix, approximate_ppr_scores
from graph_al.utils.logging import get_logger
from graph_al.data.transform import normalize_features
from graph_al.data.enum import *

from jaxtyping import Float, Int, Bool, jaxtyped
from typeguard import typechecked

class BaseDataset:
    """ Base class for datasets. Additionally, you can provide pre-defined masks for training, validation and testing. If those are not provided,
        Each dataset split will sample them freshly, with however the test set fixed among all splits.
    
        Args:
            node_features (Float[Tensor, &#39;num_nodes num_node_features&#39;]): Node features
            labels (Int[Tensor, &#39;num_nodes&#39;]): Node labels
            edge_idxs (Int[Tensor, &#39;2 num_edges&#39;]): Edge indices
            num_classes (int | None, optional): Number of classes. Defaults to None, i.e. inferred from `labels`
            mask_train (Bool[Tensor, &#39;num_nodes&#39;] | None, optional): All vertices that are available to training. If `None`, use all non-val or -test nodes
            mask_val (Bool[Tensor, &#39;num_nodes&#39;] | None, optional): _description_. If `None`, use all non-val nodes
            mask_test (Bool[Tensor, &#39;num_nodes&#39;] | None, optional): _description_. If `None`, use a (consistent) test split
        """
    
    @jaxtyped(typechecker=typechecked)
    def __init__(self, node_features: Float[Tensor, 'num_nodes num_node_features'],
                 labels: Int[Tensor, 'num_nodes'],
                 edge_idxs: Int[Tensor, '2 num_edges'], num_classes: int | None = None,
                 mask_train: Bool[Tensor, 'num_nodes'] | None = None,
                 mask_val: Bool[Tensor, 'num_nodes'] | None = None,
                 mask_test: Bool[Tensor, 'num_nodes'] | None = None,
                 node_to_idx: Dict[str, int] | None = None,
                 label_to_idx : Dict[str, int] | None = None,
                 feature_to_idx: Dict[str, int] | None = None,
                 ):
        self.node_features = node_features
        self.edge_idxs = edge_idxs
        self.labels = labels
        if num_classes is None:
            self.num_classes = int(self.labels.max().item() + 1)
        else:
            self.num_classes = num_classes
        self.mask_val = mask_val
        self.mask_test = mask_test
        self.node_to_idx = node_to_idx
        self.label_to_idx = label_to_idx
        self.feature_to_idx = feature_to_idx
        
        node_degrees_in, node_degrees_out = self.node_degrees_in, self.node_degrees_out
        
        if wandb.run is not None:
            wandb.run.config.update({
                    'data.average_in_degree' : node_degrees_in.mean().item(),
                    'data.average_out_degree' : node_degrees_out.mean().item(),
                }, allow_val_change=True)
        get_logger().info(f'Average node degrees of dataset: out={node_degrees_out.mean().item():.6f}, in={node_degrees_in.mean().item():.6f}')
        
    @property
    def node_degrees_in(self) -> Int[Tensor, 'num_nodes']:
        return torch_scatter.scatter_add(torch.ones(self.edge_idxs.size(1)), self.edge_idxs[1])
        
    @property
    def node_degrees_out(self) -> Int[Tensor, 'num_nodes']:
        return torch_scatter.scatter_add(torch.ones(self.edge_idxs.size(1)), self.edge_idxs[0])
        
    @property
    def num_nodes(self) -> int:
        return self.node_features.size(0)
    
    @property
    def num_edges(self) -> int:
        return self.edge_idxs.size(1)
    
    @property
    def num_input_features(self) -> int:
        return self.node_features.size(1)
    
    @property
    def has_multiple_splits(self) -> bool:
        return self.mask_val is None
    
    
class Data(TorchGeometricData):
    """ A data instance. """

    @jaxtyped(typechecker=typechecked)
    def add_to_train_idxs(self, idxs: Int[Tensor, 'num_acquired']):
        """ Adds new indices to training indices. """
        assert not (self.mask_train[idxs]).any(), f'Some of the indices that are to be added were already acquired'
        assert (self.mask_train_pool[idxs]).all(), f'Some of the indices that are to be added were not in the train pool'
        self.mask_train_pool[idxs] = False
        self.mask_train[idxs] = True

    @property
    def node_degrees_in(self) -> Int[Tensor, 'num_nodes']:
        return torch_scatter.scatter_add(torch.ones(self.edge_index.size(1), device=self.edge_index.device), self.edge_index[1])
        
    @property
    def node_degrees_out(self) -> Int[Tensor, 'num_nodes']:
        return torch_scatter.scatter_add(torch.ones(self.edge_index.size(1), device=self.edge_index.device), self.edge_index[0])

    @property
    def num_train(self) -> int:
        return int(self.mask_train.sum().item())
    
    @property
    @jaxtyped(typechecker=typechecked)
    def class_counts(self) -> Int[Tensor, 'num_classes']:
        return torch_scatter.scatter_add(torch.ones_like(self.y), self.y, dim_size=self.num_classes)
    
    @property
    @jaxtyped(typechecker=typechecked)
    def class_counts_train(self) -> Int[Tensor, 'num_classes']:
        y = self.y[self.mask_train]
        return torch_scatter.scatter_add(torch.ones_like(y), y, dim_size=self.num_classes)
    
    @property
    @jaxtyped(typechecker=typechecked)
    def class_prior_probabilities_train(self) -> Float[Tensor, 'num_classes']:
        counts = self.class_counts_train
        if counts.sum() == 0:
            counts = torch.ones(self.num_classes, device=counts.device, dtype=torch.float)
        else:
            counts = counts.float()
        return counts / (counts.sum() + 1e-12)
    
    @property
    def masks_valid(self) -> bool:
        """If the masks are valid

        Returns:
            bool: if masks are valid
        """
        if not ((self.get_mask(DatasetSplit.TRAIN).long() + self.get_mask(DatasetSplit.VAL).long() + \
                self.get_mask(DatasetSplit.TEST).long() + self.get_mask(DatasetSplit.TRAIN_POOL).long()) == 1).all().item():
            return False
        return True

    def print_masks(self):
        """ For debugging: Print ratios of the masks. """
        for split in DatasetSplit:
            print(split, self.get_mask(split).float().mean().item())
    

    @jaxtyped(typechecker=typechecked)
    def get_mask(self, which: DatasetSplit) -> Bool[Tensor, 'num_nodes']:
        match which:
            case DatasetSplit.TRAIN:
                return self.mask_train
            case DatasetSplit.VAL:
                return self.mask_val
            case DatasetSplit.TEST:
                return self.mask_test
            case DatasetSplit.TRAIN_POOL:
                return self.mask_train_pool
            case DatasetSplit.ALL:
                return torch.ones_like(self.mask_test, dtype=torch.bool)
            case DatasetSplit.UNLABELED:
                return ~self.mask_train
            case DatasetSplit.UNLABELED_MINUS_TEST:
                return ~(self.mask_train | self.mask_test)
            case _:
                raise ValueError(which)

    @jaxtyped(typechecker=typechecked)
    def delete_mask_from_train_and_train_pool(self, mask: Bool[Tensor, 'num_nodes'], verbose: bool=True, add_deleted_to_val: bool = True):
        """Deletes a mask from the training (pool)

        Args:
            mask (Bool[Tensor, &#39;num_nodes&#39;]): the mask to delete
            verbose (bool, optional): whether to verbose about where indices are deleted. Defaults to True.
            add_deleted_to_val (bool, optional): whether the deleted indices should be moved to the validation set. Defaults to True.
        """
        if verbose:
            match bool((mask & self.get_mask(DatasetSplit.TRAIN)).any()), bool((mask & self.get_mask(DatasetSplit.TRAIN_POOL)).any()):
                case True, True:
                    get_logger().info('Deleting indices from training and training pool masks.')
                case True, _:
                    get_logger().info('Deleting indices from training mask.')
                case _, True:
                    get_logger().info('Deleting indices from training mask.')
        mask_overlap = (self.get_mask(DatasetSplit.TRAIN) | self.get_mask(DatasetSplit.TRAIN_POOL)) & mask
        if add_deleted_to_val:
            self.mask_val |= mask_overlap
        self.mask_train &= ~mask
        self.mask_train_pool &= ~mask
        assert self.masks_valid, f'After deleting masked nodes from training set and training pool led to invalid masks'


    @jaxtyped(typechecker=typechecked)
    def get_diffused_nodes_features(self, k: int, normalize: bool = True, 
        improved: bool = False, add_self_loops: bool = True, cache: bool = True) -> Float[Tensor, 'num_nodes num_features']:
        """Gets (and caches) nodes features after GCN convolutions.
        """
        key = f'diffused_node_features_{k}_{normalize}_{improved}_{add_self_loops}'
        
        diffused = getattr(self, key, None) if cache else None
        if diffused is not None:
            return diffused # type: ignore
        # Compute the diffused node features

        edge_index, edge_weight = self.edge_index, getattr(self, 'edge_weight', None)
        x = self.x
        if normalize:
            edge_index, edge_weight = tgnn.conv.gcn_conv.gcn_norm(  # yapf: disable
                edge_index, edge_weight, x.size(0),
                improved=improved, add_self_loops=add_self_loops)
        elif edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)

        src, dst = edge_index
        for _ in range(k):
            # Perform message passing
            messages = x[src] * edge_weight[:, None] # type: ignore
            x = torch_scatter.scatter_add(messages, dst, dim=0, dim_size=x.size(0))
        if cache:
            setattr(self, key, x)
        return x

    @property
    def stochastic_adjacency_edge_weights(self) -> Float[Tensor, 'num_edges']:
        """ Gets the weight of the stochastic normalized adjacency matrix. """
        edge_weights = torch.ones(self.edge_index.size(1), device=self.edge_index.device)
        degree = torch_scatter.scatter_add(edge_weights, self.edge_index[0], dim_size=self.num_nodes)

        edge_weights /= (degree[self.edge_index[0]] + 1e-12)
        return edge_weights

    def get_appr_scores(self, teleport_probability: float=0.2, num_iterations: int = 10, cache: bool = True) -> Float[Tensor, 'num_nodes']:
        """ Lazily computes approximate personalized page rank centralities for all nodes
        
        Args:
            teleport_probability (float): Teleport probability
            num_iterations (int): For how many iterations to do power iteration
        
        Returns:
            Float[Tensor, 'num_nodes']: The approximate PPR score for each node
        """
        
        key = f'appr_scores_{teleport_probability:.4f}_{num_iterations}'
        appr_scores = getattr(self, key, None) if cache else None
        if appr_scores is not None:
            return appr_scores # type: ignore
        
        appr_scores = approximate_ppr_scores(self.edge_index, self.stochastic_adjacency_edge_weights,
                                                            teleport_probability=teleport_probability, num_iterations=num_iterations,
                                                            verbose=True, num_nodes=self.num_nodes).cpu()
        if cache:
            setattr(self, key, appr_scores)
        return appr_scores
        
    @jaxtyped(typechecker=typechecked)
    def log_appr_matrix(self, teleport_probability: float=0.2, num_iterations: int = 10, cache: bool = True) -> Float[Tensor, 'num_nodes num_nodes']:
        """ Lazily computes approximate log personalized page rank scores between all node pairs
        
        Args:
            teleport_probability (float): Teleport probability
            num_iterations (int): For how many iterations to do power iteration
        
        Returns:
            Float[Tensor, 'num_nodes num_nodes']: The log APPR matrix log(Pi), where Pi_ij is the importance of node i to node j
        """
        key = f'log_appr_matrix_{teleport_probability:.4f}_{num_iterations}'
        log_appr_matrix = getattr(self, key, None) if cache else None
        if log_appr_matrix is not None:
            return log_appr_matrix # type: ignore
        
        log_appr_matrix = approximate_ppr_matrix(self.edge_index, self.stochastic_adjacency_edge_weights,
                                                            teleport_probability=teleport_probability, num_iterations=num_iterations,
                                                            verbose=True, num_nodes=self.num_nodes).cpu().log()
        if cache:
            setattr(self, key, log_appr_matrix)
        return log_appr_matrix

    @jaxtyped(typechecker=typechecked)
    def is_pseudo_labeled(self, 
                          probabilities: Float[Tensor, 'num_nodes num_classes'], delta: float = 0.6) -> Bool[Tensor, 'num_nodes']:
        """ Whether a node is pseudo labeled by the SEAL model, i.e. in L+"""
        confidence = probabilities.max(1).values
        pseudo_labels = confidence > delta
        pseudo_labels[self.get_mask(DatasetSplit.TRAIN)] = True
        return pseudo_labels
    

class Dataset(TorchDataset):
    
    """ A wrapper around base datasets that is responsible for splitting. It also keeps
    track of the state of the sampled nodes. """
    
    def __init__(self, config: DataConfig, base: BaseDataset):
        self.base = base
        self.val_size = config.val_size
        self.normalize = config.normalize

        self.setting = config.setting
        assert self.setting == GraphSetting.TRANSDUCTIVE, f'Non transductive settings are not supported at the moment'
        
        # Setup the test mask
        if self.base.mask_test is not None:
            mask_test = self.base.mask_test
        elif config.test_size == 0:
            mask_test = torch.zeros(self.base.num_nodes, dtype=torch.bool)
        else:
            rng_test = torch.Generator()
            rng_test.manual_seed(0x8D2CBCFC3A) # A random test seed to keep test splits fixed among different calls of `self.split`
            mask_test = torch.zeros(self.base.num_nodes, dtype=torch.bool)
            for y in range(self.base.num_classes):
                mask_test |= sample_from_mask(self.base.labels == y, config.test_size, generator=rng_test)
        
        # The base data instance is directly derived from `self.base` and serves as a "template"
        self.base_data = self.transform(Data(
            x = self.base.node_features,
            y = self.base.labels,
            edge_index = self.base.edge_idxs,
            mask_test = mask_test,
            num_classes=self.base.num_classes,
        ))
        self.data = self.base_data.clone()
    
    def print_masks(self):
        self.data.print_masks()

    def transform(self, data: Data) -> Data:
        data.x = normalize_features(data.x, self.normalize, dim=-1) # type: ignore
        return data

    @property
    def masks_valid(self) -> bool:
        return self.data.masks_valid

    def reset_train_idxs(self):
        """ Resets the train idxs to the empty set and the train pool to the remaining indices. """
        self.data.mask_train = torch.zeros_like(self.data.mask_test)
        self.data.mask_train_pool = ~(self.data.mask_test | self.data.mask_val)
        assert self.masks_valid

    def split(self, generator: torch.Generator | None=None, mask_not_in_val: Bool[Tensor, 'num_nodes'] | None = None):
        """ Resets the training mask and applies a new dataset split if the dataset permits it. 
        
        Args:
            generator (torch.Generator): A generator for getting the split
            mask_not_in_val (Bool[Tensor, 'num_nodes'] | None): If given, some nodes that can never be in the validation set
        """
        self.data.mask_train = torch.zeros_like(self.data.mask_test)
        
        if self.base.mask_val is not None:
            self.data.mask_val = self.base.mask_val.to(self.data.mask_test.device)
        elif self.val_size == 0:
            self.data.mask_val = torch.zeros_like(self.data.mask_train)
        else:
            self.data.mask_val = torch.zeros_like(self.data.mask_train, dtype=torch.bool)
            for y in range(self.base.num_classes):
                if isinstance(self.val_size, float):
                    val_size = int((self.data.y == y).sum().item() * self.val_size)
                elif isinstance(self.val_size, int):
                    val_size = self.val_size
                else:
                    raise ValueError(f'Validation set size must be int or float, but got {type(self.val_size)}')
                
                mask_val_pool = (self.data.y == y) & (~self.data.mask_test)
                if mask_not_in_val is not None:
                    mask_val_pool &= (~(mask_not_in_val)).to(mask_val_pool.device)
                self.data.mask_val |= sample_from_mask(mask_val_pool, val_size, generator=generator)
        if hasattr(self.data, 'mask_ood'):
            del self.data.mask_ood
        self.reset_train_idxs()

    def cuda(self) -> 'Dataset':
        self.data = self.data.cuda()
        return self
    
    def cpu(self) -> 'Dataset':
        self.data = self.data.cpu()
        return self
        
    @jaxtyped(typechecker=typechecked)
    def add_to_train_idxs(self, idxs: Int[Tensor, 'num_acquired']):
        """ Adds new indices to training indices. """
        self.data.add_to_train_idxs(idxs)
        
    def __getitem__(self, idx: int) -> Data:
        if idx >= len(self):
            raise IndexError(f'Trying to index {self} of size {len(self)} with index {idx}')
        return self.data
    
    def __len__(self) -> int:
        return 1
    
    @property
    def num_nodes(self) -> int:
        assert self.data.num_nodes is not None, 'Data object has no nodes'
        return self.data.num_nodes
    
    @property
    def num_edges(self) -> int:
        return self.data.num_edges
    
    @property
    def num_input_features(self) -> int:
        return self.data.num_node_features
    
    @property
    def num_train_nodes(self) -> int:
        return self.data.num_train
    
    @property
    def num_classes(self) -> int:
        return int(self.data.num_classes)
    
    @property
    def has_multiple_splits(self) -> bool:
        return self.base.has_multiple_splits
        
    @property
    def node_degrees_in(self) -> Int[Tensor, 'num_nodes']:
        return self.data.node_degrees_in
        
    @property
    def node_degrees_out(self) -> Int[Tensor, 'num_nodes']:
        return self.data.node_degrees_out
        
        
        