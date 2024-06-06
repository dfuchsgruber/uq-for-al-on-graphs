from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from typing import List, Any
from enum import StrEnum, unique

from graph_al.data.enum import *

@dataclass
class DataConfig:
    
    type_: DatasetType = MISSING
    num_classes: int = MISSING
    name: str | None = None
    seed: int | None = None
    
    num_splits: int = 1
    split_number: int | None = None
    
    val_size: int | float = 0.2 # How many nodes (per class) to use as validation
    test_size: int | float = 0.2 # How many nodes (per class) to use as test
    
    
    normalize: FeatureNormalization = FeatureNormalization.NONE
    setting: GraphSetting = GraphSetting.TRANSDUCTIVE

@dataclass
class GenerativeDataConfig(DataConfig):
    """ Configuration for datasets with explicit generative process. """
    likelihood_cache_database_path: str | None = None # path for where likelihood database is cached
    likelihood_cache_database_lockfile_path: str | None = None # path for where likelihood database lockfile is stored
    likelihood_cache_storage_path: str | None = None # path for where the cached likelihoods are stored

@dataclass
class SBMConfig(GenerativeDataConfig):
    """ Configuration for the SBM """ 
    
    num_classes: int = 3
    num_nodes: Any = 90
    class_prior: List[float] | None = None # None means uniform
    affiliation_matrix: List[List[float]] | None = None
    directed: bool = False
    
    feature_dim: int = 2
    feature_sigma: float = 0.1
    
@dataclass
class RandomSBMConfig(SBMConfig):
    """ Configuration for a randomly sampled SBM """
    type_: DatasetType = DatasetType.RANDOM_SBM

    seed: int | None = None # seed specific for the graph sampled from the SBM distribution, if None is given it is randomly sampled from the dataset seed
    
    feature_class_mean_sigma: float = 1.0
    feature_class_mean_distance: float = 0.1
    feature_class_mean_sampling: SBMClassMeanSampling = SBMClassMeanSampling.EQUIDISTANT
    
    edge_probabilities: SBMEdgeProbabilitiesType = SBMEdgeProbabilitiesType.BY_SNR_AND_EXPECTED_DEGREE
    edge_probability_inter_community: float = 0.01
    edge_probability_intra_community: float = 0.1
    edge_probability_snr: float = 10.0 # p_inter / p_intra
    expected_degree: float | None = None
    
@dataclass
class DeterministicSBMConfig(SBMConfig):
    """ Configuration for a SBM with fixed node features and labels. """
    type_: DatasetType = DatasetType.DETERMINISTIC_SBM
    
    node_features: List[List[float]] = field(default_factory=lambda: [[]])
    labels: List[int] = field(default_factory=list)
    edge_index: List[List[int]] = field(default_factory=lambda: [[]])
    class_means: List[List[float]] = field(default_factory=lambda: [[]])
    
@dataclass
class NpzConfig(DataConfig):
    """ Configuration for a dataset loaded from a .npz file """
    
    path: str = MISSING
    type_: DatasetType = DatasetType.NPZ
    preprocessing: NpzFeaturePreprocessing = NpzFeaturePreprocessing.NONE
    vectorizer: NpzFeatureVectorizer = NpzFeatureVectorizer.TF_IFD
    min_token_frequency: int = 10
    normalize: FeatureNormalization = FeatureNormalization.L2
    
    use_public_split: bool = False # If to use a publicly avialable data split
    
@dataclass
class TorchGeometricDataConfig(DataConfig):
    """ Configuration for datasets from pytorch geometric. """
    
    type_: DatasetType = DatasetType.TORCH_GEOMETRIC
    root: str = 'data'
    torch_geometric_dataset: TorchGeometricDatasetType = MISSING

    largest_connected_component: bool = True
    undirected: bool = True
    
cs = ConfigStore.instance()
cs.store(name="base_config", node=DataConfig, group='data')
cs.store(name="base_sbm", node=SBMConfig, group='data')
cs.store(name="base_random_sbm", node=RandomSBMConfig, group='data')
cs.store(name="base_deterministic_sbm", node=DeterministicSBMConfig, group='data')
cs.store(name="base_npz", node=NpzConfig, group='data')
cs.store(name="base_torch_geometric", node=TorchGeometricDataConfig, group='data')


