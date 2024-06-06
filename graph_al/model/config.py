from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf, DictConfig

from typing import List, Any, Sequence

from graph_al.model.trainer.config import TrainerConfig
from graph_al.model.enum import *

@dataclass
class ModelConfig:
    
    type_: ModelType = MISSING
    num_inits: int = 1
    init_number: int | None = None
    name: str | None = None
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    
    num_ensemble_members: int | None = None # if > 1, an ensemble of this model will be created

@dataclass
class ModelConfigMultipleSamples(ModelConfig):
    """ Configuration to support drawing from multiple samples. """
    num_samples_train: int = 1
    num_samples_eval: int = 1
    # Instead of sequentially running the model multiple times, we can also
    # collate the same graph into a large batch of disconnected graphs
    # this may run into memory issues for larger graphs though
    # also, the speed improvement is not that significant...
    collate_samples: bool = False 

@dataclass
class ModelConfigMonteCarloDropout(ModelConfigMultipleSamples):
    """ Configuration for models that sample using MC dropout. """
    dropout_at_eval: bool = False
    dropout: float = 0.0

@dataclass
class BaseGCNConfig(ModelConfig):
    """ Base config class for GCN-based architectures """
    hidden_dims: List[int] = field(default_factory=lambda: [64,])
    improved: bool = False # If True, sets self-weight to 2
    cached: bool = True
    add_self_loops: bool = True
    inplace: bool = True # Performs ReLU, dropout in-place
    
@dataclass
class BayesianGCNConfig(ModelConfigMultipleSamples, BaseGCNConfig):
    """ Configuration for a Bayesian GCN """
    name: str | None = 'bgcn'
    
    type_: ModelType = ModelType.BAYESIAN_GCN
    
    rho_init: float = -3.0 # initial value for the rho parameter of the variance

@dataclass
class GCNConfig(ModelConfigMonteCarloDropout, BaseGCNConfig):
    """ Configuration for the GCN """ 
    type_: ModelType = ModelType.GCN
    name: str | None = 'gcn'
    
@dataclass
class APPNPConfig(ModelConfigMonteCarloDropout):
    """ Configuration for the GCN """ 
    
    type_: ModelType = ModelType.APPNP
    name: str | None = 'appnp'
    hidden_dims: List[int] = field(default_factory=lambda: [64,])
    dropout: float = 0.0
    cached: bool = True
    add_self_loops: bool = True
    k: int = 10
    alpha: float = 0.2
    inplace: bool = True # Performs ReLU, dropout in-place
     

@dataclass
class GPNConfig(APPNPConfig):
    """ Configuration for the Graph Posterior Network """
    
    type_: ModelType = ModelType.GPN
    name: str | None = 'gpn'
    dropout: float = 0.5
    batch_norm: bool = False
    
    flow_dim: int = 64
    num_flow_layers: int = 10
    evidence_scale: GPNEvidenceScale = GPNEvidenceScale.LATENT_NEW
    

@dataclass
class BayesianLikelihoodConfig:
    use_features: bool = True
    use_adjacency: bool = True
    use_non_edges: bool = True

@dataclass
class BayesOptimalConfig(ModelConfig):
    type_: ModelType = ModelType.BAYES_OPTIMAL
    approximation_type: ApproximationType = ApproximationType.VARIATIONAL_INFERENCE
    num_samples: int = 100
    importance_prob: float = 0.9
    normalized: bool = True
    cached: bool = True
    
    # Beam search parameters
    beam_size: int = 1000
    beam_search_num_steps: int = 100
    beam_fraction_random_assignments: float = 0.0 # How much of the beam is randomized after step
    beam_search_restarts_every: int = 0 # How often the beam search is re-initialized to increase diversity
    
    # Variational inference
    variational_inference_max_num_steps: int = 1000
    variational_inference_convergence_tolerance: float = 1e-4
    
    predict_train: bool = True # if set, predicts with 100% confidence the true label for train nodes
    prediction: BayesianPrediction = BayesianPrediction.MARGINAL
    
    # To approximate p(y_i = c | A, Y_{fixed}), use a consistent set of samples for all classes of one node
    # this maybe reduces variance (at the cost of mathematical soundness)
    total_confidence_fix_samples_per_node: bool = True
    multiprocessing: bool = True
    num_workers: int | None = None # How many worker processes to spawn for approximating the total confidence
    batch_size: int = 100_000 # How many samples / labelings to process in one numpy array
    
    # debugging stuff 
    verbose: bool = False
    confidence_likelihood: BayesianLikelihoodConfig = field(default_factory=BayesianLikelihoodConfig)
    prediction_likelihood: BayesianLikelihoodConfig = field(default_factory=BayesianLikelihoodConfig)

    # if verbose, also computes the approximation error of the total uncertainty
    # by computing the exact value by brute-force and computing the difference
    # note that this is very costly and should only be used to debug approximation strategies
    compute_approximation_error: bool = False 
    
    likelihood_cache_in_memory: bool = True
    
@dataclass
class SGCConfig(BaseGCNConfig):
    type_: ModelType = ModelType.SGC
    k: int = 2
    inverse_regularization_strength: float = 1.0
    solver: LogisticRegressionSolver = LogisticRegressionSolver.LIBLINEAR
    balanced: bool = True
    
@dataclass
class SEALConfig(GCNConfig):
    type_: ModelType = ModelType.SEAL
    alpha: float = 0.6
    hidden_dims: List[int] = field(default_factory=lambda: [16,])
    hidden_dims_discriminator: List[int] = field(default_factory=lambda: [128, 128,])
    discriminator_dropout: float = 0.5
    delta: float = 0.6 # threshold for pseudo-labels: all nodes are added to a pseudo labeled or unlabeled set
    
cs = ConfigStore.instance()
cs.store(name="base_config", node=ModelConfig, group='model')
cs.store(name="base_gcn", node=GCNConfig, group='model')
cs.store(name="base_appnp", node=APPNPConfig, group='model')
cs.store(name="base_bayesian_gcn", node=BayesianGCNConfig, group='model')
cs.store(name="base_gpn", node=GPNConfig, group='model')
cs.store(name="base_bayes_optimal", node=BayesOptimalConfig, group='model')
cs.store(name="base_sgc", node=SGCConfig, group='model')
cs.store(name="base_seal", node=SEALConfig, group='model')