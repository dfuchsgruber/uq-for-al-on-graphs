from enum import unique, StrEnum

@unique
class TrainerType(StrEnum):
    SGD = 'sgd'
    SGC = 'SGC'
    GPN = 'gpn'
    ORACLE = 'oracle'
    SEAL = 'seal'

@unique
class LossFunction(StrEnum):
    CROSS_ENTROPY = 'cross_entropy'
    CROSS_ENTROPY_AND_KL_DIVERGENCE = 'cross_entropy_and_kl_divergence'
    NONE = 'none'
    
@unique
class GPNWarmup(StrEnum):
    FLOW = 'flow'
    ENCODER = 'encoder'