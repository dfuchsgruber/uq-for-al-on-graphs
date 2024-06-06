from enum import unique, StrEnum

@unique
class AcquisitionStrategyType(StrEnum):
    BY_PREDICTION_ATTRIBUTE = 'by_prediction_attribute'
    LOGIT_ENERGY = 'logit_energy'
    RANDOM = 'random'
    CORESET = 'coreset'
    GROUND_TRUTH = 'ground_truth'
    BEST_SPLIT = 'best_split'
    BEST_ORDERED_SPLIT = 'best_ordered_split'
    FIXED_SEQUENCE = 'fixed_sequence'
    BY_DATA_ATTRIBUTE = 'by_data_attribute'
    AGE = 'AGE'
    GEEM = 'geem'
    ANRMAB = 'anrmab'
    FEAT_PROP = 'feat_prop'
    SEAL = 'seal'
    UNCERTAINTY_DIFFERENCE = 'uncertainty_difference'
    APPROXIMATE_UNCERTAINTY = 'approximate_uncertainty'
    GALAXY = 'galaxy'
    BADGE = 'badge'

@unique
class CoresetDistance(StrEnum):
    
    LATENT_FEATURES = 'latent_space'
    INPUT_FEATURES = 'input_features'
    APPR = 'appr' 

@unique
class OracleAcquisitionUncertaintyType(StrEnum):
    EPISTEMIC = 'epistemic'
    ALEATORIC = 'aleatoric'
    TOTAL = 'total'
    
@unique
class DataAttribute(StrEnum):
    IN_DEGREE = 'in_degree'
    OUT_DEGREE = 'out_degree'
    APPR = 'appr'