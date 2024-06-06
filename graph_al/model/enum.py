from enum import unique, StrEnum

@unique
class ModelType(StrEnum):
    GCN = 'gcn'
    BAYESIAN_GCN = 'bayesian_gcn'
    APPNP = 'appnp'
    GPN = 'gpn'
    BAYES_OPTIMAL = 'bayes_optimal'
    SGC = 'sgc'
    SEAL = 'seal'

@unique
class ApproximationType(StrEnum):
    BRUTE_FORCE = 'brute_force'
    LIKELIHOOD_CACHE = 'likelihood_cache'
    MONTE_CARLO = 'monte_carlo'
    LAZY_MC = 'lazy_monte_carlo'
    IMPORTANCE = 'importance'
    BEAM_SEARCH = 'beam_search'
    CACHED_BEAM_SEARCH = 'cached_beam_search'
    VARIATIONAL_INFERENCE = 'variational_inference'

@unique
class GPNEvidenceScale(StrEnum):
    LATENT_OLD = 'latent-old'
    LATENT_NEW = 'latent-new'
    LATENT_NEW_PLUS_CLASSES = 'latent-new-plus-classes'
    NONE = 'none'   

@unique
class BayesianPrediction(StrEnum):
    MARGINAL = 'marginal' # predict by maximzing p(y_i = c | Y_o, A, X) for each node independently
    JOINT = 'joint' # predict by maximizing p(y_u | Y_o, A, X) for all unobserved nodes jointly
    JOINT_BEAM_SEARCH = 'joint_beam_search' # predict by maximizing p(y_u | Y_o, A, X) via beam search for all unobserved nodes jointly
    JOINT_CACHED_BEAM_SEARCH = 'joint_cached_beam_search' # caches values of unique assignments per beam for a more precise approximation but at a higher memory overhead
    JOINT_CACHED_LIKELIHOOD = 'joint_cached_likelihood' # predict by maximizing p(y_u | Y_o, A, X) for all unobserved nodes jointly using the likelihood cache

@unique
class LogisticRegressionSolver(StrEnum):
    LBFGS = 'lbfgs'
    LIBLINEAR = 'liblinear'
    SAG = 'sag'

@unique
class PredictionAttribute(StrEnum):
    """ Several prediction attributes """
    ENTROPY = 'entropy' # H[p(y)]
    MAX_SCORE = 'max_score' # max_c p(y)_c
    """ StrEnum for how to compute epistemic confidence. """
    MUTUAL_INFORMATION = 'mutual_information' # MI[y, theta]
    PREDICTED_VARIANCE = 'predicted_variance' # var_theta[p(y)_c'], where c' is argmax E_theta[p(y)]
    TOTAL_VARIANCE = 'total_variance' # sum_c var_theta[p(y)_c]
    LOG_EVIDENCE = 'log_evidence' # for dirichlet methods (GPN) log(alpha_0) = log (sum_c beta_c)
    NONE = 'none'
    GROUND_TRUTH = 'grount_truth'
    ENERGY = 'energy'