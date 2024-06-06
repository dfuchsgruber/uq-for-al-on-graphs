from enum import StrEnum, unique

@unique
class GraphSetting(StrEnum):
    TRANSDUCTIVE = 'transductive'
    INDUCTIVE = 'inductive'

@unique
class DatasetSplit(StrEnum):
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'
    TRAIN_POOL = 'train_pool'
    ALL = 'all'
    UNLABELED = 'unlabeled' # all but train
    UNLABELED_MINUS_TEST = 'unlabeled_minus_test' # all but train and test

@unique
class DatasetType(StrEnum):
    RANDOM_SBM = 'random_sbm'
    DETERMINISTIC_SBM = 'deterministic_sbm'
    NPZ = 'npz'
    TORCH_GEOMETRIC = 'torch_geometric'

@unique
class SBMClassMeanSampling(StrEnum):
    NORMAL = 'normal'
    EQUIDISTANT = 'equidistant'
    
@unique
class SBMEdgeProbabilitiesType(StrEnum):
    CONSTANT = 'constant'
    BY_SNR_AND_EXPECTED_DEGREE = 'by_snr_and_expected_degree'

@unique
class NpzFeaturePreprocessing(StrEnum):
    BAG_OF_WORDS = 'bag-of-words'
    NONE = 'none'
    
@unique
class NpzFeatureVectorizer(StrEnum):
    TF_IFD = 'tf-ifd'
    COUNT = 'count'

@unique
class TorchGeometricDatasetType(StrEnum):
    CORA_ML = 'cora_ml'
    CITESEER = 'citeseer'
    PUBMED = 'pubmed'
    AMAZON_PHOTOS = 'amazon_photos'
    AMAZON_COMPUTERS = 'amazon_computers'
    OGBN_ARXIV = 'ogbn_arxiv'
    REDDIT = 'reddit'  
    
@unique
class FeatureNormalization(StrEnum):
    L1 = 'l1'
    L2 = 'l2'
    NONE = 'none'