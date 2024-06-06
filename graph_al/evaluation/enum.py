from enum import unique, StrEnum
import itertools
from graph_al.data.enum import DatasetSplit
from typing import Dict
from collections import ChainMap
from dataclasses import dataclass

from graph_al.model.enum import PredictionAttribute


@unique
class MetricName(StrEnum):
    LOSS = 'loss'
    ACCURACY = 'accuracy'
    F1 = 'f1'
    ECE = 'ece'
    AUCROC = 'auc_roc'
    AUCPR = 'auc_pr'

    # Meta-metrics
    NUM_ACQUIRED = 'num_acquired'
    # Entropy of the distribution of acquired class labels
    ACQUIRED_CLASS_DISTRIBUTION_ENTROPY = 'acquired_class_distribution_entropy'
    # The actual distribution of acquired class labels
    ACQUIRED_CLASS_DISTRIBUTION = 'acquired_class_distribution'
    # The acquired class counts
    ACQUIRED_CLASS_COUNTS = 'acquired_class_counts'
    
    # GPN specific
    UCE_LOSS = 'uce'
    
    ENTROPY_REGULARIZATION_LOSS = 'entropy_regularization'

    TRAIN_ENTROPY_REGULARIZATION_LOSS = 'loss/train/entropy_regularization'
    VAL_ENTROPY_REGULARIZATION_LOSS = 'loss/val/entropy_regularization'
    TEST_ENTROPY_REGULARIZATION_LOSS = 'loss/test/entropy_regularization'
    ALL_ENTROPY_REGULARIZATION_LOSS = 'loss/all/entropy_regularization'
    
    # SEAL specific
    MEAN_FEATURE_DISCREPANCY = 'mean_feature_discrepancy'
    ADVERSIAL_LOSS = 'adversarial_loss'

@dataclass(unsafe_hash=True)
class MetricTemplate:
    """ Template for a metric that is not yet finalized with respect to its representation. This can be used for passing metrics through different
    levels of the pipeline and only logging them when the template is finalized """
    name: MetricName | None = None
    dataset_split: DatasetSplit | None = None
    prediction_attribute: PredictionAttribute | None = None
    propagated: bool = True # by default, all metrics are propagated
    targets_propagated: bool = True # by default, the targets for the metric are propagated

    def __repr__(self) -> str:
        fields = [
            x for x in (self.name, self.dataset_split,
                            self.prediction_attribute)
            if x is not None
        ]
        if not self.propagated:
            fields.append('unpropagated') # type: ignore
        if not self.targets_propagated:
            fields.append('targets_unpropagated')
        return '/'.join(fields)
