from graph_al.utils._sbm import count_in_class_by_adjacency as count_in_class_by_adjacency_c
from graph_al.utils._sbm import count_in_class_by_triangular_upper_adjacency as count_in_class_by_triangular_upper_adjacency_c
from graph_al.utils._sbm import class_counts_by_node_to_affiliation_counts as class_counts_by_node_to_affiliation_counts_c

import numpy as np
from jaxtyping import jaxtyped, Int
from typeguard import typechecked
from typing import Tuple

@jaxtyped(typechecker=typechecked)
def count_in_class_by_adjacency(sampled_labels: Int[np.ndarray, 'num_samples num_nodes'],
                                edge_idxs: Int[np.ndarray, '2 num_edges'],
                                num_classes: int) -> Tuple[
                                    Int[np.ndarray, 'num_samples num_classes'],
                                    Int[np.ndarray, 'num_samples num_nodes num_classes'],
                                    Int[np.ndarray, 'num_samples num_nodes num_classes'],
                                ]:                                
    """ Counts how often a label appears in the (non-(neighbourhood of each node along axis 1 for each row in `labels`
    
    Parameters:
    -----------
    sampled_labels : ndarray, shape [num_samples, num_nodes]
        For each `num_samples` samples, what the node class assignments are (i.e. row corresponds to assignment)
    edge_idxs : ndarray, shape [2, num_edges]
        Tuples of indices (i, j), where j is in the neighbourhood of i, and i!=j always
    num_classes : int
        The number of classes
        
    Returns:
    --------
    counts : ndarray, shape [num_samples, num_classes]
        For each sample, the class counts in this sample
    counts_in_neighbourhood : ndarray, shape [num_samples, num_nodes, num_classes]
        For each sample and each node, the class counts among its neighbours
    counts_not_in_neighbourhood : ndarray, shape [num_samples, num_nodes, num_classes]
        For each sample and each node, the class counts among its non-neighbours
        Does not count self-loops as a non-neighbour
    """
    return count_in_class_by_adjacency_c(sampled_labels, edge_idxs, num_classes)
    
@jaxtyped(typechecker=typechecked)
def count_in_class_by_triangular_upper_adjacency(sampled_labels: Int[np.ndarray, 'num_samples num_nodes'],
                                edge_idxs: Int[np.ndarray, '2 num_edges'],
                                num_classes: int) -> Tuple[
                                    Int[np.ndarray, 'num_samples num_nodes num_classes'],
                                    Int[np.ndarray, 'num_samples num_nodes num_classes'],
                                    Int[np.ndarray, 'num_samples num_nodes num_classes'],
                                ]:                                
    """ Counts how often a label appears in the (non-(neighbourhood of each node along axis 1 for each row in `labels`.
    It only considers the upper triangular part of the adjacency matrix, i.e. non-neighbours are not counted if they
    don't appear in the upper triangular part of the adjacency.
    
    Parameters:
    -----------
    sampled_labels : ndarray, shape [num_samples, num_nodes]
        For each `num_samples` samples, what the node class assignments are (i.e. row corresponds to assignment)
    edge_idxs : ndarray, shape [2, num_edges]
        Tuples of indices (i, j), where j is in the neighbourhood of i, and i!=j always
    num_classes : int
        The number of classes
        
    Returns:
    --------
    counts : ndarray, shape [num_samples, num_nodes, num_classes]
        For each sample, the class counts in this sample for each row in the upper adjacency matrix.
    counts_in_neighbourhood : ndarray, shape [num_samples, num_nodes, num_classes]
        For each sample and each node, the class counts among its neighbours
    counts : ndarray, shape [num_samples, num_nodes, num_classes]
        For each sample and each node, the class counts among its non-neighbours
        Does not count self-loops as a non-neighbour
    """ 
    assert not (edge_idxs[1] <= edge_idxs[0]).any(), f'The argument `edge_idxs` you passed should describe an upper triangular matrix'
    return count_in_class_by_triangular_upper_adjacency_c(sampled_labels, edge_idxs, num_classes)

def class_counts_by_node_to_affiliation_counts(class_counts: Int[np.ndarray, 'num_samples num_nodes num_classes'], 
                                               labels: Int[np.ndarray, 'num_samples num_nodes']) -> Int[np.ndarray, 'num_samples num_classes num_classes']:
    """ Transforms class counts per node to affiliation counts, i.e. counting
    how often a class links to a class.
    
    Parameters:
    -----------
    class_counts : ndarray, shape [S, num_nodes, num_classes]
        For each node in each assignment, how often it counts a class
    labels : ndarray, shape [S, num_nodes]
        For each S samples, what the node class assignments are (i.e. row corresponds to assignment)
        
    Returns:
    --------
    affiliation_counts : ndarray, shape [S, num_classes, num_classes]
        For each assignment, how often classes are affilated
    """
    return class_counts_by_node_to_affiliation_counts_c(class_counts, labels)