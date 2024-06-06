import numpy as np
cimport numpy as np
import cython

np.import_array()

DTYPE_FLOAT = float
DTYPE_INT = int

ctypedef np.float_t DTYPE_FLOAT_t
ctypedef np.int_t DTYPE_INT_t

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def count_in_class_by_adjacency(
    np.ndarray[DTYPE_INT_t, ndim=2] labels, 
    np.ndarray[DTYPE_INT_t, ndim=2] edge_idxs,
    int num_classes):
    """ Counts how often a label appears in the (non-(neighbourhood of each node along axis 1 for each row in `labels`
    
    Parameters:
    -----------
    labels : ndarray, shape [S, num_nodes]
        For each S samples, what the node class assignments are (i.e. row corresponds to assignment)
    edge_idxs : ndarray, shape [2, num_edges]
        Tuples of indices (i, j), where j is in the neighbourhood of i, and i!=j always
    num_classes : int
        The number of classes
        
    Returns:
    --------
    counts : ndarray, shape [S, num_classes]
        For each sample, the class counts in this sample
    counts_in_neighbourhood : ndarray, shape [S, num_nodes, num_classes]
        For each sample and each node, the class counts among its neighbours
    counts : ndarray, shape [S, num_nodes, num_classes]
        For each sample and each node, the class counts among its non-neighbours
        Does not count self-loops as a non-neighbour
    """
    
    assert labels.dtype == DTYPE_INT
    cdef int S = labels.shape[0]
    cdef int N = labels.shape[1]
    cdef int E = edge_idxs.shape[1]
    cdef np.ndarray counts = np.zeros([S, num_classes], dtype=DTYPE_INT)
    cdef np.ndarray counts_adj = np.zeros([S, N, num_classes], dtype=DTYPE_INT)
    cdef np.ndarray counts_non_adj = np.zeros([S, N, num_classes], dtype=DTYPE_INT)
    
    # Per sample, count the number of nodes in each class
    for s in range(S):
        for v in range(N):
            counts[s, labels[s, v]] += 1
    
    # Per sample, per node, count the number adjacent nodes in each class
    for s in range(S):
        for e in range(E):
            counts_adj[s, edge_idxs[0, e], labels[s, edge_idxs[1, e]]] += 1

    # Compute the number of non-adjacent nodes as the difference of the above
    for s in range(S):
        for v in range(N):
            for c in range(num_classes):
                counts_non_adj[s, v, c] = counts[s, c] - counts_adj[s, v, c]
    
    for s in range(S):
        for v in range(N):
            counts_non_adj[s, v, labels[s, v]] -= 1 # we don't count the self-loop as non-adjacency
            
    return counts, counts_adj, counts_non_adj
                

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def count_in_class_by_triangular_upper_adjacency(
    np.ndarray[DTYPE_INT_t, ndim=2] labels, 
    np.ndarray[DTYPE_INT_t, ndim=2] edge_idxs,
    int num_classes):
    """ Counts how often a label appears in the (non-)neighbourhood of each node along axis 1 for each row in `labels`.
    It only considers the upper triangular part of the adjacency matrix, i.e. non-neighbours are not counted if they
    don't appear in the upper triangular part of the adjacency.
    
    Parameters:
    -----------
    labels : ndarray, shape [S, num_nodes]
        For each S samples, what the node class assignments are (i.e. row corresponds to assignment)
    edge_idxs : ndarray, shape [2, num_edges]
        Tuples of indices (i, j), where j is in the neighbourhood of i, and i!=j always
    num_classes : int
        The number of classes
        
    Returns:
    --------
    counts : ndarray, shape [S, num_nodes, num_classes]
        For each sample, the class counts in this sample for each row in the upper adjacency matrix.
    counts_in_neighbourhood : ndarray, shape [S, num_nodes, num_classes]
        For each sample and each node, the class counts among its neighbours
    counts : ndarray, shape [S, num_nodes, num_classes]
        For each sample and each node, the class counts among its non-neighbours
        Does not count self-loops as a non-neighbour
    """
    assert labels.dtype == DTYPE_INT
    cdef int S = labels.shape[0]
    cdef int N = labels.shape[1]
    cdef int E = edge_idxs.shape[1]
    cdef np.ndarray counts = np.zeros([S, N, num_classes], dtype=DTYPE_INT)
    cdef np.ndarray counts_adj = np.zeros([S, N, num_classes], dtype=DTYPE_INT)
    cdef np.ndarray counts_non_adj = np.zeros([S, N, num_classes], dtype=DTYPE_INT)
    
    # Per sample, count the number of nodes in each class per row of the upper triangular adjacency
    for s in range(S):
        for v in range(N):
            for u in range(v): # all the rows for which v is counted (we only count the upper triangular part)
                counts[s, u, labels[s, v]] += 1
    
    # Per sample, per node, count the number adjacent nodes in each class
    for s in range(S):
        for e in range(E):
            counts_adj[s, edge_idxs[0, e], labels[s, edge_idxs[1, e]]] += 1

    # Compute the number of non-adjacent nodes as the difference of the above
    for s in range(S):
        for v in range(N):
            for c in range(num_classes):
                counts_non_adj[s, v, c] = counts[s, v, c] - counts_adj[s, v, c]
    
    return counts, counts_adj, counts_non_adj

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def class_counts_by_node_to_affiliation_counts(
    np.ndarray[DTYPE_INT_t, ndim=3] class_counts,
    np.ndarray[DTYPE_INT_t, ndim=2] labels,):
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
    
    assert labels.dtype == DTYPE_INT
    cdef int num_samples = class_counts.shape[0]
    cdef int num_nodes = class_counts.shape[1]
    cdef int num_classes = class_counts.shape[2]
    
    affiliation_counts = np.zeros([num_samples, num_classes, num_classes], dtype=DTYPE_INT)
    for s in range(num_samples):
        for v in range(num_nodes):
            for c in range(num_classes):
                affiliation_counts[s, labels[s, v], c] += class_counts[s, v, c]
        
    return affiliation_counts