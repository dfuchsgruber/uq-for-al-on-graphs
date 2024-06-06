import torch
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import scipy.sparse as sp
import numpy as np

from jaxtyping import jaxtyped, Bool, Int, Float, Shaped
from typing import Iterable, Any, Literal, TypeAlias, List
from typeguard import typechecked

from graph_al.data.base import BaseDataset
from graph_al.utils.data import *
from graph_al.data.config import NpzConfig, NpzFeatureVectorizer, NpzFeaturePreprocessing, FeatureNormalization
from graph_al.utils.logging import get_logger
from graph_al.data.transform import normalize_features

@jaxtyped(typechecker=typechecked)
def load_graph(loader: np.lib.npyio.NpzFile, make_undirected: bool=True, make_unweighted: bool=True, select_lcc: bool=True, 
                    remove_self_loops: bool=True) -> Tuple[sp.csr_matrix, Bool[np.ndarray, "num_original"]]:
        """ Loads the graph from the dataset

        Args:
            loader (Dict): Npz dataloader that contains the relevant information
            make_undirected (bool, optional): If to make the graph undirected. Defaults to True.
            make_unweighted (bool, optional): If to make the graph unweighted. Defaults to True.
            select_lcc (bool, optional): If to select the largest connected component. Defaults to True.
            remove_self_loops (bool, optional): If to remove self loops. Defaults to True.

        Returns:
            _type_: _description_
        """
        
        adj_data = loader['adj_data']
        
        # Make unweighted : Replace data with ones
        if make_unweighted:
            adj_data = np.ones_like(adj_data)
        A = sp.csr_matrix((adj_data, loader['adj_indices'], loader['adj_indptr']), shape=loader['adj_shape'])
        
        # Make undirected : Maximum of A and its transpose
        if make_undirected:
            A = sparse_max(A, A.T)
        edge_index = np.array(A.nonzero())
        
        # Remove self-loops
        if remove_self_loops:
            is_self_loop = edge_index[0, :] == edge_index[1, :]
            edge_index = edge_index[:, ~is_self_loop]
            
        # Select only the lcc from A
        if select_lcc:
            n_components, labels = sp.csgraph.connected_components(A)
            label_names, label_counts = np.unique(labels, return_counts=True)
            label_lcc = label_names[label_counts.argmax()]
            
            # Only keep vertices with labels == label_lcc
            vertices_to_keep = labels == label_lcc
            A = A.tocsr()[vertices_to_keep].tocsc()[:, vertices_to_keep].tocsr()
        else:
            vertices_to_keep = np.ones(A.shape[0], dtype=bool)
            
        return A.tocsr(), vertices_to_keep

@jaxtyped(typechecker=typechecked)
def build_vectorizer(corpus: Shaped[np.ndarray, "n"], y: Int[np.ndarray, "n"], idx_to_label: Dict[int, str], 
                     corpus_labels: Iterable[int] | Literal['all']='all', min_token_frequency: int=10, 
                     normalize: FeatureNormalization=FeatureNormalization.L2, 
                     vectorizer: NpzFeatureVectorizer=NpzFeatureVectorizer.TF_IFD) -> CountVectorizer:
    """ Builds a vectorizer

    Args:
        corpus (Shaped[np.ndarray, &quot;n&quot;]): the corpus to fit
        y (Int[np.ndarray, &quot;n&quot;]): labels for each node
        idx_to_label (Dict[int, str]): mapping from index to node label
        corpus_labels (Iterable | Literal['all'], optional): which class labels to base the vectorizer on. Defaults to 'all'.
        min_token_frequency (int, optional): How often a token needs to appear. Defaults to 10.
        normalize (Any, optional): Which normalization to use. Defaults to 'l2'.
        vectorizer (Literal[&#39;tf, optional): How to vectorize. Defaults to 'tf-idf'.

    Returns:
        CountVectorizer: The vectorizer
    """
    if corpus_labels == 'all':
        corpus_labels2 = set(idx_to_label.values())
    else:
        corpus_labels2 = set(corpus_labels)
    has_corpus_label = np.array([idx_to_label[label] in corpus_labels2 for label in y])
    match vectorizer:
        case NpzFeatureVectorizer.TF_IFD:
            vectorizer_object = TfidfVectorizer(min_df=min_token_frequency, norm=normalize)
        case NpzFeatureVectorizer.COUNT:
            vectorizer_object = CountVectorizer(min_df=min_token_frequency)
        case _:
            raise RuntimeError(f'Unsupported vectorizer type {vectorizer}.')
    vectorizer_object.fit(corpus[has_corpus_label])
    return vectorizer_object

@jaxtyped(typechecker=typechecked)
def build_node_to_idx(idx_to_node: Dict[int, str], vertices_to_keep: Bool[np.ndarray, "n"]) -> Dict[str, int]:
    """ Builds the mapping from node -> idx. 

    Args:
        idx_to_node (Dict[int, str]): Mapping from int -> node name
        vertices_to_keep (Bool[np.ndarray, &quot;n&quot;]): Which vertices in the original data (with N_raw >= N vertices) to keep.

    Returns:
        Dict[str, int]: Mapping from node name to its index in X
    """
    
    node_names: List[str | None] = [None for _ in range(max(idx_to_node.keys()) + 1)]
    for idx, name in idx_to_node.items():
        node_names[idx] = name
    assert None not in node_names, f'No name specified for node at idx {node_names.index(None)}'
    node_names_array = np.array(node_names)[vertices_to_keep]
    node_to_idx = {node : idx for idx, node in enumerate(node_names_array)}
    return node_to_idx

@jaxtyped(typechecker=typechecked)
def vectorize(corpus: Shaped[np.ndarray, "n"], vectorizer: CountVectorizer) -> Tuple[sp.spmatrix, Dict[str, int]]:
    """Loads the attribute matrix of the dataset using a vectorizer. 

    Args:
        corpus (Shaped[np.ndarray, &quot;n&quot;]): The text attributes of each node.
        vectorizer (CountVectorizer): The vectorizer

    Returns:
        (sp.spmatrix): Attribute matrix
        (Dict[str, int]): Mapping from feature name (word in vocabulary) to idx in X.
    """
    
    X = vectorizer.transform(corpus)
    if hasattr(vectorizer, 'get_feature_names_out'):
        feature_names = vectorizer.get_feature_names_out()
    else:
        feature_names = vectorizer.get_feature_names() # type: ignore
    feature_to_idx = {feat : idx for idx, feat in enumerate(feature_names)}
    return X, feature_to_idx

class NpzDataset(BaseDataset):
    """ Dataset loaded from a npz file. """

    def __init__(self, config: NpzConfig):
        loader = np.load(config.path, allow_pickle=True)
        A, vertices_to_keep = load_graph(loader, make_undirected=True, select_lcc=True, remove_self_loops=False)
        get_logger().info('Data Loading - Loaded adjacency matrix.')
        y = loader['labels'][vertices_to_keep]

        if 'idx_to_class' in loader:
            idx_to_label = loader['idx_to_class'].item()
        elif 'class_names' in loader:
            idx_to_label = {idx : name for idx, name in enumerate(loader['class_names'])}
        else:
            get_logger().info(f'Data Loading - Did not find class names in {config.path}. Generating default names...')
            idx_to_label = {idx : f'class_{idx}' for idx in range(int(max(y)) + 1)}

        if 'idx_to_node' in loader:
            idx_to_node = loader['idx_to_node'].item()
        elif 'node_names' in loader:
            idx_to_node = {idx : name for idx, name in enumerate(loader['node_names'])}
        else:
            get_logger().info(f'Data Loading - Did not find node names in {config.path}. Generating default names...')
            idx_to_node = {idx : f'node_{idx}' for idx in range(vertices_to_keep.shape[0])}

        # Build the attribute matrix
        match config.preprocessing:
            case NpzFeaturePreprocessing.BAG_OF_WORDS:
                corpus = loader['attr_text']
                vectorizer = build_vectorizer(corpus[vertices_to_keep], y, idx_to_label, corpus_labels='all', 
                    min_token_frequency=config.min_token_frequency, normalize=config.normalize, vectorizer=config.vectorizer)
                X, feature_to_idx = vectorize(corpus[vertices_to_keep], vectorizer)
                X = X.todense()
                node_to_idx = build_node_to_idx(idx_to_node, vertices_to_keep)
            case NpzFeaturePreprocessing.NONE:
                if 'features' in loader:
                    # Dense features
                    X = loader['features'][vertices_to_keep]
                    node_to_idx = build_node_to_idx(idx_to_node, vertices_to_keep)
                    feature_to_idx = {f'feature_{i}' : i for i in range(X.shape[1])}
                else:
                    # Sparse features
                    X = sp.csr_matrix((loader['attr_data'], loader['attr_indices'], loader['attr_indptr']), shape=loader['attr_shape'])[vertices_to_keep].todense()
                    node_to_idx = build_node_to_idx(idx_to_node, vertices_to_keep)
                if 'attr_names' in loader and len(loader['attr_names']):
                    feature_to_idx = {f'{feature}' : int(idx) for idx, feature in enumerate(loader['attr_names'])}
                elif 'idx_to_attr' in loader:
                    feature_to_idx = {f'{attr}' : int(idx) for idx, attr in loader['idx_to_attr'].item().items()}
                else:
                    feature_to_idx = {f'feature_{i}' : i for i in range(X.shape[1])}
                assert len(feature_to_idx) == X.shape[1], f'Mismatching sizes for attr_names {len(feature_to_idx)} != {X.shape[1]}'
                assert len(set(feature_to_idx.values())) == X.shape[1]
            case _:
                raise ValueError(f'Unknown preprocessing for features of type {config.preprocessing}')
        get_logger().info('Data Loading - Built attribute matrix.')

        X = normalize_features(torch.from_numpy(X), config.normalize, dim=1).numpy()
        
        label_to_idx = make_mapping_collatable({label : idx for idx, label in idx_to_label.items() if idx in y})
        y, label_to_idx, _ = compress_labels(y, label_to_idx)
        
        mask_train, mask_val, mask_test = None, None, None
        if config.use_public_split:
            mask_train = loader.get('mask_train', None)
            if mask_train is not None:
                mask_train = torch.from_numpy(mask_train[vertices_to_keep]).bool()
            mask_val = loader.get('mask_val', None)
            if mask_val is not None:
                mask_val = torch.from_numpy(mask_val[vertices_to_keep]).bool()
            mask_test = loader.get('mask_test', None)
            if mask_test is not None:
                mask_test = torch.from_numpy(mask_test[vertices_to_keep]).bool()
            
        
        super().__init__(
            node_features = torch.from_numpy(X).float(),
            edge_idxs = torch.tensor(np.array(A.nonzero())).long(),
            labels = torch.tensor(y).long(),
            node_to_idx = make_mapping_collatable(node_to_idx),
            label_to_idx = make_mapping_collatable(label_to_idx),
            feature_to_idx = make_mapping_collatable(feature_to_idx),
            mask_train=mask_train,
            mask_val=mask_val,
            mask_test=mask_test,
        )
        
        
        