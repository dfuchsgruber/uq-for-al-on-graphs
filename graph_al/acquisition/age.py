from graph_al.acquisition.base import BaseAcquisitionStrategy
from graph_al.acquisition.config import AcquisitionStrategyAGEConfig, AcquisitionStrategyAGELikeConfig
from graph_al.data.base import BaseDataset, Dataset
from graph_al.model.base import BaseModel
from graph_al.model.prediction import Prediction

from jaxtyping import Float, Int, jaxtyped, Bool, Shaped
from typeguard import typechecked
from torch import Tensor
from typing import Dict, Tuple, Any
from sklearn.cluster import KMeans
import torch
from torch_geometric.utils import to_dense_adj
import numpy as np
import networkx as nx

from graph_al.model.config import ModelConfig

def percd(input,k): 
    return torch.mean((input > input[k]).float())

def perc(input,k): 
    return torch.mean((input < input[k]).float())


class AcquisitionStrategyAGELike(BaseAcquisitionStrategy):
    """ Strategy that uses a mix of uncertainty and representativeness and centrality. """

    def __init__(self, config: AcquisitionStrategyAGELikeConfig):
        super().__init__(config)
        self.num_clusters = config.num_clusters
        self.centrality_measure = None
                
    @property
    def is_stateful(self) -> bool:
        return True # has centrality as a state

    def reset(self):
        super().reset()
        self.centrality_measure = None
                
    @jaxtyped(typechecker=typechecked)
    def _calculate_centrality(self, dataset: Dataset) -> Float[Tensor, 'num_nodes']:
        adj = to_dense_adj(dataset.data.edge_index)
        graph = nx.from_numpy_array(adj[0].cpu().numpy())
        centralities = []
        centralities.append(nx.pagerank(graph))                #print 'page rank: check.'
        L = len(centralities[0])
        Nc = len(centralities)
        cenarray = np.zeros((Nc,L))
        for i in range(Nc):
            cenarray[i][list(centralities[i].keys())]=list(centralities[i].values())
        normcen = (cenarray.astype(float)-np.min(cenarray,axis=1)[:,None])/(np.max(cenarray,axis=1)-np.min(cenarray,axis=1))[:,None]
        normcen =  torch.tensor(normcen).float() # centrality for all possible nodes
        return normcen[0]
    
    @jaxtyped(typechecker=typechecked)
    def _centrality(self, mask_train: Bool[Tensor, 'num_nodes_train_pool'], 
                  mask_train_pool: Bool[Tensor, 'num_nodes'], 
                  dataset: Dataset): #-> Float[Tensor, 'num_nodes_train_pool']:
        if self.centrality_measure is None:
            self.centrality_measure = self._calculate_centrality(dataset=dataset)
        return self.centrality_measure.to(mask_train_pool.device)[mask_train_pool]
    
    @jaxtyped(typechecker=typechecked)
    def _entropy(self, mask_train: Bool[Tensor, 'num_nodes'], 
                  mask_train_pool: Bool[Tensor, 'num_nodes'], 
                  prediction: Prediction) -> Float[Tensor, 'num_nodes_train_pool']:
        x = prediction.logits

        if x is None:
            raise RuntimeError(f'Model does not predict attribute requested by AGE entropy')

        x = torch.softmax(x, 2)[0]
        entropies = torch.distributions.Categorical(x).entropy()
        return entropies[mask_train_pool]
        
    @jaxtyped(typechecker=typechecked)
    def _representativeness(self, mask_train: Bool[Tensor, 'num_nodes'], 
                  mask_train_pool: Bool[Tensor, 'num_nodes'],
                  prediction: Prediction) -> Float[Tensor, 'num_nodes_train_pool']:
        """ computes the distance metric used for the coreset algorithm """
        x = prediction.logits
        
        if x is None:
            raise RuntimeError(f'Model does not predict attribute requested by AGE representativeness')
        x = x[0]
        
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=0).fit(x.cpu().numpy())
        ed = torch.cdist(x[mask_train_pool], torch.tensor(kmeans.cluster_centers_).to(x.device))
        ed_score = torch.min(ed, dim=1).values
        edprec = torch.tensor([percd(ed_score,i) for i in range(len(ed_score))]).to(x.device)

        return edprec



class AcquisitionStrategyAGE(AcquisitionStrategyAGELike):
    """ Strategy that uses a mix of uncertainty and representativeness.
     
     
    References:
    [1] : https://arxiv.org/abs/1705.05085
    """
    def __init__(self, config: AcquisitionStrategyAGEConfig):
        super().__init__(config)
        self.basef = config.basef
    
        
    @torch.no_grad()
    def acquire_one(self, mask_acquired: Bool[Tensor, 'num_nodes'], prediction: Prediction | None, model: BaseModel, dataset: Dataset, model_config: ModelConfig, 
        generator: torch.Generator) -> Tuple[int, Dict[str, Tensor | None]]:

        mask_train, mask_train_pool = dataset.data.mask_train.clone(), dataset.data.mask_train_pool.clone()
        
        # time param are num acquired which only makes sense if we only use one at a time.
        time_value = torch.sum(mask_train)
        
        gamma = torch.tensor(np.random.beta(1, 1.005 - self.basef ** time_value.cpu())).to(dataset.data.x.device)
        alpha = beta = (1 - gamma) / 2
        
        assert prediction is not None, f'Need a model prediction for AGE acquisition'

        idxs_train_pool = torch.where(mask_train_pool)[0]
        if mask_train.sum() == 0:
            # No training instances, sample one index randomly from pool
            sampled_idx = idxs_train_pool[torch.randint(idxs_train_pool.size(0), (1,), generator=generator).item()] # type: ignore
            representativeness, entropy, centrality = None, None, None
        else:
            # Select the instance from the pool according to age
            representativeness = self._representativeness(
                mask_train=mask_train, 
                mask_train_pool=mask_train_pool, 
                prediction=prediction
                )
            entropy = self._entropy(
                mask_train=mask_train, 
                mask_train_pool=mask_train_pool, 
                prediction=prediction
                )
            centrality = self._centrality(
                mask_train=mask_train, 
                mask_train_pool=mask_train_pool, 
                dataset=dataset
                )
            final_weight = alpha * entropy + beta * representativeness + gamma * centrality
            # getting the correct index since final weight only includes train pool
            sampled_idx = torch.tensor(range(len(mask_train_pool)))[mask_train_pool.cpu()][final_weight.cpu().argmax()]

        return int(sampled_idx.item()), {'mask_train' : mask_train, 'mask_train_pool' : mask_train_pool,
            'representativeness' : representativeness, 'entropy' : entropy, 'centrality' : centrality}
