from abc import abstractmethod
import torch
import torch.nn as nn
from typing import Dict, Any, List, Iterable

from graph_al.model.prediction import Prediction
from graph_al.data.base import Dataset
from graph_al.model.config import ModelConfig, ModelConfigMultipleSamples, ModelConfigMonteCarloDropout
from graph_al.data.base import Data

from torch_geometric.data.batch import Batch

from torch import Tensor
from jaxtyping import jaxtyped, Shaped, Float
from typeguard import typechecked

class BaseModel(nn.Module):
    """ Base class for models. """
    
    def __init__(self, config: ModelConfig, dataset: Dataset):
        super().__init__()
        self.name = config.name

    def reset_cache(self):
        ...
        
    @abstractmethod
    def reset_parameters(self, generator: torch.Generator):
        ...
        
    @abstractmethod
    def predict(self, batch: Data, acquisition: bool=False) -> Prediction:
        ...

    @property
    def prediction_changes_at_eval(self) -> bool:
        return False

class BaseModelMultipleSamples(BaseModel):
    """ Base model class for models that can sample multiple predictions probabilistically. """
    
    def __init__(self, config: ModelConfigMultipleSamples, dataset: Dataset):
        super().__init__(config, dataset)
        self.num_samples_eval = config.num_samples_eval
        self.num_samples_train = config.num_samples_train
        self._collate_samples = config.collate_samples
        
    @abstractmethod
    def predict_multiple(self, batch: Data, num_samples: int, acquisition: bool=False) -> Prediction:
        ...
    
    @typechecked
    def collate_samples(self, batch: Data, num_samples: int) -> Data:
        """ Collates the input data `num_samples` times for parallel GPU computation. """
        return Batch.from_data_list([batch for _ in range(num_samples)]) # type: ignore
    
    @jaxtyped(typechecker=typechecked)
    def split_predicted_tensor(self, tensor: Shaped[Tensor, 'num_samples_collated ...'] | None, num_samples: int) -> Shaped[Tensor, 'num_samples num_nodes ...'] | None:
        """ Splits a tensor on a collated batch obtained from `self.collate_samples` """
        if tensor is None:
            return None
        split = tensor.resize(num_samples, tensor.size(0) // num_samples, *(list(tensor.size())[1:]))
        return split # type: ignore
    
    
    @typechecked    
    def predict(self, batch: Data, acquisition: bool=False) -> Prediction:
        num_samples = self.num_samples_train if self.training else self.num_samples_eval
        return self.predict_multiple(batch, num_samples, acquisition=acquisition)

    @property
    def prediction_changes_at_eval(self) -> bool:
        return True
        
class BaseModelMonteCarloDropout(BaseModelMultipleSamples):
    """ Base model class for models that can sample multiple predictions probabilistically using MC dropout . """
    
    def __init__(self, config: ModelConfigMonteCarloDropout, dataset: Dataset):
        super().__init__(config, dataset)
        self.dropout = config.dropout
        self.dropout_at_eval = config.dropout_at_eval
        
class Ensemble(BaseModel):
    """ Wrapper for an ensemble of models. """
    
    
    def __init__(self, config: ModelConfig, dataset: Dataset, models: List[BaseModel]):
        super().__init__(config, dataset)
        self.models = nn.ModuleList(models)

    def reset_cache(self):
        for model in self.models:
            model.reset_cache()

    @property
    def prediction_changes_at_eval(self) -> bool:
        return any(model.prediction_changes_at_eval for model in self.models)
        
    @typechecked
    def reset_parameters(self, generator: torch.Generator):
        for model in self.models:
            model.reset_parameters(generator) # type: ignore
        
    @typechecked
    def predict(self, batch: Data, acquisition: bool=False) -> Prediction:
        return Prediction.collate(self.predict_each(batch, acquisition=acquisition)) # type: ignore
    
    @typechecked
    def predict_each(self, batch: Data, acquisition: bool=False) -> List[Prediction]:
        """ Predictions by each ensemble member

        Args:
            batch (Data): the batch for which to predict
            acquisition (bool, optional): if it is an acquisition prediction. Defaults to False.

        Returns:
            List[Prediction]: the predictions for each member
        """
        return [model.predict(batch, acquisition=acquisition) for model in self.models] # type: ignore
    