
from os import PathLike
from graph_al.registry import Registry
import numpy as np
import uuid
from pathlib import Path
import os

from jaxtyping import Int, Float, jaxtyped
from typing import Any, Tuple
from typeguard import typechecked

class ConditionalLogLikelihoodRegistry(Registry):
    """ Class to store conditional log likelihoods. """

    @typechecked
    def __init__(self, database_path: PathLike[str] | str, storage_dir: PathLike[str] | str, lockfile_path: PathLike[str] | str):
        super().__init__(database_path, lockfile_path, key=str)
        self.storage_dir = Path(storage_dir)
        os.makedirs(storage_dir, exist_ok=True)

    @jaxtyped(typechecker=typechecked)
    def __setitem__(self, key: Any, value: Tuple[Int[np.ndarray, 'num_assignments num_nodes'],
                Float[np.ndarray, 'num_assignments']],
        ):
        path = self.storage_dir / f'{uuid.uuid4()}.npz'
        np.savez(path, assignments=value[0], log_likelihoods=value[1])
        super().__setitem__(key, str(path))
    
    @jaxtyped(typechecker=typechecked)
    def __getitem__(self, key: Any) -> Tuple[Int[np.ndarray, 'num_assignments num_nodes'],
                Float[np.ndarray, 'num_assignments']]:
        path = super().__getitem__(key)
        storage = np.load(path)
        return storage['assignments'], storage['log_likelihoods']