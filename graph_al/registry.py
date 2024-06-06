from typing import Any
from filelock import SoftFileLock
from tinydb import TinyDB, Query
from datetime import datetime
import os
from os import PathLike

class Registry:
    """ Thread-safe database for artifacts. 
    
    Parameters:
    -----------
    database_path : str
        Path to the database file.
    lockfile_path : str
        Path to the lockfile.
    key : Function
        A function from elements to put in the database to a representation string.
    """
    
    def __init__(self, database_path: PathLike[str] | str, lockfile_path: PathLike[str] | str, key=str):
        os.makedirs(os.path.dirname(database_path), exist_ok=True)
        os.makedirs(os.path.dirname(lockfile_path), exist_ok=True)
        self.database_path = database_path
        self.lock = SoftFileLock(lockfile_path)
        self.key = key

    def __getitem__(self, key: Any) -> str:
        key = self.key(key)
        with self.lock:
            db = TinyDB(self.database_path)
            query = Query()
            result = db.search(query.key == key)
            if len(result) == 0:
                raise KeyError(key)
            elif len(result) > 1:
                raise RuntimeError(f'Multiple entries found for {key}')
            else:
                return result[0]['value']
            
    def __setitem__(self, key: Any, value: str):
        key = self.key(key)
        with self.lock:
            db = TinyDB(self.database_path)
            query = Query()
            db.remove(query.key == key)
            db.insert({'key' : key, 'value' : value, 'timestamp' : str(datetime.now(tz=None))})

    def items(self):
        """ Iterates over the items. """
        with self.lock:
            db = TinyDB(self.database_path)
            for item in db:
                yield (item['key'], item['value'])
    
    def __contains__(self, key: Any) -> bool:
        key = self.key(key)
        with self.lock:
            db = TinyDB(self.database_path)
            query = Query()
            return len(db.search(query.key == key)) > 0
