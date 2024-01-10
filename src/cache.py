import os
import pickle
from typing import Dict, List, Tuple
from .connection import DuckDBPyConnection
from .operations import create_table_if_not_exists, write_embedding_to_table


EmbeddingCache = Dict[Tuple[str, str], List[float]]


def load_pickle_cache(pickle_path: str) -> EmbeddingCache:
    if os.path.exists(pickle_path):
        with open(pickle_path, "rb") as file:
            return pickle.load(file)
    return {}


def write_pickle_cache_to_duckdb(con: DuckDBPyConnection, pickle_path: str) -> None:
    cache = load_pickle_cache(pickle_path)
    create_table_if_not_exists(con)
    for key, value in cache.items():
        write_embedding_to_table(con, key[0], key[1], value)
