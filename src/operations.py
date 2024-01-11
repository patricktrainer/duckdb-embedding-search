import os
import pickle
from typing import List, Tuple, Dict
from .connection import DuckDBPyConnection
from duckdb.typing import DuckDBPyType


ARRAY_TYPE = DuckDBPyType(list[float])  # type: ignore
PickleCache = Dict[Tuple[str, str], List[float]]


def write_embedding_to_table(
    con: DuckDBPyConnection, text: str, model: str, embedding: List[float]
) -> DuckDBPyConnection:
    create_table_if_not_exists(con)
    con.execute("INSERT INTO embeddings VALUES (?, ?, ?)", [text, model, embedding])
    return con


def create_table_if_not_exists(con) -> None:
    con.from_query(
        f"CREATE TABLE IF NOT EXISTS embeddings (text VARCHAR, model VARCHAR, embedding {ARRAY_TYPE})"
    )


def is_key_in_table(con: DuckDBPyConnection, key: Tuple[str, str]) -> bool:
    create_table_if_not_exists(con)
    result = con.execute(
        "SELECT EXISTS(SELECT * FROM embeddings WHERE text=? AND model=?)",
        [key[0], key[1]],
    ).fetchone()
    if result:
        return result[0]
    return False


def list_keys_in_table(
    con: DuckDBPyConnection, keys: List[Tuple[str, str]]
) -> list[tuple[str, str]]:
    keys_in_table = []

    for key in keys:
        if is_key_in_table(con, key):
            keys_in_table.append(key)
    return keys_in_table


def load_pickle_cache(pickle_path: str) -> PickleCache:
    if os.path.exists(pickle_path):
        with open(pickle_path, "rb") as file:
            return pickle.load(file)
    return {}


def write_pickle_cache_to_duckdb(con: DuckDBPyConnection, pickle_path: str) -> None:
    cache = load_pickle_cache(pickle_path)
    create_table_if_not_exists(con)
    for key, value in cache.items():
        write_embedding_to_table(con, key[0], key[1], value)


# Function to save the cache to a file
def save_pickle_cache(cache: PickleCache, cache_path: str) -> None:
    with open(cache_path, "wb") as file:
        pickle.dump(cache, file)


def get_embedding_from_table(con: DuckDBPyConnection, text: str, model: str) -> List[float]:
    result = con.execute(
        "SELECT embedding FROM embeddings WHERE text=? AND model=?", [text, model]
    ).fetchone()
    if result:
        return result[0]
    raise ValueError(f"Embedding for {text} with model {model} not found in table")
