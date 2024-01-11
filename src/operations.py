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
    """
    Writes the given embedding to the `embeddings` table in the database.

    Args:
        con (DuckDBPyConnection): The connection to the DuckDB database.
        text (str): The text associated with the embedding.
        model (str): The model used to generate the embedding.
        embedding (List[float]): The embedding vector.

    Returns:
        DuckDBPyConnection: The connection to the DuckDB database after the insertion.
    """
    create_table_if_not_exists(con)
    con.execute("INSERT INTO embeddings VALUES (?, ?, ?)", [text, model, embedding])
    return con


def create_table_if_not_exists(con) -> None:
    """
    Creates a table named `embeddings` if it doesn't already exist in the database.

    Args:
        con: The database connection object.

    Returns:
        None
    """
    con.from_query(
        f"CREATE TABLE IF NOT EXISTS embeddings (text VARCHAR, model VARCHAR, embedding {ARRAY_TYPE})"
    )


def is_key_in_table(con: DuckDBPyConnection, key: Tuple[str, str]) -> bool:
    """
    Check if a key exists in the embeddings table.

    Args:
        con (DuckDBPyConnection): The connection to the DuckDB database.
        key (Tuple[str, str]): The key to check in the format (text, model).

    Returns:
        bool: True if the key exists in the table, False otherwise.
    """
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
    """
    Returns a list of keys that exist in the specified table.

    Args:
        con (DuckDBPyConnection): The connection to the DuckDB database.
        keys (List[Tuple[str, str]]): The keys to check in the table.

    Returns:
        List[Tuple[str, str]]: A list of keys that exist in the table.
    """
    keys_in_table = []

    for key in keys:
        if is_key_in_table(con, key):
            keys_in_table.append(key)
    return keys_in_table


def load_pickle_cache(pickle_path: str) -> PickleCache:
    """
    Load a pickle cache from the given file path.

    Args:
        pickle_path (str): The path to the pickle file.

    Returns:
        PickleCache: The loaded pickle cache.

    """
    if os.path.exists(pickle_path):
        with open(pickle_path, "rb") as file:
            return pickle.load(file)
    return {}


def write_pickle_cache_to_duckdb(con: DuckDBPyConnection, pickle_path: str) -> None:
    """
    Writes the contents of a pickle cache to a DuckDB database.

    Args:
        con (DuckDBPyConnection): The connection to the DuckDB database.
        pickle_path (str): The path to the pickle cache file.

    Returns:
        None
    """
    cache = load_pickle_cache(pickle_path)
    create_table_if_not_exists(con)
    for key, value in cache.items():
        write_embedding_to_table(con, key[0], key[1], value)


# Function to save the cache to a file
def save_pickle_cache(cache: PickleCache, cache_path: str) -> None:
    """
    Save the given cache object as a pickle file.

    Args:
        cache (PickleCache): The cache object to be saved.
        cache_path (str): The path to save the pickle file.

    Returns:
        None
    """
    with open(cache_path, "wb") as file:
        pickle.dump(cache, file)


def get_embedding_from_table(con: DuckDBPyConnection, text: str, model: str) -> List[float]:
    """
    Retrieves the embedding from the 'embeddings' table based on the given text and model.

    Args:
        con (DuckDBPyConnection): The connection to the DuckDB database.
        text (str): The text to search for in the 'text' column of the table.
        model (str): The model to search for in the 'model' column of the table.

    Returns:
        List[float]: The embedding associated with the given text and model.

    Raises:
        ValueError: If the embedding for the given text and model is not found in the table.
    """
    result = con.execute(
        "SELECT embedding FROM embeddings WHERE text=? AND model=?", [text, model]
    ).fetchone()
    if result:
        return result[0]
    raise ValueError(f"Embedding for {text} with model {model} not found in table")
