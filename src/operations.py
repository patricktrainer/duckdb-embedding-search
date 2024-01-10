from typing import List, Tuple
from .connection import DuckDBPyConnection
from duckdb.typing import DuckDBPyType

ARRAY_TYPE = DuckDBPyType(list[float]) # type: ignore


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
