import os
import pickle
from typing import List, Tuple, Dict
import duckdb
from duckdb import DuckDBPyConnection
from duckdb.typing import DuckDBPyType
from openai import OpenAI

# Cache for storing embeddings
client = OpenAI(api_key="")
EmbeddingCache = Dict[Tuple[str, str], List[float]]
ARRAY_TYPE = DuckDBPyType(list[float])


def open_connection(dbname=None) -> duckdb.DuckDBPyConnection:
    """
    Connects to a local duckdb database and returns a connection object.

    Returns:
        duckdb.DuckDBPyConnection: A connection object to the local database.
    """
    if dbname:
        return duckdb.connect(f"{dbname}.db")
    else:
        return duckdb.connect(":memory:")


def load_extension(
    con: duckdb.DuckDBPyConnection, extension: str
) -> DuckDBPyConnection:
    try:
        con.install_extension(extension)
        con.load_extension(extension)
    except:
        print(f"Could not load extension {extension}")
        pass
    return con


def write_embedding_to_table(
    con: duckdb.DuckDBPyConnection, text: str, model: str, embedding: List[float]
) -> DuckDBPyConnection:
    create_table_if_not_exists(con)
    con.execute("INSERT INTO embeddings VALUES (?, ?, ?)", [text, model, embedding])
    return con


def create_table_if_not_exists(con) -> None:
    con.from_query(
        f"CREATE TABLE IF NOT EXISTS embeddings (text VARCHAR, model VARCHAR, embedding {ARRAY_TYPE})"
    )


def is_key_in_table(con: duckdb.DuckDBPyConnection, key: Tuple[str, str]) -> bool:
    create_table_if_not_exists(con)
    result = con.execute(
        "SELECT EXISTS(SELECT * FROM embeddings WHERE text=? AND model=?)",
        [key[0], key[1]],
    ).fetchone()
    if result:
        return result[0]
    return False


# Function to get embeddings, using the cache
def get_embeddings_with_cache(
    texts: List[str], model: str, con: duckdb.DuckDBPyConnection
) -> List[List[float]]:
    embeddings = []
    for text in texts:
        keys = [(text, model) for text in texts]
        # check to see if embedding is in duckdb table
        if list_keys_in_table(con, keys):
            print("Found embedding in table")
            # if so, retrieve it
            result = con.execute(
                "SELECT embedding FROM embeddings WHERE text=? AND model=?",
                [text, model],
            ).fetchone()
            if result is not None:
                embedding = result[0]
                embeddings.append(embedding)
            else:
                print("Embedding not found in table")
                print("Creating new embedding")
                # if not, create it
                embedding = create_embedding(text, model)
                # and write it to the table
                write_embedding_to_table(con, text, model, embedding)
                embeddings.append(embedding)
        else:
            print("Embedding not found in table")
            print("Creating new embedding")
            # if not, create it
            embedding = create_embedding(text, model)
            # and write it to the table
            write_embedding_to_table(con, text, model, embedding)
            embeddings.append(embedding)
    return embeddings


def create_embedding(
    text: str, model: str = "text-embedding-ada-002", **kwargs
) -> List[float]:
    # replace newlines, which can negatively affect performance.
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text], model=model, **kwargs)
    return response.data[0].embedding


def list_keys_in_table(
    con: duckdb.DuckDBPyConnection, keys: List[Tuple[str, str]]
) -> list[tuple[str, str]]:
    keys_in_table = []

    for key in keys:
        if is_key_in_table(con, key):
            keys_in_table.append(key)
    return keys_in_table


def load_pickle_cache(pickle_path: str) -> EmbeddingCache:
    if os.path.exists(pickle_path):
        with open(pickle_path, "rb") as file:
            return pickle.load(file)
    return {}


def cosine_similarity(con: duckdb.DuckDBPyConnection, l1, l2) -> float:
    return con.execute(f"SELECT list_cosine_similarity({l1}, {l2})").fetchall()[0][0]


def get_similarity(
    con: duckdb.DuckDBPyConnection, text: str, model: str
) -> list[tuple[str, float]]:
    # calculate the cosine similarity between the test_text and all other texts in the table
    sql = """
        WITH q1 AS (
            SELECT 
                ? as text, 
                ?::DOUBLE[] AS embedding
        ),

        q2 AS (
            select 
                distinct text, 
                embedding::DOUBLE[] as embedding
            from embeddings
        )

        SELECT 
            b.text, 
            list_cosine_similarity(a.embedding::DOUBLE[], b.embedding::DOUBLE[]) AS similarity
        FROM q1 a
        join q2 b on a.text != b.text
        ORDER BY similarity DESC
        LIMIT 10
        """

    embedding = get_embeddings_with_cache([text], model, con)[0]
    result = con.execute(sql, [text, embedding]).fetchall()
    return result


def write_pickle_cache_to_duckdb(
    con: duckdb.DuckDBPyConnection, pickle_path: str
) -> None:
    cache = load_pickle_cache(pickle_path)
    create_table_if_not_exists(con)
    for key, value in cache.items():
        write_embedding_to_table(con, key[0], key[1], value)


if __name__ == "__main__":
    con = open_connection("embeddings")
    model = "text-embedding-ada-002"

    test_text = """
    One thing I’ve noticed is that many engineers, when they’re looking for a library on Github, they check the last commit time. They think that the more recent the last commit is, the better supported the library is.
But what about an archived project that does exactly what you need it to do, has 0 bugs, and has been stable for years? That’s like finding a hidden gem in a thrift store!
Most engineers I see nowadays will automatically discard a library that is not "constantly" updated... Implying it's a good thing :)"""

    sim = get_similarity(con, test_text, model)
    for res in sim:
        print(
            f"""
            text: {res[0]}
            similarity: {res[1]}
            """
        )
