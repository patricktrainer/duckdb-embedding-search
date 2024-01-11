from typing import List
from .operations import write_embedding_to_table, list_keys_in_table
import src.openai_client as openai_client
from .connection import DuckDBPyConnection
from src.utils import load_pickle_cache, save_pickle_cache


# Function to get embeddings, using the cache
def pickle_embeddings(
    texts: List[str], model: str, pickle_path: str
) -> List[List[float]]:
    embeddings = []
    pickle_cache = load_pickle_cache(pickle_path)

    for text in texts:
        key = (text, model)
        if key not in pickle_cache:
            pickle_cache[key] = openai_client.create_embedding(text, model=model)
        embeddings.append(pickle_cache[key])
    save_pickle_cache(pickle_cache, pickle_path)
    return embeddings


# Function to get embeddings, using the cache
def duckdb_embeddings(
    texts: List[str], model: str, con: DuckDBPyConnection
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
                embedding = openai_client.create_embedding(text, model)
                # and write it to the table
                write_embedding_to_table(con, text, model, embedding)
                embeddings.append(embedding)
        else:
            print("Embedding not found in table")
            print("Creating new embedding")
            # if not, create it
            embedding = openai_client.create_embedding(text, model)
            # and write it to the table
            write_embedding_to_table(con, text, model, embedding)
            embeddings.append(embedding)
    return embeddings


def cosine_similarity(con: DuckDBPyConnection, l1, l2) -> float:
    return con.execute(f"SELECT list_cosine_similarity({l1}, {l2})").fetchall()[0][0]


def get_similarity(
    con: DuckDBPyConnection, text: str, model: str
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

    embedding = duckdb_embeddings([text], model, con)[0]
    result = con.execute(sql, [text, embedding]).fetchall()
    return result
