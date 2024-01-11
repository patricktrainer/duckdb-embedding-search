from src.connection import open_connection
from src.operations import load_pickle_cache, write_pickle_cache_to_duckdb


if __name__ == "__main__":
    pickle_path = "data/embeddings_cache.pkl"
    pickle_cache = load_pickle_cache(pickle_path)

    with open_connection("hn_embeddings") as con:
        print("Writing embeddings to table... This may take a minute or two.\n")
        write_pickle_cache_to_duckdb(con, pickle_path)
        print("Done.\n")
        