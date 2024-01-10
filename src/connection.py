import duckdb
from duckdb import DuckDBPyConnection

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
