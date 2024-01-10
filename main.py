from src.connection import open_connection
from src.embedding import get_similarity

if __name__ == "__main__":
    # existing main functionality
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
