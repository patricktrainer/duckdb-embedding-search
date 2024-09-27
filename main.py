from src.connection import open_connection
from src.embedding import get_similarity
from src.rag import rag_pipeline

if __name__ == "__main__":
    model = "text-embedding-ada-002"
    test_query = "What are the pros and cons of using older, stable libraries versus newer, frequently updated ones?"

    with open_connection("hn_embeddings") as con:
        # Example of using the original similarity search
        print("Original Similarity Search Results:")
        for result in get_similarity(con, test_query, model):
            print(
                f"""
                text: {result[0]}
                similarity: {result[1]}
                """
            )

        # Example of using the new RAG pipeline
        print("\nRAG Pipeline Response:")
        rag_response = rag_pipeline(con, test_query, model)
        print(rag_response)