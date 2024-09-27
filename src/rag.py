import openai
from typing import List, Tuple
from src.embedding import get_similarity
from src.connection import DuckDBPyConnection


def retrieve_relevant_documents(
    con: DuckDBPyConnection, query: str, model: str, top_k: int = 3
) -> List[Tuple[str, float]]:
    """
    Retrieve the most relevant documents based on the input query.

    Args:
        con (DuckDBPyConnection): The connection to the DuckDB database.
        query (str): The input query.
        model (str): The name of the embedding model to use.
        top_k (int): The number of top documents to retrieve.

    Returns:
        List[Tuple[str, float]]: A list of tuples containing the relevant documents and their similarity scores.
    """
    similarities = get_similarity(con, query, model)
    return similarities[:top_k]


def generate_response(query: str, relevant_docs: List[Tuple[str, float]]):
    """
    Generate a response using the retrieved documents and the original query.

    Args:
        query (str): The original input query.
        relevant_docs (List[Tuple[str, float]]): A list of relevant documents and their similarity scores.

    Returns:
        str: The generated response.
    """
    context = "\n".join([doc[0] for doc in relevant_docs])
    prompt = f"""Given the following context and query, generate a relevant response:

Context:
{context}

Query: {query}

Response:"""

    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": """
                You are a helpful assistant that generates responses based on given context.
                Your response should provide examples of code if relevant.
                Answer the user's question to the best of your ability as if you were a software developer.
                """
            },
            {"role": "user", "content": prompt},
        ],
    )

    return response.choices[0].message.content


def rag_pipeline(con: DuckDBPyConnection, query: str, model: str):
    """
    Execute the full RAG pipeline: retrieve relevant documents and generate a response.

    Args:
        con (DuckDBPyConnection): The connection to the DuckDB database.
        query (str): The input query.
        model (str): The name of the embedding model to use.

    Returns:
        str: The generated response.
    """
    relevant_docs = retrieve_relevant_documents(con, query, model)
    response = generate_response(query, relevant_docs)
    return response
