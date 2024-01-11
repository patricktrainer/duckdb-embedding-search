from dotenv import load_dotenv
from typing import List
from openai import OpenAI

load_dotenv()

import os


def get_openai_client() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=key)
    return client

def create_embedding(text: str, model: str = "text-embedding-ada-002", **kwargs) -> List[float]:
    # replace newlines, which can negatively affect performance.
    try: 
        client = get_openai_client()
    except Exception as e:
        print(e)
        return []
    
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text], model=model, **kwargs)
    return response.data[0].embedding

