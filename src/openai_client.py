from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

import os


def get_openai_client() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=key)
    return client
