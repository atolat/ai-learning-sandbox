import os
import openai
from typing import List
from dotenv import load_dotenv

load_dotenv()

class Embedder:
    def __init__(self, model_name: str = "text-embedding-3-small"):
        self.model_name = model_name
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not set in environment.")
        self.client = openai.OpenAI(api_key=self.api_key)

    def embed(self, texts: List[str]) -> List[List[float]]:
        response = self.client.embeddings.create(
            model=self.model_name,
            input=texts
        )
        return [embedding.embedding for embedding in response.data]
