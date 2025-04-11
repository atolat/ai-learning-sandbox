from typing import List, Tuple
import numpy as np

class VectorStore:
    def __init__(self):
        self.texts: List[str] = []
        self.vectors: List[List[float]] = []

    def add(self, text: str, embedding: List[float]) -> None:
        self.texts.append(text)
        self.vectors.append(embedding)

    def search(self, query: List[float], k: int = 3) -> List[Tuple[str, float]]:
        """
        Returns top-k (text, similarity) tuples
        """
        similarities = [self.cosine_similarity(query, vector) for vector in self.vectors]
        top_indices = np.argsort(similarities)[::-1][:k]
        return [(self.texts[i], similarities[i]) for i in top_indices]

    def cosine_similarity(self, query_vector: List[float], vector: List[float]) -> float:
        """
        Computes cosine similarity between two vectors
        """
        return np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
