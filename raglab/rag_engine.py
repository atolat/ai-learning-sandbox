from typing import List
from raglab.embedding import Embedder
from raglab.vector_store import VectorStore
import openai

class RAGEngine:
    def __init__(self, embedder: Embedder, vector_store: VectorStore, llm_model: str = "gpt-3.5-turbo"):
        self.embedder = embedder
        self.vector_store = vector_store
        self.llm_model = llm_model
        self.client = openai.OpenAI()

    def query(self, question: str, k: int = 3) -> str:
        # 1. Embed the question
        query_vector = self.embedder.embed([question])[0]

        # 2. Retrieve top-k most relevant chunks
        top_chunks_with_scores = self.vector_store.search(query_vector, k=k)
        top_chunks = [text for text, _ in top_chunks_with_scores]

        # 3. Build the prompt
        prompt = self._build_prompt(question, top_chunks)

        # 4. Send prompt to the LLM and get a response
        return self._call_llm(prompt)

    def _build_prompt(self, question: str, chunks: List[str]) -> str:
        context = "\n\n".join(chunks)
        prompt = f"""You are a helpful assistant.

Here is some context:

{context}

Now answer the following question:

{question}

Answer:"""
        return prompt

    def _call_llm(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
