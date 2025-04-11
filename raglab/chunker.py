from pathlib import Path
from typing import List
import tiktoken


class Chunker:
    def __init__(self, model_name: str = "cl100k_base"):
        """
        Initializes the Chunker with a tokenizer compatible with OpenAI models.
        """
        self.tokenizer = tiktoken.get_encoding(model_name)

    def chunk_text(self, text: str, max_tokens: int = 200, overlap: int = 20) -> List[str]:
        """
        Splits the given text into token-aware chunks.
        """
        tokens = self.tokenizer.encode(text)
        chunks = []
        start = 0

        while start < len(tokens):
            end = start + max_tokens
            chunk = tokens[start:end]
            decoded = self.tokenizer.decode(chunk)
            chunks.append(decoded)
            start += max_tokens - overlap

        return chunks

    def chunk_file(self, file_path: str, max_tokens: int = 200, overlap: int = 20) -> List[str]:
        """
        Loads text from a file and returns token-aware chunks.
        """
        text = Path(file_path).read_text(encoding="utf-8")
        return self.chunk_text(text, max_tokens=max_tokens, overlap=overlap)
