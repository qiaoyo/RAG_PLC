from typing import List, Tuple

from .embedding import Embedder
from .vector_store import FaissStore


class Retriever:
    def __init__(self, store: FaissStore, embedder: Embedder):
        self.store = store
        self.embedder = embedder

    def search(self, query: str, top_k: int = 5) -> List[Tuple[float, dict]]:
        query_embedding = self.embedder.encode([query])[0]
        return self.store.search(query_embedding, top_k=top_k)
