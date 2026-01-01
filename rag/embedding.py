from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str, device: str = "cuda"):
        self.model = SentenceTransformer(model_name, device=device)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: List[str], batch_size: int = 32, normalize: bool = True) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            show_progress_bar=True,
        )
        return embeddings.astype("float32")
