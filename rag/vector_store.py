import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import faiss
import numpy as np

from utils import ensure_dir, flatten_metadata_items


class FaissStore:
    def __init__(
        self,
        dimension: int,
        index: faiss.Index = None,
        documents: List[Dict[str, Any]] = None,
        index_path: Path | str = None,
        metadata_path: Path | str = None,
    ):
        self.dimension = dimension
        self.index = index or faiss.IndexFlatIP(dimension)
        self.documents = documents or []
        self.index_path = Path(index_path) if index_path else None
        self.metadata_path = Path(metadata_path) if metadata_path else None

    def add_embeddings(self, embeddings: np.ndarray, documents: List[Dict[str, Any]]) -> None:
        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension {embeddings.shape[1]} != store dim {self.dimension}")
        self.index.add(embeddings)
        self.documents.extend(flatten_metadata_items(documents))

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[float, Dict[str, Any]]]:
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        scores, indices = self.index.search(query_embedding, top_k)
        hits: List[Tuple[float, Dict[str, Any]]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            hits.append((float(score), self.documents[idx]))
        return hits

    def save(self, index_path: Path, metadata_path: Path) -> None:
        ensure_dir(index_path.parent)
        ensure_dir(metadata_path.parent)
        faiss.write_index(self.index, str(index_path))
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, index_path: Path, metadata_path: Path) -> "FaissStore":
        index = faiss.read_index(str(index_path))
        with open(metadata_path, "r", encoding="utf-8") as f:
            documents = json.load(f)
        return cls(dimension=index.d, index=index, documents=documents, index_path=index_path, metadata_path=metadata_path)
