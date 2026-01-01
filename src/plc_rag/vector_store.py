"""FAISS vector store helper with metadata persistence."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import faiss
import numpy as np

from .config import VectorStoreConfig
from .data_models import ChunkRecord, RetrievalResult

LOGGER = logging.getLogger(__name__)


class FaissVectorStore:
    """In-memory FAISS index with disk persistence."""

    def __init__(
        self,
        config: VectorStoreConfig,
        dim: int | None = None,
        recreate: bool = False,
    ):
        self.config = config
        self.storage_dir = Path(config.storage_dir)
        self.index_path = self.storage_dir / "index.faiss"
        self.meta_path = self.storage_dir / "metadata.jsonl"
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self._metadata: List[ChunkRecord] = []
        self.index: faiss.IndexFlatIP | None = None

        if self.index_path.exists() and not recreate:
            self._load_existing()
        else:
            if dim is None:
                raise ValueError("dim must be provided when creating a new vector store.")
            self.index = faiss.IndexFlatIP(dim)
            LOGGER.info("Initialized new FAISS index with dimension %s", dim)

    @property
    def dim(self) -> int:
        if self.index is None:
            raise RuntimeError("Index has not been initialized.")
        return self.index.d

    def _load_existing(self) -> None:
        self.index = faiss.read_index(str(self.index_path))
        self._metadata = []
        with self.meta_path.open("r", encoding="utf-8") as fp:
            for line in fp:
                data = json.loads(line)
                self._metadata.append(ChunkRecord(text=data["text"], metadata=data["metadata"]))
        LOGGER.info(
            "Loaded FAISS index from %s with %s vectors", self.index_path, len(self._metadata)
        )

    def add(self, embeddings: np.ndarray, chunks: Sequence[ChunkRecord]) -> None:
        if self.index is None:
            raise RuntimeError("Index is not initialized.")
        if embeddings.shape[0] != len(chunks):
            raise ValueError("Embeddings and chunk counts do not match.")

        embeddings = embeddings.astype("float32")
        self.index.add(embeddings)
        self._metadata.extend(chunks)
        LOGGER.info("Added %s vectors to store; total=%s", len(chunks), len(self._metadata))

    def save(self) -> None:
        if self.index is None:
            raise RuntimeError("Index is not initialized.")
        faiss.write_index(self.index, str(self.index_path))
        with self.meta_path.open("w", encoding="utf-8") as fp:
            for chunk in self._metadata:
                record = {"text": chunk.text, "metadata": chunk.metadata}
                fp.write(json.dumps(record, ensure_ascii=False) + "\n")
        LOGGER.info("Persisted FAISS index to %s", self.index_path)

    def search(self, query_embedding: np.ndarray, top_k: int) -> List[RetrievalResult]:
        if self.index is None:
            raise RuntimeError("Index is not initialized.")
        if query_embedding.ndim == 1:
            query_embedding = query_embedding[np.newaxis, :]
        scores, indices = self.index.search(query_embedding.astype("float32"), top_k)
        hits: List[RetrievalResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self._metadata[idx]
            hits.append(RetrievalResult(text=chunk.text, metadata=chunk.metadata, score=float(score)))
        return hits
