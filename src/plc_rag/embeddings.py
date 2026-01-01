"""Embedding utilities built on top of sentence-transformers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from .config import EmbeddingConfig
from .data_models import ChunkRecord

LOGGER = logging.getLogger(__name__)


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():  # pragma: no cover - mac support
        return "mps"
    return "cpu"


class EmbeddingBackend:
    """Thin wrapper around SentenceTransformer with batching support."""

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        device = config.device or default_device()
        self.model = SentenceTransformer(config.model_name, device=device)
        LOGGER.info("Loaded embedding model %s on %s", config.model_name, device)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize,
            show_progress_bar=True,
        )

    def embed_chunks(self, chunks: Sequence[ChunkRecord]) -> np.ndarray:
        texts = [chunk.text for chunk in chunks]
        return self.encode(texts)
