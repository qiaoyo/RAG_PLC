"""Utilities for chunking documents into retrieval-ready segments."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Iterable, List

from .data_models import ChunkRecord, DocumentRecord


def _window(sequence: List[str], size: int, step: int) -> Iterable[List[str]]:
    if size <= 0:
        raise ValueError("chunk size must be >= 1")
    for start in range(0, len(sequence), step):
        yield sequence[start : start + size]


@dataclass(slots=True)
class TextChunker:
    """Word-level chunker with overlap control."""

    chunk_size: int = 800
    chunk_overlap: int = 120
    min_chunk_size: int = 200

    def split(self, document: DocumentRecord) -> List[ChunkRecord]:
        words = document.content.split()
        if not words:
            return []

        step = max(self.chunk_size - self.chunk_overlap, 1)
        chunks: List[ChunkRecord] = []
        for idx, window in enumerate(_window(words, self.chunk_size, step)):
            if len(window) < self.min_chunk_size and idx != 0:
                continue

            text = " ".join(window)
            metadata = {
                **document.metadata,
                "chunk_index": idx,
                "num_words": len(window),
            }
            chunks.append(ChunkRecord(text=text, metadata=metadata))

        return chunks

    def split_corpus(self, documents: Iterable[DocumentRecord]) -> List[ChunkRecord]:
        return list(itertools.chain.from_iterable(self.split(doc) for doc in documents))
