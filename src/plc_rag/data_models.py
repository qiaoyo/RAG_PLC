"""Lightweight dataclasses shared across the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass(slots=True)
class DocumentRecord:
    """Represents a full document before chunking."""

    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ChunkRecord:
    """Represents a chunked segment ready for embedding."""

    text: str
    metadata: Dict[str, Any]


@dataclass(slots=True)
class RetrievalResult:
    """Single retrieval hit result."""

    text: str
    metadata: Dict[str, Any]
    score: float
