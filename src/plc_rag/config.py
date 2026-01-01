"""Centralized configuration dataclasses used across the PLC RAG stack."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass(slots=True)
class CorpusConfig:
    """Paths and chunking parameters for building the retrieval corpus."""

    instruction_corpus: Path = Path("data/data_sample/train_001.json")
    books_dir: Path = Path("books")
    chunk_size: int = 800
    chunk_overlap: int = 120
    min_chunk_size: int = 200
    limit_records: Optional[int] = None
    limit_book_pages: Optional[int] = None
    output_chunks_file: Path = Path("data/processed/chunks.jsonl")


@dataclass(slots=True)
class EmbeddingConfig:
    """Embedding model configuration."""

    model_name: str = "BAAI/bge-large-zh-v1.5"
    device: Optional[str] = None
    batch_size: int = 32
    normalize: bool = True


@dataclass(slots=True)
class VectorStoreConfig:
    """FAISS vector store configuration."""

    storage_dir: Path = Path("data/vector_store")
    top_k: int = 5


@dataclass(slots=True)
class GeneratorConfig:
    """LLM generation configuration used by the RAG pipeline."""

    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    max_new_tokens: int = 512
    temperature: float = 0.2
    device_map: str = "auto"
    use_4bit: bool = True
    torch_dtype: str = "bfloat16"


@dataclass(slots=True)
class LoraFineTuneConfig:
    """Parameters for LoRA-based supervised fine-tuning."""

    base_model: str = "Qwen/Qwen2.5-7B-Instruct"
    dataset_path: Path = Path("data/data_sample/train_001.json")
    output_dir: Path = Path("artifacts/llm_lora")
    micro_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    num_epochs: int = 3
    max_seq_length: int = 2048
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    warmup_steps: int = 100
    save_steps: int = 200
    logging_steps: int = 20
    fp16: bool = False
    bf16: bool = True
    seed: int = 42


@dataclass(slots=True)
class EmbeddingFineTuneConfig:
    """Sentence-transformer fine-tuning parameters."""

    base_model: str = "BAAI/bge-large-zh-v1.5"
    dataset_path: Path = Path("data/data_sample/train_001.json")
    output_dir: Path = Path("artifacts/embedding_model")
    batch_size: int = 32
    num_epochs: int = 2
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    max_train_samples: Optional[int] = 5000


@dataclass(slots=True)
class RagRuntimeConfig:
    """Aggregated configuration used by the CLI."""

    corpus: CorpusConfig = field(default_factory=CorpusConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
