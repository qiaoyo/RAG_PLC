"""Fine-tune sentence-transformer style embedding models."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import List

from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader

from .config import EmbeddingFineTuneConfig

LOGGER = logging.getLogger(__name__)


def _load_examples(dataset_path: Path, max_samples: int | None) -> List[InputExample]:
    with Path(dataset_path).open("r", encoding="utf-8") as fp:
        data = json.load(fp)

    examples: List[InputExample] = []
    for row in data[:max_samples or len(data)]:
        instruction = row.get("instruction", "")
        in_text = row.get("input", "")
        description = row.get("description", "")
        answer = row.get("output", "")
        text_a = "\n".join(filter(None, [instruction, in_text, description]))
        text_b = answer
        examples.append(InputExample(texts=[text_a, text_b], label=1.0))
    return examples


def train_embedding_model(config: EmbeddingFineTuneConfig) -> None:
    """Fine-tune a dual-encoder using cosine similarity loss."""

    examples = _load_examples(config.dataset_path, config.max_train_samples)
    if not examples:
        raise ValueError("No training samples found for embedding fine-tuning.")

    model = SentenceTransformer(config.base_model)
    train_loader = DataLoader(examples, shuffle=True, batch_size=config.batch_size)
    loss = losses.CosineSimilarityLoss(model)
    warmup_steps = math.ceil(len(train_loader) * config.num_epochs * config.warmup_ratio)

    LOGGER.info(
        "Starting embedding fine-tune: samples=%s epochs=%s lr=%s",
        len(examples),
        config.num_epochs,
        config.learning_rate,
    )

    model.fit(
        train_objectives=[(train_loader, loss)],
        epochs=config.num_epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": config.learning_rate},
        output_path=str(config.output_dir),
        show_progress_bar=True,
    )

    LOGGER.info("Embedding model saved to %s", config.output_dir)
