"""End-to-end RAG pipeline for PLC code generation and explanation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Sequence

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from .config import GeneratorConfig, RagRuntimeConfig
from .data_models import RetrievalResult
from .embeddings import EmbeddingBackend
from .vector_store import FaissVectorStore

LOGGER = logging.getLogger(__name__)


def _select_dtype(name: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return mapping.get(name.lower(), torch.bfloat16)


class PromptBuilder:
    """Prompt factory that stitches retrieved context with user questions."""

    template: str = (
        "你是一个资深PLC控制工程师，善于给出符合IEC 61131-3标准的结构化文本(ST)代码。\n"
        "根据以下知识库内容回答用户问题或生成示例代码。\n\n"
        "【上下文】\n{context}\n\n"
        "【问题】\n{question}\n\n"
        "请给出详尽的分析以及结构化文本代码，并解释关键步骤。"
    )

    def build(self, question: str, contexts: Sequence[RetrievalResult]) -> str:
        context_text = "\n---\n".join(
            f"来源: {hit.metadata.get('source', 'unknown')} (score={hit.score:.3f})\n{hit.text}"
            for hit in contexts
        )
        return self.template.format(context=context_text, question=question.strip())


class LLMGenerator:
    """Thin wrapper for Hugging Face causal LMs."""

    def __init__(self, config: GeneratorConfig):
        self.config = config
        quantization_config = (
            BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=_select_dtype(config.torch_dtype))
            if config.use_4bit
            else None
        )
        LOGGER.info("Loading LLM %s (4bit=%s)", config.model_name, config.use_4bit)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            device_map=config.device_map,
            torch_dtype=_select_dtype(config.torch_dtype),
            quantization_config=quantization_config,
            trust_remote_code=True,
        )

    def generate(self, prompt: str, max_new_tokens: int, temperature: float) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
        )[0]
        return self.tokenizer.decode(output_ids, skip_special_tokens=True)


@dataclass(slots=True)
class RagComponents:
    embedder: EmbeddingBackend
    vector_store: FaissVectorStore
    generator: LLMGenerator
    prompt_builder: PromptBuilder


class RagPipeline:
    """Coordinates retrieval and generation."""

    def __init__(self, components: RagComponents, rag_config: RagRuntimeConfig):
        self.components = components
        self.rag_config = rag_config

    def retrieve(self, query: str) -> List[RetrievalResult]:
        query_embedding = self.components.embedder.encode([query])
        return self.components.vector_store.search(
            query_embedding, self.rag_config.vector_store.top_k
        )

    def run(self, question: str) -> dict:
        hits = self.retrieve(question)
        prompt = self.components.prompt_builder.build(question, hits)
        answer = self.components.generator.generate(
            prompt,
            max_new_tokens=self.rag_config.generator.max_new_tokens,
            temperature=self.rag_config.generator.temperature,
        )
        return {
            "question": question,
            "answer": answer,
            "context": hits,
        }
