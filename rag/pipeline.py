import argparse
import logging
from pathlib import Path
from typing import Dict, List

from generator import LLMGenerator
from retriever import Retriever
from settings import (
    DEFAULT_EMBED_MODEL,
    DEFAULT_GEN_MODEL,
    DEFAULT_TOKENIZER_PATH,
    INDEX_PATH,
    METADATA_PATH,
)
from utils import extract_code_block
from vector_store import FaissStore
from embedding import Embedder

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

SYSTEM_PROMPT = (
    "你是工业控制与PLC编程专家，擅长IEC 61131-3 标准的 ST 语言。"
    "请基于提供的上下文生成可直接使用的PLC代码，保证变量声明完整、逻辑自洽。"
    "如果上下文不足，请给出最佳实践的安全实现。"
)


class RAGPipeline:
    def __init__(self, retriever: Retriever, generator: LLMGenerator, top_k: int = 5):
        self.retriever = retriever
        self.generator = generator
        self.top_k = top_k

    def build_prompt(self, instruction: str, user_input: str, contexts: List[Dict]) -> str:
        context_block = "\n\n".join(
            [f"[来源:{ctx.get('source')}] {ctx.get('text')}" for ctx in contexts]
        )
        return (
            f"{SYSTEM_PROMPT}\n\n"
            f"检索到的材料:\n{context_block}\n\n"
            f"用户指令: {instruction}\n"
            f"补充输入: {user_input or '无'}\n"
            "请输出符合要求的PLC代码，仅给出代码本身。"
        )

    def generate(self, instruction: str, user_input: str = "", top_k: int | None = None) -> Dict:
        k = top_k or self.top_k
        hits = self.retriever.search(f"{instruction}\n{user_input}", top_k=k)
        contexts = [hit[1] for hit in hits]
        prompt = self.build_prompt(instruction, user_input, contexts)
        raw_output = self.generator.generate(prompt)
        code = extract_code_block(raw_output)
        return {
            "prompt": prompt,
            "raw_output": raw_output,
            "code": code,
            "contexts": contexts,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single-query PLC RAG generation")
    parser.add_argument("--query", type=str, required=True, help="用户指令")
    parser.add_argument("--input", type=str, default="", help="可选补充输入")
    parser.add_argument("--top-k", type=int, default=5, help="检索条数")
    parser.add_argument("--index-path", type=Path, default=INDEX_PATH, help="FAISS索引路径")
    parser.add_argument("--meta-path", type=Path, default=METADATA_PATH, help="索引元数据")
    parser.add_argument("--embed-model", type=str, default=DEFAULT_EMBED_MODEL, help="检索使用的嵌入模型")
    parser.add_argument("--gen-model", type=str, default=DEFAULT_GEN_MODEL, help="生成模型路径或名称")
    parser.add_argument("--tokenizer", type=str, default=DEFAULT_TOKENIZER_PATH, help="分词器路径")
    return parser.parse_args()


def build_pipeline(args: argparse.Namespace) -> RAGPipeline:
    store = FaissStore.load(args.index_path, args.meta_path)
    embedder = Embedder(model_name=args.embed_model)
    retriever = Retriever(store=store, embedder=embedder)
    generator = LLMGenerator(
        model_name_or_path=args.gen_model,
        tokenizer_path=args.tokenizer,
    )
    return RAGPipeline(retriever=retriever, generator=generator, top_k=args.top_k)


if __name__ == "__main__":
    args = parse_args()
    pipeline = build_pipeline(args)
    result = pipeline.generate(instruction=args.query, user_input=args.input)
    logger.info("生成代码:\n%s", result["code"])
