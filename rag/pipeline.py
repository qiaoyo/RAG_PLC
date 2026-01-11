import argparse
import logging
from pathlib import Path
from typing import Dict, List

import sys

# Ensure project root on sys.path when run as script
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
from plc_eval.dataset_utils import load_dataset, load_splits, split_dataset, save_splits

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
    parser.add_argument("--query", type=str, help="用户指令（单条模式）")
    parser.add_argument("--input", type=str, default="", help="可选补充输入（单条模式）")
    parser.add_argument("--top-k", type=int, default=5, help="检索条数")
    parser.add_argument("--index-path", type=Path, default=INDEX_PATH, help="FAISS索引路径")
    parser.add_argument("--meta-path", type=Path, default=METADATA_PATH, help="索引元数据")
    parser.add_argument("--embed-model", type=str, default=DEFAULT_EMBED_MODEL, help="检索使用的嵌入模型")
    parser.add_argument("--gen-model", type=str, default=DEFAULT_GEN_MODEL, help="生成模型路径或名称")
    parser.add_argument("--tokenizer", type=str, default=DEFAULT_TOKENIZER_PATH, help="分词器路径")
    parser.add_argument("--gen-device", type=str, default="cuda:1", help="生成模型使用的设备，如 cuda:0")
    parser.add_argument("--gen-8bit", action="store_true", help="以8bit量化加载生成模型")
    parser.add_argument("--gen-4bit", action="store_true", help="以4bit量化加载生成模型")
    parser.add_argument("--batch-split", type=Path, help="可选：plc_eval/splits.json 路径，启用批量生成")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="批量生成使用的数据切分")
    parser.add_argument("--dataset", type=Path, default=Path("data/data_full/train_full_normalized.json"), help="数据集路径")
    parser.add_argument("--seed", type=int, default=42, help="批量分割使用的随机种子")
    parser.add_argument("--limit", type=int, default=None, help="批量生成条数上限")
    parser.add_argument("--output-dir", type=Path, default=Path("data/result"), help="批量输出根目录")
    return parser.parse_args()


def build_pipeline(args: argparse.Namespace) -> RAGPipeline:
    store = FaissStore.load(args.index_path, args.meta_path)
    embedder = Embedder(model_name=args.embed_model)
    retriever = Retriever(store=store, embedder=embedder)
    generator = LLMGenerator(
        model_name_or_path=args.gen_model,
        tokenizer_path=args.tokenizer,
        device=args.gen_device,
        load_in_8bit=args.gen_8bit,
        load_in_4bit=args.gen_4bit,
    )
    return RAGPipeline(retriever=retriever, generator=generator, top_k=args.top_k)


if __name__ == "__main__":
    args = parse_args()
    pipeline = build_pipeline(args)
    if args.batch_split:
        split_file = args.batch_split
        if split_file.exists():
            splits = load_splits(split_file)
        else:
            data = load_dataset(args.dataset)
            train, val, test = split_dataset(data, seed=args.seed)
            splits = {"train": train, "val": val, "test": test}
            split_file.parent.mkdir(parents=True, exist_ok=True)
            save_splits(splits, split_file)
        samples = splits.get(args.split, [])
        subset = samples[: args.limit] if args.limit else samples
        out_root = args.output_dir / args.split
        out_root.mkdir(parents=True, exist_ok=True)
        for idx, item in enumerate(subset, start=1):
            out_dir = out_root / f"{idx:03d}"
            out_dir.mkdir(parents=True, exist_ok=True)
            result = pipeline.generate(instruction=item["instruction"], user_input=item.get("input", ""))
            raw_path = out_dir / "raw_code.txt"
            raw_code_path = out_dir / "raw_code.st"
            raw_path.write_text(result["raw_output"], encoding="utf-8")
            raw_code_path.write_text(result["code"], encoding="utf-8")
            logger.info("Saved #%03d to %s", idx, out_dir)
    else:
        result = pipeline.generate(instruction=args.query, user_input=args.input)
        logger.info("生成代码:\n%s", result["code"])
