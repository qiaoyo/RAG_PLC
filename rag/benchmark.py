import argparse
import json
import logging
from pathlib import Path

from data_models import PLCExample
from evaluation import BenchmarkRunner, LLMJudge
from ingestion import load_plc_dataset
from pipeline import RAGPipeline
from retriever import Retriever
from settings import (
    DATASET_PATH,
    DEFAULT_EMBED_MODEL,
    DEFAULT_GEN_MODEL,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_TOKENIZER_PATH,
    INDEX_PATH,
    METADATA_PATH,
)
from embedding import Embedder
from vector_store import FaissStore
from generator import LLMGenerator

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PLC RAG benchmark with LLM-as-judge")
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH, help="Path to PLC JSON dataset")
    parser.add_argument("--index-path", type=Path, default=INDEX_PATH, help="FAISS index path")
    parser.add_argument("--meta-path", type=Path, default=METADATA_PATH, help="FAISS metadata path")
    parser.add_argument("--results", type=Path, default=Path("rag/artifacts/bench_results.json"), help="Where to save results")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on evaluated samples")
    parser.add_argument("--embed-model", type=str, default=DEFAULT_EMBED_MODEL, help="Embedding model for retrieval")
    parser.add_argument("--gen-model", type=str, default=DEFAULT_GEN_MODEL, help="Generation model")
    parser.add_argument("--judge-model", type=str, default=DEFAULT_JUDGE_MODEL, help="Judge model")
    parser.add_argument("--tokenizer", type=str, default=DEFAULT_TOKENIZER_PATH, help="Tokenizer path")
    parser.add_argument("--gen-device", type=str, default="cuda:1", help="CUDA device for generation model, e.g., cuda:0")
    parser.add_argument("--judge-device", type=str, default="cuda:0", help="CUDA device for judge model, e.g., cuda:1")
    parser.add_argument("--gen-8bit", action="store_true", help="Load generation model in 8-bit (bitsandbytes)")
    parser.add_argument("--judge-8bit", action="store_true", help="Load judge model in 8-bit (bitsandbytes)")
    parser.add_argument("--gen-4bit", action="store_true", help="Load generation model in 4-bit (bitsandbytes)")
    parser.add_argument("--judge-4bit", action="store_true", help="Load judge model in 4-bit (bitsandbytes)")
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
    return RAGPipeline(retriever=retriever, generator=generator)


def main() -> None:
    args = parse_args()
    dataset = load_plc_dataset(args.dataset)
    pipeline = build_pipeline(args)
    judge = LLMJudge(
        model_name=args.judge_model,
        tokenizer_path=args.tokenizer,
        device=args.judge_device,
        load_in_8bit=args.judge_8bit,
        load_in_4bit=args.judge_4bit,
    )
    runner = BenchmarkRunner(pipeline=pipeline, judge=judge, dataset=dataset)
    metrics = runner.run(limit=args.limit)
    args.results.parent.mkdir(parents=True, exist_ok=True)
    with open(args.results, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    logger.info("Benchmark finished. Mean score %.3f | Pass rate %.2f%%", metrics["mean_score"], metrics["pass_rate"] * 100)
    logger.info("Results saved to %s", args.results)


if __name__ == "__main__":
    main()
