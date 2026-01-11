import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

from plc_eval.code_score import evaluate_candidate
from plc_eval.dataset_utils import load_dataset, load_splits, save_splits, split_dataset
from plc_eval.st_wrapper import wrap_st_code

# Optional: for logic correctness via LLM
try:
    from rag.evaluation import LLMJudge
except Exception:  # noqa: BLE001
    LLMJudge = None


def maybe_build_judge(args) -> Optional[object]:
    if args.logic_mode == "llm":
        if LLMJudge is None:
            raise RuntimeError("rag.evaluation.LLMJudge not available; set logic_mode=similarity or install deps.")
        return LLMJudge(
            model_name=args.judge_model,
            tokenizer_path=args.tokenizer,
            provider=args.judge_provider,
            device=args.judge_device,
            tensor_parallel_size=args.judge_tp,
            max_model_len=args.judge_max_len,
            load_in_8bit=args.judge_8bit,
            load_in_4bit=args.judge_4bit,
        )
    return None


def get_split(dataset_path: Path, split_file: Path, split_name: str, seed: int) -> List[Dict]:
    if split_file.exists():
        splits = load_splits(split_file)
    else:
        data = load_dataset(dataset_path)
        train, val, test = split_dataset(data, train_ratio=0.8, val_ratio=0.1, seed=seed)
        splits = {"train": train, "val": val, "test": test}
        split_file.parent.mkdir(parents=True, exist_ok=True)
        save_splits(splits, split_file)
    if split_name not in splits:
        raise ValueError(f"Unknown split {split_name}")
    return splits[split_name]


def evaluate_dataset(
    samples: List[Dict],
    main_c_path: Path,
    limit: Optional[int],
    iec2c_flags: str,
    logic_mode: str,
    judge,
    candidate_provider=None,
) -> Dict:
    subset = samples[:limit] if limit else samples
    results: List[Dict] = []
    for item in subset:
        if candidate_provider:
            candidate_code = candidate_provider(item)
        else:
            candidate_code = item["output"]
        reference_code = item["output"]
        result = evaluate_candidate(
            candidate_code=candidate_code,
            reference_code=reference_code,
            main_c_path=main_c_path,
            iec2c_flags=iec2c_flags,
            judge=judge if logic_mode == "llm" else None,
        )
        result["instruction"] = item.get("instruction", "")
        result["input"] = item.get("input", "")
        results.append(result)
    mean_score = sum(r["score"] for r in results) / len(results) if results else 0.0
    return {"mean_score": mean_score, "results": results}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PLC benchmark runner (compile/static/runtime/logic)")
    parser.add_argument("--dataset", type=Path, default=Path("data/data_full/train_full_normalized.json"))
    parser.add_argument("--split-file", type=Path, default=Path("plc_eval/splits.json"))
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--main-c", type=Path, default=Path("plc_eval/main.c"))
    parser.add_argument("--iec2c-flags", type=str, default="")
    parser.add_argument("--logic-mode", type=str, default="similarity", choices=["similarity", "llm"])
    parser.add_argument("--results", type=Path, default=Path("plc_eval/benchmark_results.json"))
    # LLM judge options
    parser.add_argument("--judge-model", type=str, default="Qwen2.5-7B-Instruct")
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--judge-provider", type=str, default="hf", choices=["hf", "vllm"])
    parser.add_argument("--judge-device", type=str, default="cuda:1")
    parser.add_argument("--judge-tp", type=int, default=1)
    parser.add_argument("--judge-max-len", type=int, default=8192)
    parser.add_argument("--judge-8bit", action="store_true")
    parser.add_argument("--judge-4bit", action="store_true")
    # RAG generation options
    parser.add_argument("--use-rag", action="store_true", help="Use RAG pipeline to generate candidate code")
    parser.add_argument("--index-path", type=Path, default=Path("rag/artifacts/plc_faiss.index"))
    parser.add_argument("--meta-path", type=Path, default=Path("rag/artifacts/plc_faiss_meta.json"))
    parser.add_argument("--embed-model", type=str, default="BAAI/bge-large-zh-v1.5")
    parser.add_argument("--gen-model", type=str, default="Qwen2.5-Coder-14B-Instruct")
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--gen-top-k", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = get_split(args.dataset, args.split_file, args.split, args.seed)
    judge = maybe_build_judge(args) if args.logic_mode == "llm" else None
    candidate_provider = None
    if args.use_rag:
        from rag.embedding import Embedder
        from rag.generator import LLMGenerator
        from rag.pipeline import RAGPipeline
        from rag.retriever import Retriever
        from rag.vector_store import FaissStore

        store = FaissStore.load(args.index_path, args.meta_path)
        embedder = Embedder(model_name=args.embed_model)
        retriever = Retriever(store=store, embedder=embedder)
        generator = LLMGenerator(
            model_name_or_path=args.gen_model,
            tokenizer_path=args.tokenizer or args.gen_model,
        )
        rag_pipeline = RAGPipeline(retriever=retriever, generator=generator, top_k=args.gen_top_k)

        def provider(item: Dict) -> str:
            result = rag_pipeline.generate(item["instruction"], item.get("input", ""))
            return result["code"]

        candidate_provider = provider

    summary = evaluate_dataset(
        samples=samples,
        main_c_path=args.main_c,
        limit=args.limit,
        iec2c_flags=args.iec2c_flags,
        logic_mode=args.logic_mode,
        judge=judge,
        candidate_provider=candidate_provider,
    )
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Mean score: {summary['mean_score']:.3f} | Saved to {args.results}")


if __name__ == "__main__":
    main()
