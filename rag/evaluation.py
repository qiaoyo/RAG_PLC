import argparse
import json
import logging
import statistics
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

from data_models import PLCExample
from generator import LLMGenerator
from pipeline import RAGPipeline
from settings import DEFAULT_JUDGE_MODEL
from utils import extract_code_block, extract_first_json

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


EVAL_PROMPT = """你是PLC代码评测专家。请对比候选代码与参考代码，关注：
1) 变量声明与类型是否匹配要求
2) 时序/边沿检测/滤波等逻辑是否正确
3) 安全性与复位逻辑是否合理
请输出 JSON：{{"score":0-1,"pass":true|false,"reason":"简要中文说明"}}。

用户指令:
{instruction}

补充输入:
{user_input}

参考代码:
{reference_code}

候选代码:
{candidate_code}
"""


class LLMJudge:
    def __init__(
        self,
        model_name: str = DEFAULT_JUDGE_MODEL,
        tokenizer_path: Optional[str] = None,
        device: Optional[str | int] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        provider: str = "hf",
        tensor_parallel_size: int = 1,
        max_model_len: int = 8192,
    ):
        self.generator = LLMGenerator(
            model_name_or_path=model_name,
            tokenizer_path=tokenizer_path,
            provider=provider,
            device=device,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
        )

    def score(self, example: PLCExample, candidate_code: str) -> Dict:
        prompt = EVAL_PROMPT.format(
            instruction=example.instruction,
            user_input=example.input or "无",
            reference_code=example.output,
            candidate_code=candidate_code,
        )
        raw = self.generator.generate(prompt, max_new_tokens=256, temperature=0.0)
        parsed = extract_first_json(raw)
        score = float(parsed.get("score", 0.0))
        passed = bool(parsed.get("pass", score >= 0.6))
        reason = parsed.get("reason", raw.strip())
        return {"score": score, "pass": passed, "reason": reason, "raw": raw}

    def score_batch(self, items: List[Dict]) -> List[Dict]:
        prompts = [
            EVAL_PROMPT.format(
                instruction=item["example"].instruction,
                user_input=item["example"].input or "无",
                reference_code=item["example"].output,
                candidate_code=item["candidate_code"],
            )
            for item in items
        ]
        raw_outputs = self.generator.generate(prompts, max_new_tokens=256, temperature=0.0)
        results: List[Dict] = []
        for item, raw in zip(items, raw_outputs):
            parsed = extract_first_json(raw)
            score = float(parsed.get("score", 0.0))
            passed = bool(parsed.get("pass", score >= 0.6))
            reason = parsed.get("reason", raw.strip())
            results.append(
                {
                    "instruction": item["example"].instruction,
                    "input": item["example"].input,
                    "reference": item["example"].output,
                    "candidate": item["candidate_code"],
                    "score": score,
                    "pass": passed,
                    "reason": reason,
                }
            )
        return results


class BenchmarkRunner:
    def __init__(self, pipeline: RAGPipeline, judge: LLMJudge, dataset: List[PLCExample]):
        self.pipeline = pipeline
        self.judge = judge
        self.dataset = dataset

    def run(self, limit: Optional[int] = None, batch_size: int = 4) -> Dict:
        results: List[Dict] = []
        subset = self.dataset[:limit] if limit else self.dataset
        buffer: List[Dict] = []
        for example in tqdm(subset, desc="Benchmarking"):
            generated = self.pipeline.generate(example.instruction, example.input)
            candidate_code = extract_code_block(generated["raw_output"])
            buffer.append({"example": example, "candidate_code": candidate_code})
            if len(buffer) >= batch_size:
                results.extend(self.judge.score_batch(buffer))
                buffer = []
        if buffer:
            results.extend(self.judge.score_batch(buffer))
        scores = [item["score"] for item in results]
        mean_score = statistics.mean(scores) if scores else 0.0
        pass_rate = sum(1 for s in results if s["pass"]) / len(results) if results else 0.0
        return {"mean_score": mean_score, "pass_rate": pass_rate, "details": results}
