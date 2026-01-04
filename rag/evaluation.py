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
    def __init__(self, model_name: str = DEFAULT_JUDGE_MODEL, tokenizer_path: Optional[str] = None):
        self.generator = LLMGenerator(model_name_or_path=model_name, tokenizer_path=tokenizer_path, provider="hf")

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


class BenchmarkRunner:
    def __init__(self, pipeline: RAGPipeline, judge: LLMJudge, dataset: List[PLCExample]):
        self.pipeline = pipeline
        self.judge = judge
        self.dataset = dataset

    def run(self, limit: Optional[int] = None) -> Dict:
        results: List[Dict] = []
        subset = self.dataset[:limit] if limit else self.dataset
        for example in tqdm(subset, desc="Benchmarking"):
            generated = self.pipeline.generate(example.instruction, example.input)
            candidate_code = extract_code_block(generated["raw_output"])
            judgment = self.judge.score(example, candidate_code)
            results.append(
                {
                    "instruction": example.instruction,
                    "input": example.input,
                    "reference": example.output,
                    "candidate": candidate_code,
                    "score": judgment["score"],
                    "pass": judgment["pass"],
                    "reason": judgment["reason"],
                }
            )
        scores = [item["score"] for item in results]
        mean_score = statistics.mean(scores) if scores else 0.0
        pass_rate = sum(1 for s in results if s["pass"]) / len(results) if results else 0.0
        return {"mean_score": mean_score, "pass_rate": pass_rate, "details": results}
