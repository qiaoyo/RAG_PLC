#!/usr/bin/env bash
set -euo pipefail

# 路径与模型可按需覆盖
GEN_MODEL="${GEN_MODEL:-/media/simple/another_Downloads/Qwen2.5-Coder-14B-Instruct}"
JUDGE_MODEL="${JUDGE_MODEL:-/media/simple/another_Downloads/Qwen2.5-7B-Instruct}"
TOKENIZER="${TOKENIZER:-$GEN_MODEL}"
DATASET="data/data_full/train_full_normalized.json"
INDEX_PATH="rag/artifacts/full_faiss.index"
META_PATH="rag/artifacts/full_faiss_meta.json"

echo "[1/2] 构建向量索引 (全量 677 条 + books)..."
python -m rag.ingestion \
  --dataset "${DATASET}" \
  --books books \
  --index-path "${INDEX_PATH}" \
  --meta-path "${META_PATH}" \
  --embed-model BAAI/bge-large-zh-v1.5

echo "[2/2] 运行基准评测 (批量判分)..."
python -m rag.benchmark \
  --dataset "${DATASET}" \
  --index-path "${INDEX_PATH}" \
  --meta-path "${META_PATH}" \
  --gen-model "${GEN_MODEL}" \
  --judge-model "${JUDGE_MODEL}" \
  --tokenizer "${TOKENIZER}" \
  --gen-provider hf \
  --judge-provider hf \
  --gen-device cuda:0 \
  --judge-device cuda:1 \
  --gen-8bit \
  --judge-8bit \
  --gen-max-len 2048 \
  --judge-max-len 2048 \
  --judge-batch-size 8 \
  --results rag/artifacts/full_benchmark_results.json

echo "完成。结果保存至 rag/artifacts/full_benchmark_results.json"

你所做的不就是把已有的代码跑了一遍吗？我希望重新从更深刻的意义上来构建benchmark。 而不是现在仅仅让llm读一段代码，我希望从编译成功性，静态分析，运行稳定，逻辑正确。，按照2：2：2：4的比例来打分（0-1）