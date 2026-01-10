# PLC RAG (本地检索增强生成与评测栈)

面向 PLC 代码生成/评测的本地可运行方案，覆盖数据摄取、向量检索、代码生成、LLM 评测 benchmark，以及 LoRA 微调样例。

## 推荐软件环境（Ubuntu 22 + 48G+24G GPU）
- Python 3.10（或 3.11）；`conda create -n plc_rag python=3.10`
- CUDA 12.1 驱动；PyTorch GPU 版：`pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121`
- 依赖安装：`pip install -r rag/requirements.txt`
- 可选：`pip install flash-attn --no-build-isolation`（提升大模型推理速度，如遇编译失败可跳过）

## 数据与路径
- 电子书：`books/*.pdf`
- PLC 代码样本：`data/data_sample/train_001.json`（全量数据同格式）
- DeepSeek Tokenizer：`code/deepseek_v3_tokenizer/`（可在加载模型时作为 tokenizer 路径）
- RAG 工程与产物：`rag/`（向量索引默认写入 `rag/artifacts/`）

## 快速开始
1) 构建向量索引（默认使用 `BAAI/bge-large-zh-v1.5`）  
   ```bash
   cd /home/qiaoyo/python_proj/PLC_RAG/PLC_RAG
   python -m rag.ingestion --dataset data/data_sample/train_001.json --books books \
     --index-path rag/artifacts/plc_faiss.index --meta-path rag/artifacts/plc_faiss_meta.json
   ```

2) 单条 RAG 生成  
   ```bash
   python -m rag.pipeline --query "用st语言实现一个16Dword大小的先进先出内存功能块" \
     --input "" --top-k 5 --gen-model Qwen2.5-7B-Instruct \
     --tokenizer code/deepseek_v3_tokenizer
   ```

3) Benchmark（LLM-as-judge，默认 `Qwen2.5-14B-Instruct`）  
   ```bash
   python -m rag.benchmark --dataset data/data_sample/train_001.json \
     --results rag/artifacts/bench_results.json \
     --judge-model Qwen2.5-14B-Instruct --gen-model Qwen2.5-7B-Instruct
   ```
   结果包含均值得分与逐条说明，便于对比不同模型/检索配置。

4) LoRA 微调示例（针对 PLC 指令-代码生成）  
   ```bash
   python -m rag.finetune --model Qwen2.5-7B-Instruct \
     --tokenizer code/deepseek_v3_tokenizer \
     --dataset data/data_sample/train_001.json \
     --output rag/artifacts/qwen7b-plc-lora --lr 1e-4 --epochs 2
   ```

## 全量 677 条基准评测（自动脚本）
运行 `rag/run_full_benchmark.sh`（可在脚本顶部或环境变量覆盖模型路径），默认：
- 数据集：`data/data_full/train_full_normalized.json`
- 构建索引到 `rag/artifacts/full_faiss.index`
- 生成/判分模型：GEN 默认 `Qwen2.5-Coder-14B-Instruct`，JUDGE 默认 `Qwen2.5-7B-Instruct`
- HF + bitsandbytes 8bit，判分批量 8，`max_len=2048`
```bash
cd /home/qiaoyo/python_proj/PLC_RAG/PLC_RAG
GEN_MODEL=/path/to/gen JUDGE_MODEL=/path/to/judge TOKENIZER=/path/to/tokenizer \
  ./rag/run_full_benchmark.sh
```
结果写入 `rag/artifacts/full_benchmark_results.json`。

## 设计要点
- 检索：PDF 章节与已整理的 PLC 代码/描述一起构建向量库（支持 GPU FAISS）。  
- 生成：默认 HuggingFace `transformers`，可切换 `vllm` 提升吞吐。  
- 评测：LLM-as-judge，对比模型输出与参考 PLC 代码，给出 0-1 评分与理由。  
- 微调：LoRA SFT，直接用现有 JSON 数据格式生成指令-回复样本。

## 可选优化思路
- 更强嵌入模型（如 bge-m3/bge-large-zh-v1.5）与跨文档重排序。  
- 长上下文大模型（Qwen2.5-32B/14B) + vLLM 张量并行，更好覆盖书籍段落。  
- 数据增广：将书中相关章节与代码库拼接，生成合成 Q&A 强化检索。  
- 训练小模型 reranker 或校验器，结合 LLM-as-judge 形成级联评测。
