# PLC_RAG

本项目提供完整的本地化PLC代码生成 RAG（Retrieval-Augmented Generation）流程：从书籍/代码数据构建向量索引，到基于可微调LLM的推理与训练管线。所有可执行代码位于 `src/` 目录，其中 `src/plc_rag/cli.py` 提供统一入口。

## 推荐软件环境

| 层级 | 建议配置 |
| --- | --- |
| 操作系统 | Ubuntu 22.04，NVIDIA 535+ 驱动，CUDA 12.1，cuDNN 9，开启持久化模式以发挥 48G GPU 性能 |
| 系统依赖 | `sudo apt install build-essential git-lfs python3.10 python3.10-venv libgl1 poppler-utils` |
| Python 环境 | `python3.10 -m venv .venv && source .venv/bin/activate` 或使用 `conda env create python=3.10` |
| 核心 Python 包 | `torch==2.2.1+cu121`, `transformers>=4.40`, `accelerate`, `bitsandbytes`, `peft`, `datasets`, `sentencepiece`, `sentence-transformers`, `faiss-gpu`, `typer`, `pypdf`, `langchain-text-splitters`, `tiktoken`（可选 token 统计） |
| 性能加速 | `xformers`, `flash-attn`、`deepspeed`（需要额外编译/安装） |

示例安装命令：

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch==2.2.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate bitsandbytes peft datasets sentencepiece \
    sentence-transformers faiss-gpu typer pypdf langchain-text-splitters tiktoken xformers
```

## 数据来源

- `books/`：8 本 PLC/ST 相关电子书（PDF），用于构建技术背景知识。
- `data/data_sample/train_001.json`：677 条自建 PLC 代码样本，与 `data/data_full/train_full.json` 结构一致。

## 代码结构

```
src/
└── plc_rag/
    ├── cli.py                 # Typer 命令行，统一调度
    ├── config.py              # 数据/模型/训练配置
    ├── datasets.py            # 数据加载、PDF 解析、JSONL 读写
    ├── chunkers.py            # 文本分块逻辑
    ├── embeddings.py          # sentence-transformer 嵌入封装
    ├── vector_store.py        # FAISS 向量库持久化
    ├── rag_pipeline.py        # 检索 + 生成主流程
    ├── finetune.py            # LoRA 微调工具
    ├── embedding_finetune.py  # 嵌入模型微调
    └── utils.py               # 日志等辅助函数
```

## 使用流程

1. **数据切分**
   ```bash
   python -m src.plc_rag.cli ingest \
     --code-file data/data_sample/train_001.json \
     --books-dir books \
     --output-file data/processed/chunks.jsonl
   ```
   可通过 `--limit-records` / `--limit-book-pages` 做快速调试。

2. **构建向量索引**
   ```bash
   python -m src.plc_rag.cli index \
     --chunks-file data/processed/chunks.jsonl \
     --vector-store-dir data/vector_store \
     --embedding-model BAAI/bge-large-zh-v1.5
   ```

3. **本地 RAG 推理**
   ```bash
   python -m src.plc_rag.cli query \
     "如何实现带异步复位的JK触发器？" \
     --vector-store-dir data/vector_store \
     --llm Qwen/Qwen2.5-7B-Instruct
   ```

4. **LLM LoRA 微调**
   ```bash
   python -m src.plc_rag.cli finetune-llm \
     --dataset-path data/data_sample/train_001.json \
     --base-model Qwen/Qwen2.5-7B-Instruct \
     --output-dir artifacts/llm_lora
   ```

5. **嵌入模型微调**
   ```bash
   python -m src.plc_rag.cli finetune-embedding \
     --dataset-path data/data_sample/train_001.json \
     --base-model BAAI/bge-large-zh-v1.5 \
     --output-dir artifacts/embedding_model
   ```

执行命令前请输入仓库根目录并激活虚拟环境。若需在 48G GPU 上训练更大的模型，可在 CLI 参数中调整 batch size、LoRA rank，或替换 `llm` 与嵌入模型名称。
