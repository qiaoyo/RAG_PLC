import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

BOOKS_DIR = PROJECT_ROOT / "books"
DATASET_PATH = PROJECT_ROOT / "data" / "data_full" / "train_full_normalized.json"

ARTIFACT_DIR = BASE_DIR / "artifacts"
INDEX_PATH = ARTIFACT_DIR / "plc_faiss.index"
METADATA_PATH = ARTIFACT_DIR / "plc_faiss_meta.json"

DEFAULT_EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "BAAI/bge-large-zh-v1.5")
DEFAULT_GEN_MODEL = os.getenv("RAG_GEN_MODEL", "/media/simple/another_Downloads/Qwen2.5-Coder-32B-Instruct")

DEFAULT_JUDGE_MODEL = os.getenv("RAG_JUDGE_MODEL", "/media/simple/another_Downloads/models/llms/qwen-14b")
# DEFAULT_TOKENIZER_PATH = os.getenv(
#     "RAG_TOKENIZER_PATH", str(PROJECT_ROOT / "code" / "deepseek_v3_tokenizer")
# )

DEFAULT_TOKENIZER_PATH = os.getenv("RAG_TOKENIZER_PATH", "/media/simple/another_Downloads/Qwen2.5-Coder-32B-Instruct")