import argparse
import json
import logging
from pathlib import Path
from typing import List

from pypdf import PdfReader
from tqdm import tqdm

from .data_models import Document, PLCExample
from .embedding import Embedder
from .settings import (
    BOOKS_DIR,
    DATASET_PATH,
    DEFAULT_EMBED_MODEL,
    INDEX_PATH,
    METADATA_PATH,
)
from .utils import chunk_text, ensure_dir
from .vector_store import FaissStore

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_plc_dataset(path: Path) -> List[PLCExample]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [PLCExample(**item) for item in raw]


def dataset_to_documents(dataset: List[PLCExample]) -> List[Document]:
    documents: List[Document] = []
    for idx, item in enumerate(dataset):
        signals_in = ", ".join([f"{sig.get('name')}({sig.get('type')})" for sig in item.inputs]) if item.inputs else ""
        signals_out = ", ".join([f"{sig.get('name')}({sig.get('type')})" for sig in item.outputs]) if item.outputs else ""
        text = (
            f"指令: {item.instruction}\n"
            f"描述: {item.description}\n"
            f"输入信号: {signals_in}\n"
            f"输出信号: {signals_out}\n"
            f"库依赖: {item.library_dependency}\n"
            f"标准: {item.iec_standard}\n"
            f"PLC代码:\n{item.output}\n"
        )
        documents.append(
            Document(
                doc_id=f"plc-{idx}",
                text=text,
                metadata={"source": "plc_json", "id": idx, "instruction": item.instruction},
            )
        )
    return documents


def read_pdf_documents(books_dir: Path, chunk_size: int, overlap: int) -> List[Document]:
    documents: List[Document] = []
    for pdf_path in sorted(books_dir.glob("*.pdf")):
        try:
            reader = PdfReader(pdf_path)
            pages_text = []
            for page in reader.pages:
                content = page.extract_text()
                if content:
                    pages_text.append(content)
            if not pages_text:
                continue
            full_text = "\n".join(pages_text)
            chunks = chunk_text(full_text, chunk_size=chunk_size, overlap=overlap)
            for idx, chunk in enumerate(chunks):
                documents.append(
                    Document(
                        doc_id=f"{pdf_path.stem}-{idx}",
                        text=chunk,
                        metadata={"source": "book", "file": pdf_path.name, "chunk_id": idx},
                    )
                )
            logger.info("book: %s get chunks: %d",pdf_path,len(chunks))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to read %s: %s", pdf_path, exc)
    return documents


def build_index(
    dataset_path: Path = DATASET_PATH,
    books_dir: Path = BOOKS_DIR,
    embed_model: str = DEFAULT_EMBED_MODEL,
    index_path: Path = INDEX_PATH,
    meta_path: Path = METADATA_PATH,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> None:
    ensure_dir(index_path.parent)
    ensure_dir(meta_path.parent)

    logger.info("Loading dataset from %s", dataset_path)
    dataset = load_plc_dataset(dataset_path)
    plc_docs = dataset_to_documents(dataset)
    logger.info("Loaded %d PLC examples", len(plc_docs))

    logger.info("Loading PDFs from %s", books_dir)
    pdf_docs = read_pdf_documents(books_dir, chunk_size=chunk_size, overlap=chunk_overlap)
    logger.info("Loaded %d PDF chunks", len(pdf_docs))

    documents = plc_docs + pdf_docs
    texts = [doc.text for doc in documents]
    metadata = [{"doc_id": doc.doc_id, **doc.metadata, "text": doc.text} for doc in documents]

    logger.info("Building embeddings with %s", embed_model)
    embedder = Embedder(embed_model)
    embeddings = embedder.encode(texts)

    store = FaissStore(dimension=embedder.dimension)
    store.add_embeddings(embeddings, metadata)
    store.save(index_path=index_path, metadata_path=meta_path)
    logger.info("Index saved to %s and metadata to %s", index_path, meta_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FAISS index for PLC RAG")
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH, help="Path to PLC JSON dataset")
    parser.add_argument("--books", type=Path, default=BOOKS_DIR, help="Directory containing PDF books")
    parser.add_argument("--embed-model", type=str, default=DEFAULT_EMBED_MODEL, help="Sentence embedding model")
    parser.add_argument("--index-path", type=Path, default=INDEX_PATH, help="Where to save FAISS index")
    parser.add_argument("--meta-path", type=Path, default=METADATA_PATH, help="Where to save metadata")
    parser.add_argument("--chunk-size", type=int, default=800, help="Chunk size for book text")
    parser.add_argument("--chunk-overlap", type=int, default=120, help="Chunk overlap for book text")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_index(
        dataset_path=args.dataset,
        books_dir=args.books,
        embed_model=args.embed_model,
        index_path=args.index_path,
        meta_path=args.meta_path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
