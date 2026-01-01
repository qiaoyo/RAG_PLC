"""Dataset utilities for PLC instructions and book corpora."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, Iterator, List, Optional

from .data_models import ChunkRecord, DocumentRecord

LOGGER = logging.getLogger(__name__)


def _ensure_text(value: Optional[str]) -> str:
    return value.strip() if isinstance(value, str) else ""


def format_instruction_record(raw: dict) -> str:
    """Turn a JSON instruction record into a plain-text document."""

    parts: List[str] = []
    instruction = _ensure_text(raw.get("instruction"))
    if instruction:
        parts.append(f"【指令】\n{instruction}")

    inputs = _ensure_text(raw.get("input"))
    if inputs:
        parts.append(f"【输入】\n{inputs}")

    description = _ensure_text(raw.get("description"))
    if description:
        parts.append(f"【说明】\n{description}")

    output = _ensure_text(raw.get("output"))
    if output:
        parts.append(f"【PLC代码】\n{output}")

    if not parts and raw:
        parts.append(json.dumps(raw, ensure_ascii=False))

    return "\n\n".join(parts)


def load_instruction_corpus(
    file_path: Path, limit: Optional[int] = None
) -> List[DocumentRecord]:
    """Load PLC instruction dataset stored as JSON list."""

    file_path = Path(file_path)
    with file_path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)

    documents: List[DocumentRecord] = []
    for idx, row in enumerate(data):
        if limit is not None and idx >= limit:
            break

        text = format_instruction_record(row)
        metadata = {
            "source": str(file_path),
            "kind": "instruction",
            "row_id": idx,
            "tags": row.get("tags", []),
            "iec_standard": row.get("iec_standard"),
        }
        documents.append(DocumentRecord(content=text, metadata=metadata))

    LOGGER.info("Loaded %s instruction records from %s", len(documents), file_path)
    return documents


def _extract_pdf_text(pdf_path: Path, max_pages: Optional[int]) -> str:
    """Extract raw text from a PDF using PyPDF."""

    from pypdf import PdfReader  # Local import to keep dependency optional

    reader = PdfReader(str(pdf_path))
    pages = reader.pages
    texts: List[str] = []

    page_count = min(len(pages), max_pages) if max_pages is not None else len(pages)
    for page_idx in range(page_count):
        page = pages[page_idx]
        try:
            texts.append(page.extract_text() or "")
        except Exception as exc:  # pragma: no cover - best effort extraction
            LOGGER.warning("Failed to parse %s page %s: %s", pdf_path, page_idx, exc)

    return "\n".join(texts)


def load_books_corpus(
    books_dir: Path, limit_pages: Optional[int] = None
) -> List[DocumentRecord]:
    """Load PDF books into document records."""

    documents: List[DocumentRecord] = []
    books_dir = Path(books_dir)

    if not books_dir.exists():
        LOGGER.warning("Books directory %s does not exist; skipping.", books_dir)
        return documents

    for pdf_path in sorted(books_dir.glob("*.pdf")):
        text = _extract_pdf_text(pdf_path, limit_pages)
        if not text.strip():
            continue

        documents.append(
            DocumentRecord(
                content=text,
                metadata={
                    "source": str(pdf_path),
                    "kind": "book",
                    "title": pdf_path.stem,
                },
            )
        )

    LOGGER.info("Loaded %s book documents from %s", len(documents), books_dir)
    return documents


def save_chunks(chunks: Iterable[ChunkRecord], output_file: Path) -> None:
    """Persist chunk records as JSONL."""

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as fp:
        for chunk in chunks:
            record = {"text": chunk.text, "metadata": chunk.metadata}
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")

    LOGGER.info("Saved chunks to %s", output_file)


def load_chunks(input_file: Path) -> List[ChunkRecord]:
    """Load chunk records from JSONL file."""

    chunks: List[ChunkRecord] = []
    with Path(input_file).open("r", encoding="utf-8") as fp:
        for line in fp:
            data = json.loads(line)
            chunks.append(ChunkRecord(text=data["text"], metadata=data["metadata"]))
    return chunks
