import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
    text = text.replace("\r", "\n")
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    merged = "\n".join(paragraphs)
    chunks: List[str] = []
    start = 0
    while start < len(merged):
        end = start + chunk_size
        chunks.append(merged[start:end])
        start = max(end - overlap, start + 1)
    return chunks


def extract_first_json(text: str) -> Dict[str, Any]:
    pattern = re.compile(r"\{.*?\}", re.DOTALL)
    for match in pattern.finditer(text):
        snippet = match.group(0)
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            continue
    return {}


def extract_code_block(text: str) -> str:
    fenced = re.findall(r"```(?:[a-zA-Z0-9]+)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        return fenced[0].strip()
    markers = ["PROGRAM", "FUNCTION_BLOCK", "VAR_INPUT", "VAR"]
    for marker in markers:
        idx = text.find(marker)
        if idx != -1:
            return text[idx:].strip()
    return text.strip()


def flatten_metadata_items(items: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for item in items:
        new_item = {}
        for key, value in item.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                new_item[key] = value
            else:
                new_item[key] = str(value)
        result.append(new_item)
    return result
