"""Normalize train_full.json by flattening metadata to top-level fields.

规范化 train_full.json，移除 metadata 嵌套，使所有记录与前 50 条保持一致。
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DEFAULT_INPUT = Path("data/data_full/train_full.json")
DEFAULT_OUTPUT = DEFAULT_INPUT.with_name(f"{DEFAULT_INPUT.stem}_normalized.json")


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    """Read the dataset and ensure it is a list of dict entries."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list at {path}, got {type(data).__name__}")
    return data


def flatten_metadata(entry: Dict[str, Any], index: int) -> Dict[str, Any]:
    """Move metadata contents to the top level and drop the wrapper."""
    if "metadata" not in entry:
        return entry

    metadata = entry.pop("metadata")
    if metadata is None:
        return entry
    if not isinstance(metadata, dict):
        raise ValueError(f"Entry {index} has non-dict metadata: {type(metadata).__name__}")

    for key, value in metadata.items():
        if key in entry and entry[key] not in (None, "", [], {}):
            logger.debug("Entry %d keeps existing value for %s", index, key)
            continue
        entry[key] = value
    return entry


def normalize_dataset(dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return a new dataset with metadata flattened for every entry."""
    normalized: List[Dict[str, Any]] = []
    for idx, item in enumerate(dataset):
        if not isinstance(item, dict):
            raise ValueError(f"Item {idx} is not a dict: {type(item).__name__}")
        normalized.append(flatten_metadata(dict(item), idx))
    return normalized


def write_dataset(path: Path, dataset: List[Dict[str, Any]]) -> None:
    """Write dataset to JSON with UTF-8 encoding."""
    with path.open("w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
        f.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Flatten metadata in train_full.json to match the first 50 records."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to the original train_full.json file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to write the normalized JSON file.",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite the input file after creating a .bak backup.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path: Path = args.input
    output_path: Path = args.output

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if args.inplace:
        output_path = input_path
        backup_path = input_path.with_suffix(f"{input_path.suffix}.bak")
        backup_path.write_bytes(input_path.read_bytes())
        logger.info("Created backup at %s", backup_path)

    dataset = load_dataset(input_path)
    if len(dataset) != 677:
        logger.warning("Unexpected dataset length: %d (expected 677)", len(dataset))

    normalized = normalize_dataset(dataset)
    write_dataset(output_path, normalized)
    logger.info("Normalized dataset written to %s", output_path)


if __name__ == "__main__":
    main()
