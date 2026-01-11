import json
import random
from pathlib import Path
from typing import Dict, List, Tuple


def load_dataset(path: Path) -> List[Dict]:
    """Load PLC dataset from JSON list."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Dataset must be a list of dicts")
    return data


def split_dataset(
    data: List[Dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split dataset into train/val/test."""
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1.0")
    data = list(data)
    random.Random(seed).shuffle(data)
    n = len(data)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = data[:n_train]
    val = data[n_train : n_train + n_val]
    test = data[n_train + n_val :]
    return train, val, test


def save_splits(
    splits: Dict[str, List[Dict]],
    path: Path,
) -> None:
    """Persist splits to JSON for reproducibility."""
    payload = {k: v for k, v in splits.items()}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_splits(path: Path) -> Dict[str, List[Dict]]:
    """Load previously saved splits."""
    return json.loads(path.read_text(encoding="utf-8"))

if __name__ =="__main__":
    full_data = load_dataset(path=Path("../data/data_full/train_full_normalized.json"))
    train, val, test = split_dataset(full_data, train_ratio=0.8, val_ratio=0.1, seed=42)
    splits = {"train": train, "val": val, "test": test}
    split_file = Path("./split.json")
    split_file.parent.mkdir(parents=True, exist_ok=True)
    save_splits(splits, split_file)