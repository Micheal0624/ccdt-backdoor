import os
import json
import random
from typing import Any, Dict

import numpy as np
import torch
import yaml


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(obj: Dict[str, Any], path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def top1_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def get_dataset_label(dataset, index: int) -> int:
    if hasattr(dataset, "targets"):
        return int(dataset.targets[index])
    if hasattr(dataset, "labels"):
        return int(dataset.labels[index])
    if hasattr(dataset, "_labels"):
        return int(dataset._labels[index])
    if hasattr(dataset, "samples"):
        return int(dataset.samples[index][1])
    _, y = dataset[index]
    return int(y)
