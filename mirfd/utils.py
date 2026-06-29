from __future__ import annotations

from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_checkpoint(path: str | Path, model: torch.nn.Module, **payload: Any) -> None:
    checkpoint = {"model": model.state_dict(), **payload}
    torch.save(checkpoint, path)


def load_checkpoint(path: str | Path, model: torch.nn.Module, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location)
    state = checkpoint.get("model", checkpoint)
    model.load_state_dict(state, strict=True)
    return checkpoint if isinstance(checkpoint, dict) else {"model": state}


class AverageMeter:
    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / max(self.count, 1)
