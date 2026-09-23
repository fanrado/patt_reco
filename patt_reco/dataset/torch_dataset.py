"""torch `Dataset` over one generated npz split.

The splits are small enough to hold in memory, so the arrays are read once in
`__init__` and indexed directly -- no lazy handles, no per-worker re-opening.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    """Images and labels from a single `<split>.npz` written by `build.py`."""

    def __init__(self, path):
        self.path = Path(path)
        with np.load(self.path) as data:
            self.images = np.ascontiguousarray(data["images"], dtype=np.float32)
            self.labels = np.ascontiguousarray(data["labels"], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        image = torch.from_numpy(self.images[i]).unsqueeze(0)
        return image, int(self.labels[i])

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.path.name}, n={len(self)})"
