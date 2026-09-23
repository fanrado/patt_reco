"""Everything a training run writes to disk.

A run directory holds two JSON files describing how the run was configured
and where it ran, a CSV of per-epoch metrics, and checkpoints. That is enough
to compare runs and to reload a model later -- no TensorBoard, no external
experiment tracker.
"""
from __future__ import annotations

import csv
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from ..config import config_hash, to_dict
from .config import TrainConfig


def _git_sha() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return "unknown"


def _device_name() -> str:
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "cpu"


class Run:
    """Owns one run directory and everything written into it."""

    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = cfg.run.name or "train"
        self.dir = Path(cfg.run.out_dir) / f"{stamp}_{name}"
        self.dir.mkdir(parents=True, exist_ok=True)

        self.metrics_path = self.dir / "metrics.csv"
        self._columns: list[str] | None = None

        (self.dir / "config.json").write_text(json.dumps(
            {"config": to_dict(cfg), "config_hash": config_hash(cfg)},
            indent=2, default=str))

        (self.dir / "environment.json").write_text(json.dumps({
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "device": _device_name(),
            "git_sha": _git_sha(),
        }, indent=2))

    # -- metrics -----------------------------------------------------------
    def log(self, epoch: int, **metrics) -> None:
        """Append one row to metrics.csv; the header follows what is logged."""
        row = {"epoch": epoch, **metrics}
        new_file = self._columns is None
        if new_file:
            self._columns = list(row)
        with open(self.metrics_path, "a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._columns,
                                    extrasaction="ignore")
            if new_file:
                writer.writeheader()
            writer.writerow(row)

    # -- checkpoints -------------------------------------------------------
    def save(self, model, name: str) -> Path:
        """Write a checkpoint that can be reloaded without a training config."""
        path = self.dir / name
        torch.save({
            "state_dict": model.state_dict(),
            "config": self.cfg,
            "height": getattr(model, "height", None),
            "width": getattr(model, "width", None),
        }, path)
        return path

    def save_best(self, model) -> Path:
        return self.save(model, "best.pt")

    def save_last(self, model) -> Path:
        return self.save(model, "last.pt")

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.dir})"
