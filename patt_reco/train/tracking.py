"""Run directories, metric logging and per-epoch previews.

Local only -- TensorBoard and a CSV, no cloud tracker. A run directory is
self-describing: config, git SHA, environment, metrics and checkpoints, so a
result can be reproduced from the directory alone.
"""
from __future__ import annotations

import csv
import json
import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


class Run:
    """One training or evaluation run on disk."""

    def __init__(self, root, cfg=None, name: str = "", tag: str = "run"):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = f"_{name}" if name else ""
        self.dir = Path(root) / f"{stamp}_{tag}{suffix}"
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "previews").mkdir(exist_ok=True)

        self.csv_path = self.dir / "metrics.csv"
        self._csv_fields: list[str] | None = None
        self._writer = None

        if cfg is not None:
            self.save_json("config.json", asdict(cfg))
        self.save_json("environment.json", {
            "git_sha": git_sha(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "argv": sys.argv,
            "started": stamp,
        })

    # -- artefacts ----------------------------------------------------------
    def save_json(self, name: str, payload: dict) -> Path:
        path = self.dir / name
        path.write_text(json.dumps(payload, indent=2, default=_jsonable))
        return path

    def log_row(self, row: dict) -> None:
        """Append one row to metrics.csv, fixing the columns on the first call."""
        if self._writer is None:
            self._csv_fields = list(row)
            handle = self.csv_path.open("w", newline="")
            self._writer = csv.DictWriter(handle, fieldnames=self._csv_fields)
            self._writer.writeheader()
            self._handle = handle
        self._writer.writerow({k: row.get(k, "") for k in self._csv_fields})
        self._handle.flush()

    @property
    def tensorboard(self):
        if not hasattr(self, "_tb"):
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._tb = SummaryWriter(str(self.dir / "tb"))
            except Exception:
                self._tb = None
        return self._tb

    def scalar(self, tag: str, value: float, step: int) -> None:
        writer = self.tensorboard
        if writer is not None:
            writer.add_scalar(tag, value, step)

    def close(self) -> None:
        if getattr(self, "_tb", None) is not None:
            self._tb.close()
        if getattr(self, "_handle", None) is not None:
            self._handle.close()


def _jsonable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)
