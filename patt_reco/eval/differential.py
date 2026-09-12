"""Differential metrics: performance *versus* a covariate, not averaged over it.

A single mIoU hides exactly the failure modes this project exists to expose. The
plan's stress axes -- multiplicity, SNR, crossing angle, occupancy -- are all of
this shape: bin the events by a covariate, accumulate a separate confusion
matrix per bin, and report the curve.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import N_CLASSES
from .metrics_sem import SemanticMetrics


@dataclass
class DifferentialMetrics:
    """One `SemanticMetrics` per bin of a covariate."""

    name: str
    edges: np.ndarray
    n_classes: int = N_CLASSES
    bins: list = field(default_factory=list)
    values: list = field(default_factory=list)

    def __post_init__(self):
        self.edges = np.asarray(self.edges, dtype=float)
        if not self.bins:
            self.bins = [SemanticMetrics(self.n_classes) for _ in range(len(self.edges) - 1)]

    def bin_index(self, value: float) -> int:
        return int(np.clip(np.searchsorted(self.edges, value, side="right") - 1,
                           0, len(self.bins) - 1))

    def update(self, value: float, pred, true, n_contrib=None) -> None:
        self.values.append(float(value))
        self.bins[self.bin_index(value)].update(pred, true, n_contrib)

    def curve(self, mode: str = "exclusive") -> dict:
        centres, miou, fg_iou, counts = [], [], [], []
        for i, metrics in enumerate(self.bins):
            centres.append(0.5 * (self.edges[i] + self.edges[i + 1]))
            counts.append(metrics.n_events)
            if metrics.n_events == 0:
                miou.append(float("nan")); fg_iou.append(float("nan")); continue
            miou.append(metrics.summary(mode)["miou"])
            fg_iou.append(metrics.foreground_summary(mode)["iou"])
        return {"name": self.name, "edges": self.edges.tolist(), "centre": centres,
                "miou": miou, "foreground_iou": fg_iou, "n_events": counts}

    def format_table(self, mode: str = "exclusive") -> str:
        curve = self.curve(mode)
        lines = [f"{self.name:>16}{'events':>9}{'mIoU':>9}{'fg IoU':>9}",
                 "-" * 43]
        for i in range(len(curve["centre"])):
            low, high = self.edges[i], self.edges[i + 1]
            label = f"{low:g}-{high:g}"
            lines.append(f"{label:>16}{curve['n_events'][i]:9d}"
                         f"{curve['miou'][i]:9.4f}{curve['foreground_iou'][i]:9.4f}")
        return "\n".join(lines)
