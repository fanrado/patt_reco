"""Semantic-segmentation metrics.

Everything is accumulated as a confusion matrix, because every number the plan
asks for -- per-class IoU, mIoU, pixel accuracy, the confusion matrix itself --
falls out of one, and accumulating counts rather than per-batch averages is the
only way to get an unbiased dataset-level IoU.

The overlap rule (see `detector/projection.py`) means a pixel fed by several
objects carries the label of its largest contributor. That choice biases
results, so every metric is reported three ways and the caller picks:

    all        every hit pixel, ambiguous ones included at face value
    exclusive  pixels with exactly one contributor
    weighted   every pixel, ambiguous ones down-weighted by 1 / n_contrib

`exclusive` is the honest headline number; `all` is what a naive evaluation
would print; the gap between them is the size of the label ambiguity.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import CLASS_NAMES, EMPTY, N_CLASSES

MODES = ("all", "exclusive", "weighted")


def confusion_matrix(pred: np.ndarray, true: np.ndarray, n_classes: int = N_CLASSES,
                     weights: np.ndarray | None = None) -> np.ndarray:
    """[n_classes, n_classes] float array, rows = truth, columns = prediction."""
    pred = np.asarray(pred).reshape(-1)
    true = np.asarray(true).reshape(-1)
    index = true.astype(np.int64) * n_classes + pred.astype(np.int64)
    counts = np.bincount(index, weights=None if weights is None else
                         np.asarray(weights).reshape(-1), minlength=n_classes**2)
    return counts.reshape(n_classes, n_classes).astype(np.float64)


def iou_from_confusion(cm: np.ndarray) -> np.ndarray:
    """Per-class IoU; NaN for classes absent from both truth and prediction."""
    true_total = cm.sum(axis=1)
    pred_total = cm.sum(axis=0)
    intersection = np.diag(cm)
    union = true_total + pred_total - intersection
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = np.where(union > 0, intersection / union, np.nan)
    return iou


def dice_from_confusion(cm: np.ndarray) -> np.ndarray:
    true_total, pred_total = cm.sum(axis=1), cm.sum(axis=0)
    intersection = np.diag(cm)
    denom = true_total + pred_total
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0, 2.0 * intersection / denom, np.nan)


def purity_efficiency(cm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-class precision and recall, in the names a physicist expects."""
    with np.errstate(invalid="ignore", divide="ignore"):
        purity = np.where(cm.sum(axis=0) > 0, np.diag(cm) / cm.sum(axis=0), np.nan)
        efficiency = np.where(cm.sum(axis=1) > 0, np.diag(cm) / cm.sum(axis=1), np.nan)
    return purity, efficiency


@dataclass
class SemanticMetrics:
    """Confusion-matrix accumulator, one per reporting mode."""

    n_classes: int = N_CLASSES
    matrices: dict = field(default_factory=dict)
    n_pixels: int = 0
    n_events: int = 0

    def __post_init__(self):
        if not self.matrices:
            self.matrices = {m: np.zeros((self.n_classes, self.n_classes), dtype=np.float64)
                             for m in MODES}

    def update(self, pred: np.ndarray, true: np.ndarray,
               n_contrib: np.ndarray | None = None) -> None:
        """Accumulate one event (or one batch) of per-pixel predictions."""
        pred = np.asarray(pred).reshape(-1)
        true = np.asarray(true).reshape(-1)
        self.matrices["all"] += confusion_matrix(pred, true, self.n_classes)
        self.n_pixels += pred.size
        self.n_events += 1

        if n_contrib is None:
            # no ambiguity information: the other two modes degenerate to `all`
            self.matrices["exclusive"] += confusion_matrix(pred, true, self.n_classes)
            self.matrices["weighted"] += confusion_matrix(pred, true, self.n_classes)
            return

        contrib = np.asarray(n_contrib).reshape(-1)
        single = contrib <= 1
        self.matrices["exclusive"] += confusion_matrix(pred[single], true[single],
                                                       self.n_classes)
        weights = 1.0 / np.maximum(contrib.astype(np.float64), 1.0)
        self.matrices["weighted"] += confusion_matrix(pred, true, self.n_classes, weights)

    # -- reporting ---------------------------------------------------------
    def summary(self, mode: str = "exclusive", include_empty: bool = False) -> dict:
        if mode not in self.matrices:
            raise KeyError(f"mode must be one of {MODES}, got {mode!r}")
        cm = self.matrices[mode]
        iou = iou_from_confusion(cm)
        purity, efficiency = purity_efficiency(cm)

        classes = range(0 if include_empty else 1, self.n_classes)
        total = cm.sum()
        return {
            "mode": mode,
            "n_events": self.n_events,
            "pixel_accuracy": float(np.diag(cm).sum() / total) if total > 0 else float("nan"),
            "miou": float(np.nanmean([iou[c] for c in classes])),
            "per_class": {CLASS_NAMES[c]: {"iou": float(iou[c]),
                                           "purity": float(purity[c]),
                                           "efficiency": float(efficiency[c]),
                                           "n_true": float(cm[c].sum())}
                          for c in classes},
        }

    def foreground_summary(self, mode: str = "exclusive") -> dict:
        """Collapse to hit vs empty -- the part a classical threshold can also do."""
        cm = self.matrices[mode]
        empty_empty = cm[EMPTY, EMPTY]
        empty_hit = cm[EMPTY, 1:].sum()
        hit_empty = cm[1:, EMPTY].sum()
        hit_hit = cm[1:, 1:].sum()
        union = hit_hit + empty_hit + hit_empty
        return {
            "iou": float(hit_hit / union) if union > 0 else float("nan"),
            "purity": float(hit_hit / (hit_hit + empty_hit)) if hit_hit + empty_hit > 0 else float("nan"),
            "efficiency": float(hit_hit / (hit_hit + hit_empty)) if hit_hit + hit_empty > 0 else float("nan"),
            "true_negative": float(empty_empty),
        }

    def format_table(self, mode: str = "exclusive") -> str:
        s = self.summary(mode)
        fg = self.foreground_summary(mode)
        lines = [
            f"mode={mode}  events={s['n_events']}  pixel_acc={s['pixel_accuracy']:.4f}  "
            f"mIoU={s['miou']:.4f}",
            f"{'class':<12}{'IoU':>9}{'purity':>9}{'effic.':>9}{'truth px':>12}",
            "-" * 51,
        ]
        for name, row in s["per_class"].items():
            lines.append(f"{name:<12}{row['iou']:9.4f}{row['purity']:9.4f}"
                         f"{row['efficiency']:9.4f}{row['n_true']:12.0f}")
        lines.append("-" * 51)
        lines.append(f"{'foreground':<12}{fg['iou']:9.4f}{fg['purity']:9.4f}{fg['efficiency']:9.4f}")
        return "\n".join(lines)

    def confusion_table(self, mode: str = "exclusive", normalise: bool = True) -> str:
        cm = self.matrices[mode].copy()
        if normalise:
            row_sums = cm.sum(axis=1, keepdims=True)
            cm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)
        header = "true\\pred   " + "".join(f"{CLASS_NAMES[c][:7]:>8}" for c in range(self.n_classes))
        lines = [header, "-" * len(header)]
        for c in range(self.n_classes):
            cells = "".join(f"{cm[c, k]:8.3f}" if normalise else f"{cm[c, k]:8.0f}"
                            for k in range(self.n_classes))
            lines.append(f"{CLASS_NAMES[c]:<12}{cells}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "n_events": self.n_events,
            "n_pixels": self.n_pixels,
            "modes": {m: self.summary(m) for m in MODES},
            "foreground": {m: self.foreground_summary(m) for m in MODES},
            "confusion": {m: self.matrices[m].tolist() for m in MODES},
        }
