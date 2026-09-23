"""Classification metrics, computed from plain numpy arrays.

All hand-rolled: the project depends on numpy, matplotlib and pyyaml, and a
handful of counting operations is not worth a scikit-learn dependency.
"""
from __future__ import annotations

import numpy as np

from ..config import CLASS_NAMES, N_CLASSES


def accuracy(y_true, y_pred) -> float:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if y_true.size == 0:
        return float("nan")
    return float((y_true == y_pred).mean())


def confusion_matrix(y_true, y_pred, n_classes: int = N_CLASSES) -> np.ndarray:
    """Counts as `[true, predicted]`.

    **Rows are the true classes and columns the predicted ones** -- reading
    the transpose is the usual way to misinterpret a confusion matrix, so it
    is worth being explicit.
    """
    y_true = np.asarray(y_true).ravel().astype(int)
    y_pred = np.asarray(y_pred).ravel().astype(int)
    flat = np.bincount(y_true * n_classes + y_pred, minlength=n_classes ** 2)
    return flat.reshape(n_classes, n_classes)


def per_class_recall(cm) -> np.ndarray:
    """Fraction of each true class that was predicted correctly (row-wise).

    This quantity is called **efficiency** in the project's vocabulary.

    A class absent from the truth has undefined recall and yields nan.
    """
    cm = np.asarray(cm, dtype=float)
    support = cm.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(support > 0, np.diag(cm) / support, np.nan)


def per_class_precision(cm) -> np.ndarray:
    """Fraction of each predicted class that was correct (column-wise).

    This quantity is called **purity** in the project's vocabulary: of
    everything labelled class c, how much really was class c.

    A class that was never predicted has undefined precision and yields nan.
    """
    cm = np.asarray(cm, dtype=float)
    predicted = cm.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(predicted > 0, np.diag(cm) / predicted, np.nan)


def _average_ranks(x: np.ndarray) -> np.ndarray:
    """Ranks starting at 1, with tied values sharing their average rank."""
    n = x.size
    order = np.argsort(x, kind="mergesort")
    ordered = x[order]
    values, first = np.unique(ordered, return_index=True)
    last = np.r_[first[1:], n] - 1
    shared = 0.5 * (first + last) + 1.0          # average rank within each tie group
    ranks = np.empty(n, dtype=float)
    ranks[order] = shared[np.searchsorted(values, ordered)]
    return ranks


def roc_auc(y_true, scores) -> float:
    """Binary ROC-AUC via the Mann-Whitney U identity.

    AUC = (sum of the positives' ranks - n_pos * (n_pos + 1) / 2)
          / (n_pos * n_neg)

    Ranks are averaged over ties, so tied scores contribute 0.5 as they
    should. Returns nan when either class is absent.
    """
    y_true = np.asarray(y_true).ravel()
    scores = np.asarray(scores, dtype=float).ravel()
    positive = y_true == 1
    n_pos = int(positive.sum())
    n_neg = int(y_true.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _average_ranks(scores)
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def format_report(y_true, y_pred, scores) -> str:
    """A short human-readable summary of a set of predictions."""
    cm = confusion_matrix(y_true, y_pred)
    purity = per_class_precision(cm)
    efficiency = per_class_recall(cm)
    names = [CLASS_NAMES.get(i, str(i)) for i in range(cm.shape[0])]
    width = max(len(n) for n in names)

    lines = [f"accuracy   {accuracy(y_true, y_pred):.4f}",
             f"roc auc    {roc_auc(y_true, scores):.4f}",
             "",
             f"{'':<{width}}    purity (precision)  efficiency (recall)"]
    lines += [f"  {name:<{width}}  {p:>16.4f}  {e:>19.4f}"
              for name, p, e in zip(names, purity, efficiency)]
    lines += ["",
              "confusion (rows = true, columns = predicted)",
              " " * (width + 2) + "".join(f"{n:>8}" for n in names)]
    for name, row in zip(names, cm):
        lines.append(f"  {name:<{width}}" + "".join(f"{v:>8d}" for v in row))
    return "\n".join(lines)
