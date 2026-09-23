#!/usr/bin/env python3
"""Score a trained checkpoint on a held-out split.

    python scripts/evaluate.py runs/<run>/best.pt --data data/tracks_vs_showers/test.npz
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import torch
from torch.utils.data import DataLoader

from patt_reco.config import CLASS_NAMES
from patt_reco.dataset.torch_dataset import ImageDataset
from patt_reco.models import CNN
from patt_reco.train.metrics import (accuracy, confusion_matrix, format_report,
                                     per_class_recall, roc_auc)
from patt_reco.viz.display import plot_grid


def load_model(path: Path):
    """Rebuild the model from the checkpoint alone -- no training config needed."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    m = checkpoint["config"].model
    model = CNN(checkpoint["height"], checkpoint["width"],
                n_filters=m.n_filters, kernel_size=m.kernel_size, pool=m.pool,
                hidden=m.hidden, dropout=m.dropout)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def predict(model, loader, device):
    y_true, y_pred, probs = [], [], []
    for images, labels in loader:
        logits = model(images.to(device))
        p = torch.softmax(logits, dim=1)
        y_true.append(labels.numpy())
        y_pred.append(p.argmax(dim=1).cpu().numpy())
        probs.append(p.cpu().numpy())
    return (np.concatenate(y_true), np.concatenate(y_pred), np.concatenate(probs))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path, help="a best.pt / last.pt written by Run.save")
    parser.add_argument("--data", type=Path, required=True, help="the npz split to score")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--n-worst", type=int, default=16,
                        help="how many misclassifications to plot")
    args = parser.parse_args()

    use_cuda = (args.device == "cuda"
                or (args.device == "auto" and torch.cuda.is_available()))
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")

    model = load_model(args.checkpoint).to(device)
    dataset = ImageDataset(args.data)
    loader = DataLoader(dataset, batch_size=256)

    y_true, y_pred, probs = predict(model, loader, device)
    scores = probs[:, 1]

    print(f"checkpoint {args.checkpoint}")
    print(f"data       {args.data}  ({len(dataset)} images, device {device.type})\n")
    print(format_report(y_true, y_pred, scores))

    out_dir = args.checkpoint.parent
    cm = confusion_matrix(y_true, y_pred)
    report = {
        "accuracy": accuracy(y_true, y_pred),
        "per_class_recall": {CLASS_NAMES.get(i, str(i)): float(r)
                             for i, r in enumerate(per_class_recall(cm))},
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_note": "rows are true classes, columns predicted",
        "roc_auc": roc_auc(y_true, scores),
        "checkpoint": str(args.checkpoint),
        "data": str(args.data),
        "n_images": int(len(dataset)),
    }
    report_path = out_dir / "test_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {report_path}")

    # the worst mistakes: ranked by how confident the model was in the wrong class
    wrong = np.flatnonzero(y_true != y_pred)
    if wrong.size == 0:
        print("no misclassifications -- skipping misclassified.png")
        return

    confidence = probs[wrong, y_pred[wrong]]
    worst = wrong[np.argsort(-confidence)][: args.n_worst]

    figure = plot_grid([dataset.images[i] for i in worst], None,
                       ncols=min(8, len(worst)))
    for ax, i in zip([a for a in figure.get_axes() if a.images], worst):
        ax.set_title(f"{CLASS_NAMES.get(int(y_true[i]), y_true[i])}"
                     f" -> {CLASS_NAMES.get(int(y_pred[i]), y_pred[i])}"
                     f"  {probs[i, y_pred[i]]:.2f}", fontsize=8)
    figure.tight_layout()      # the titles are longer than the ones plot_grid laid out
    figure_path = out_dir / "misclassified.png"
    figure.savefig(figure_path, dpi=110)
    print(f"wrote {figure_path}  ({len(worst)} of {wrong.size} misclassified)")


if __name__ == "__main__":
    main()
