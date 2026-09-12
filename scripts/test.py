#!/usr/bin/env python3
"""PHASE 3 - model testing.

Evaluates a trained checkpoint on a *held-out test split of the training
distribution*. This is the in-distribution number. It is not the validation in
phase 4, which deliberately goes outside that distribution.

    python scripts/test.py runs/<run>/best.pt
    python scripts/test.py runs/<run>/best.pt --data data/l2_noise/test --baselines
    python scripts/test.py --baselines-only --data data/l2_noise/test

Reports mIoU three ways (see eval/metrics_sem.py on the overlap rule), per-class
IoU/purity/efficiency, the confusion matrix, and the classical-baseline floor.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from patt_reco.cliutil import describe_device, pick_device
from patt_reco.config import N_CLASSES
from patt_reco.eval.evaluate import evaluate_baseline, evaluate_model
from patt_reco.eval.metrics_sem import MODES, SemanticMetrics
from patt_reco.models.baselines import BASELINES
from patt_reco.train.tracking import Run


def report(title: str, metrics: SemanticMetrics, show_confusion: bool = False) -> dict:
    print(f"\n--- {title} ---")
    print(metrics.format_table("exclusive"))
    spread = {m: metrics.summary(m)["miou"] for m in MODES}
    print(f"  mIoU by overlap handling:  " +
          "  ".join(f"{m}={v:.4f}" for m, v in spread.items()))
    if show_confusion:
        print("\n" + metrics.confusion_table("exclusive"))
    return metrics.to_dict()


def save_previews(run, dataset, model, device, n_events: int) -> None:
    """Truth vs prediction, side by side -- the failure gallery starts here."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    from patt_reco.viz.event_display import _class_cmap

    model.eval()
    for i in range(min(n_events, len(dataset))):
        item = dataset[i]
        with torch.no_grad():
            logits = model({"views": item["views"][None].to(device)})["sem_logits"]
        pred = logits.argmax(dim=2)[0].cpu().numpy()
        true = item["semantic"].numpy()
        n_views = true.shape[0]

        fig, axes = plt.subplots(3, n_views, squeeze=False, figsize=(3.2 * n_views, 9.6))
        for v in range(n_views):
            axes[0][v].imshow(item["views"][v, 0].numpy().T, origin="lower", aspect="auto",
                              cmap="RdBu_r", vmin=-6, vmax=6)
            axes[1][v].imshow(true[v].T, origin="lower", aspect="auto", interpolation="nearest",
                              cmap=_class_cmap(), vmin=0, vmax=N_CLASSES - 1)
            axes[2][v].imshow(pred[v].T, origin="lower", aspect="auto", interpolation="nearest",
                              cmap=_class_cmap(), vmin=0, vmax=N_CLASSES - 1)
            axes[0][v].set_title(f"view {v}", fontsize=9)
        for row, label in enumerate(("input (sigma)", "truth", "prediction")):
            axes[row][0].set_ylabel(label, fontsize=9)
        agree = float((pred == true)[true > 0].mean()) if (true > 0).any() else float("nan")
        fig.suptitle(f"event {item['index']}  --  hit-pixel agreement {agree:.3f}", fontsize=10)
        fig.tight_layout()
        fig.savefig(run.dir / "previews" / f"test_{item['index']:05d}.png", dpi=100)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path, nargs="?", default=None)
    parser.add_argument("--data", type=Path, default=Path("data/l2_noise/test"))
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-events", type=int, default=0, help="0 = all")
    parser.add_argument("--baselines", action="store_true",
                        help="also evaluate the classical floor")
    parser.add_argument("--baselines-only", action="store_true")
    parser.add_argument("--baseline-events", type=int, default=200,
                        help="baselines are slow; evaluate this many events")
    parser.add_argument("--previews", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("runs"))
    args = parser.parse_args()

    if args.checkpoint is None and not args.baselines_only:
        parser.error("pass a checkpoint, or --baselines-only")

    from patt_reco.dataset.torch_dataset import PattRecoDataset, make_loader

    dataset = PattRecoDataset(args.data, augment=False, mix_prob=0.0, freeze_noise=True)
    print(f"PHASE 3  testing on {args.data}  ({len(dataset)} events)")

    run = Run(args.out, None, args.data.parent.name, tag="test")
    payload = {"data": str(args.data), "n_events": len(dataset), "results": {}}

    if not args.baselines_only:
        import torch
        from patt_reco.train.loop import load_checkpoint

        device = pick_device(args.device)
        model, train_cfg, checkpoint = load_checkpoint(args.checkpoint, device)
        print(f"  checkpoint {args.checkpoint}")
        print(f"  trained {checkpoint['epoch'] + 1} epochs, "
              f"val mIoU {checkpoint['metrics'].get('val_miou')}")
        print(f"  device {describe_device(device)}")

        loader = make_loader(dataset, args.batch_size, False, args.num_workers)
        max_batches = -(-args.max_events // args.batch_size) if args.max_events else 0
        metrics, rows = evaluate_model(model, loader, device, max_batches=max_batches,
                                       per_event=True)
        payload["results"][train_cfg.model.name] = report(train_cfg.model.name, metrics,
                                                          show_confusion=True)
        payload["per_event"] = rows
        if args.previews:
            save_previews(run, dataset, model, device, args.previews)

    if args.baselines or args.baselines_only:
        for name, factory in BASELINES.items():
            metrics, _ = evaluate_baseline(factory(), dataset, args.baseline_events)
            payload["results"][name] = report(f"baseline: {name}", metrics)

    if len(payload["results"]) > 1:
        print("\n--- summary ---")
        print(f"{'model':<14}{'mIoU':>9}{'fg IoU':>9}{'pixel acc':>12}")
        for name, result in payload["results"].items():
            s = result["modes"]["exclusive"]
            print(f"{name:<14}{s['miou']:9.4f}{result['foreground']['exclusive']['iou']:9.4f}"
                  f"{s['pixel_accuracy']:12.4f}")

    path = run.save_json("test_report.json", payload)
    print(f"\nreport -> {path}")
    run.close()


if __name__ == "__main__":
    main()
