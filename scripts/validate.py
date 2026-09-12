#!/usr/bin/env python3
"""PHASE 4 - model validation on busy events and out-of-distribution data.

This is the acceptance test, and it is deliberately harder than anything the
model was trained on. Nothing here shares a distribution with the training set.

    python scripts/validate.py runs/<run>/best.pt
    python scripts/validate.py runs/<run>/best.pt --snr 0.5,1,2,4 --baselines

Four measurements, each answering a specific question:

  busy          L3, 10-40 objects: does it hold up when the event is crowded?
  multiplicity  mIoU binned by object count: does it *extrapolate* past the
                2-5 objects it trained on, or fall off a cliff?
  snr           noise scaled up and down: where does it break relative to the
                classical floor?
  ood / shift   L4a unseen geometry, L4b unseen detector: did it learn the
                pattern, or this detector?

A single number would hide all four, so everything is reported differentially.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from patt_reco.cliutil import describe_device, pick_device
from patt_reco.config import N_CLASSES
from patt_reco.eval.differential import DifferentialMetrics
from patt_reco.eval.evaluate import evaluate_baseline, evaluate_model
from patt_reco.eval.metrics_sem import SemanticMetrics
from patt_reco.models.baselines import BASELINES
from patt_reco.train.tracking import Run

MULTIPLICITY_EDGES = np.array([0, 5, 10, 15, 20, 25, 30, 40, 60, 1000])
OCCUPANCY_EDGES = np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.01])


def evaluate_dataset(model, dataset, device, batch_size, num_workers, max_events=0,
                     differential=True):
    """Overall metrics plus mIoU binned by multiplicity and occupancy."""
    import torch
    from patt_reco.dataset.torch_dataset import make_loader

    loader = make_loader(dataset, batch_size, False, num_workers)
    metrics = SemanticMetrics(N_CLASSES)
    by_multiplicity = DifferentialMetrics("n objects", MULTIPLICITY_EDGES)
    by_occupancy = DifferentialMetrics("occupancy", OCCUPANCY_EDGES)

    max_batches = -(-max_events // batch_size) if max_events else 0
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(loader):
            if max_batches and step >= max_batches:
                break
            views = batch["views"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model({"views": views})["sem_logits"]
            pred = logits.argmax(dim=2).cpu().numpy()
            true = batch["semantic"].numpy()
            contrib = batch["n_contrib"].numpy()
            metrics.update(pred, true, contrib)

            if differential:
                for b in range(pred.shape[0]):
                    cov = dataset.reader.covariates(int(batch["index"][b]))
                    by_multiplicity.update(cov["n_objects"], pred[b], true[b], contrib[b])
                    by_occupancy.update(cov["occupancy"], pred[b], true[b], contrib[b])

    return metrics, by_multiplicity, by_occupancy


def snr_sweep(model, dataset_factory, device, scales, batch_size, num_workers,
              max_events) -> dict:
    """Sweep the noise amplitude. `scale` multiplies every noise component."""
    import torch
    from patt_reco.dataset.torch_dataset import make_loader

    out = {}
    for scale in scales:
        dataset = dataset_factory(noise_scale=scale)
        loader = make_loader(dataset, batch_size, False, num_workers)
        metrics, _ = evaluate_model(model, loader, device,
                                    max_batches=-(-max_events // batch_size) if max_events else 0)
        summary = metrics.summary("exclusive")
        out[f"{scale:g}"] = {
            "noise_scale": scale,
            "miou": summary["miou"],
            "foreground_iou": metrics.foreground_summary("exclusive")["iou"],
        }
        print(f"  noise x{scale:<5g}  mIoU {summary['miou']:.4f}   "
              f"fg IoU {out[f'{scale:g}']['foreground_iou']:.4f}")
    return out


def plot_curves(run, payload) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [("multiplicity", "objects in event"), ("occupancy", "occupancy"),
              ("snr", "noise scale (x nominal)")]
    available = [(k, label) for k, label in panels if k in payload]
    if not available:
        return

    fig, axes = plt.subplots(1, len(available), figsize=(4.6 * len(available), 3.8),
                             squeeze=False)
    for ax, (key, xlabel) in zip(axes[0], available):
        if key == "snr":
            entries = sorted(payload[key].values(), key=lambda d: d["noise_scale"])
            x = [e["noise_scale"] for e in entries]
            ax.plot(x, [e["miou"] for e in entries], "o-", label="mIoU")
            ax.plot(x, [e["foreground_iou"] for e in entries], "s--", label="foreground IoU")
            ax.set_xscale("log")
        else:
            curve = payload[key]
            ax.plot(curve["centre"], curve["miou"], "o-", label="mIoU")
            ax.plot(curve["centre"], curve["foreground_iou"], "s--", label="foreground IoU")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("IoU")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Phase 4 - differential validation", fontsize=11)
    fig.tight_layout()
    path = run.dir / "validation_curves.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  curves -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--busy", type=Path, default=Path("data/l3_busy/all"))
    parser.add_argument("--ood", type=Path, default=Path("data/l4_ood_shape/all"))
    parser.add_argument("--shift", type=Path, default=Path("data/l4_domain_shift/all"))
    parser.add_argument("--snr", default="0.5,1,2,4",
                        help="comma-separated noise scales, or empty to skip")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-events", type=int, default=1000, help="0 = all")
    parser.add_argument("--baselines", action="store_true")
    parser.add_argument("--baseline-events", type=int, default=200)
    parser.add_argument("--out", type=Path, default=Path("runs"))
    args = parser.parse_args()

    from patt_reco.dataset.torch_dataset import PattRecoDataset
    from patt_reco.train.loop import load_checkpoint

    device = pick_device(args.device)
    model, train_cfg, checkpoint = load_checkpoint(args.checkpoint, device)
    print(f"PHASE 4  validation on {describe_device(device)}")
    print(f"  checkpoint {args.checkpoint}  (val mIoU "
          f"{checkpoint['metrics'].get('val_miou')})")
    print(f"  trained on {train_cfg.data.train}\n")

    run = Run(args.out, None, "validate", tag="validate")
    payload = {"checkpoint": str(args.checkpoint), "trained_on": train_cfg.data.train}

    # --- busy events ------------------------------------------------------
    if args.busy.exists():
        print(f"[busy] {args.busy}")
        busy = PattRecoDataset(args.busy, augment=False, mix_prob=0.0, freeze_noise=True)
        metrics, by_mult, by_occ = evaluate_dataset(
            model, busy, device, args.batch_size, args.num_workers, args.max_events)
        print(metrics.format_table("exclusive"))
        print("\n" + by_mult.format_table())
        print("\n" + by_occ.format_table())
        payload["busy"] = metrics.to_dict()
        payload["multiplicity"] = by_mult.curve()
        payload["occupancy"] = by_occ.curve()

        if args.baselines:
            for name, factory in BASELINES.items():
                base_metrics, _ = evaluate_baseline(factory(), busy, args.baseline_events)
                fg = base_metrics.foreground_summary("exclusive")
                print(f"  baseline {name:<11} mIoU "
                      f"{base_metrics.summary('exclusive')['miou']:.4f}  fg IoU {fg['iou']:.4f}")
                payload.setdefault("busy_baselines", {})[name] = base_metrics.to_dict()

        # --- SNR sweep ----------------------------------------------------
        if args.snr.strip():
            scales = [float(x) for x in args.snr.split(",")]
            print("\n[snr sweep]")
            payload["snr"] = snr_sweep(
                model, lambda noise_scale: PattRecoDataset(
                    args.busy, augment=False, mix_prob=0.0, freeze_noise=True,
                    noise_scale=noise_scale),
                device, scales, args.batch_size, args.num_workers, args.max_events)
    else:
        print(f"[busy] {args.busy} not found - run scripts/generate.py --levels l3")

    # --- out of distribution ---------------------------------------------
    for key, path, label in (("ood_shape", args.ood, "unseen geometry"),
                             ("domain_shift", args.shift, "unseen detector")):
        if not path.exists():
            print(f"\n[{key}] {path} not found - skipping")
            continue
        print(f"\n[{key}] {path}  ({label})")
        dataset = PattRecoDataset(path, augment=False, mix_prob=0.0, freeze_noise=True)
        metrics, _, _ = evaluate_dataset(model, dataset, device, args.batch_size,
                                         args.num_workers, args.max_events,
                                         differential=False)
        print(metrics.format_table("exclusive"))
        payload[key] = metrics.to_dict()

    # --- headline ---------------------------------------------------------
    print("\n--- validation summary ---")
    print(f"{'set':<16}{'mIoU':>9}{'fg IoU':>9}")
    for key in ("busy", "ood_shape", "domain_shift"):
        if key in payload:
            s = payload[key]["modes"]["exclusive"]
            print(f"{key:<16}{s['miou']:9.4f}{payload[key]['foreground']['exclusive']['iou']:9.4f}")

    plot_curves(run, payload)
    path = run.save_json("validation_report.json", payload)
    print(f"report -> {path}")
    run.close()


if __name__ == "__main__":
    main()
