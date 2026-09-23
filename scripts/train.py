#!/usr/bin/env python3
"""Train the classifier from a training config.

    python scripts/train.py configs/train/base.yaml
    python scripts/train.py configs/train/base.yaml --overfit
    python scripts/train.py configs/train/base.yaml --device cpu
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from patt_reco.cliutil import apply_overrides
from patt_reco.dataset.torch_dataset import ImageDataset
from patt_reco.models import CNN
from patt_reco.train import Run, Trainer, load_train_yaml


def pick_device(requested: str) -> torch.device:
    """Choose a device and say out loud why, rather than falling back quietly."""
    available = torch.cuda.is_available()

    if requested == "cpu":
        print("device: cpu (requested)")
        return torch.device("cpu")

    if requested == "cuda":
        if not available:
            print("device: cpu -- CUDA was requested but torch.cuda.is_available() "
                  "is False; falling back")
            return torch.device("cpu")
        print(f"device: cuda (requested) -- {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")

    if available:
        print(f"device: cuda (auto) -- {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    print("device: cpu (auto) -- no CUDA device available")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path, help="a YAML training config")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="PATH=VALUE",
                        help="override a training config field, e.g. "
                             "--set optim.epochs=8. Repeatable. Applied before the "
                             "run directory is created, so config.json records what "
                             "was actually trained.")
    parser.add_argument("--overfit", action="store_true",
                        help="drive one batch to near-zero loss and exit; "
                             "a wiring check, so no run directory is created")
    args = parser.parse_args()

    cfg = apply_overrides(load_train_yaml(args.config), args.overrides)
    device = pick_device(args.device)

    torch.manual_seed(cfg.run.seed)
    np.random.seed(cfg.run.seed)

    root = Path(cfg.data.root)
    train_set = ImageDataset(root / "train.npz")
    val_set = ImageDataset(root / "val.npz")
    train_loader = DataLoader(train_set, batch_size=cfg.data.batch_size, shuffle=True,
                              num_workers=cfg.data.num_workers)
    val_loader = DataLoader(val_set, batch_size=cfg.data.batch_size,
                            num_workers=cfg.data.num_workers)

    height, width = train_set.images.shape[1:]
    m = cfg.model
    model = CNN(height, width, n_filters=m.n_filters, kernel_size=m.kernel_size,
                pool=m.pool, hidden=m.hidden, dropout=m.dropout,
                n_blocks=m.n_blocks)
    print(f"data:   {len(train_set)} train, {len(val_set)} val, {height}x{width}")
    print(f"model:  {model.n_parameters():,} parameters")

    if args.overfit:
        # no Run: a wiring check should not leave a run directory behind
        trainer = Trainer(cfg, model, train_loader, val_loader, device, None)
        final = trainer.overfit_one_batch()
        ok = final < 0.05
        print(f"\noverfit check: final loss {final:.6f} -- "
              f"{'reached near zero, plumbing OK' if ok else 'DID NOT converge; the plumbing is wrong'}")
        return

    run = Run(cfg)
    print(f"run:    {run.dir}")
    best = Trainer(cfg, model, train_loader, val_loader, device, run).train()

    print(f"\nbest validation accuracy {best:.4f}")
    print(f"next: python scripts/evaluate.py {run.dir / 'best.pt'} "
          f"--data {root / 'test.npz'}")


if __name__ == "__main__":
    main()
