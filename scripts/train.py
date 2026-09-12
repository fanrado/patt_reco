#!/usr/bin/env python3
"""PHASE 2 - model training.

    python scripts/train.py configs/train/base.yaml
    python scripts/train.py configs/train/smoke.yaml           # 2-minute wiring check
    python scripts/train.py configs/train/base.yaml --set optim.epochs=5
    python scripts/train.py configs/train/base.yaml --overfit   # the one test that matters

`--overfit` drives a single batch to near-zero loss. If that fails, nothing else
about a training run is worth reading -- it is the cheapest possible check that
the labels, the loss and the model agree with each other.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from patt_reco.cliutil import apply_overrides, describe_device, pick_device
from patt_reco.config import N_CLASSES
from patt_reco.dataset.torch_dataset import PattRecoDataset, make_loader
from patt_reco.losses.focal_dice import SemanticLoss, class_weights_from_dataset
from patt_reco.models.registry import build_model
from patt_reco.train.config import load_train_yaml
from patt_reco.train.loop import Trainer
from patt_reco.train.tracking import Run


def build_datasets(cfg):
    train = PattRecoDataset(cfg.data.train, augment=cfg.data.augment,
                            mix_prob=cfg.data.mix_prob,
                            max_tick_shift=cfg.data.max_tick_shift,
                            seed=cfg.run.seed, freeze_noise=False)
    val = PattRecoDataset(cfg.data.val, augment=False, mix_prob=0.0,
                          seed=cfg.run.seed, freeze_noise=True)
    if cfg.data.max_train_events:
        train.n_events = min(train.n_events, cfg.data.max_train_events)
    if cfg.data.max_val_events:
        val.n_events = min(val.n_events, cfg.data.max_val_events)
    return train, val


def overfit_one_batch(model, loss_fn, device, batch, steps: int = 400) -> tuple[float, float]:
    """Drive one batch to near-zero loss. Catches label/loss/masking bugs.

    Judged on the *reduction* as well as the final value: the absolute loss scale
    depends on the class weights, so a fixed threshold alone would mislabel a
    healthy small model as broken. A genuine bug plateaus early and high.
    """
    model.train().to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=3e-3)
    views = batch["views"].to(device)
    first = last = None
    for step in range(steps):
        optimiser.zero_grad(set_to_none=True)
        loss, parts = loss_fn(model({"views": views}), batch)
        loss.backward()
        optimiser.step()
        last = parts["loss"]
        if first is None:
            first = last
        if step % 40 == 0 or step == steps - 1:
            print(f"    step {step:4d}  loss {last:.5f}")
    reduction = 1.0 - last / max(first, 1e-9)
    print(f"  loss {first:.4f} -> {last:.4f}  ({100 * reduction:.1f}% reduction)")
    return last, reduction


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="PATH=VALUE")
    parser.add_argument("--overfit", action="store_true",
                        help="overfit a single batch instead of training")
    parser.add_argument("--overfit-steps", type=int, default=400)
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args()

    cfg = apply_overrides(load_train_yaml(args.config), args.overrides)
    device = pick_device(args.device)
    torch.manual_seed(cfg.run.seed)
    np.random.seed(cfg.run.seed)

    print(f"PHASE 2  training on {describe_device(device)}")
    train_set, val_set = build_datasets(cfg)
    print(f"  train {len(train_set)} events ({cfg.data.train})")
    print(f"  val   {len(val_set)} events ({cfg.data.val})")

    model = build_model(cfg.model.name, n_classes=N_CLASSES,
                        base_width=cfg.model.base_width, depth=cfg.model.depth,
                        in_channels=cfg.model.in_channels)
    print(f"  model {cfg.model.name}: {model.n_parameters / 1e6:.2f} M parameters")

    weights = class_weights_from_dataset(train_set.reader, cfg.loss.class_weight_events,
                                         N_CLASSES, cfg.loss.class_weighting)
    print("  class weights (" + cfg.loss.class_weighting + "): "
          + " ".join(f"{w:.2f}" for w in weights))
    loss_fn = SemanticLoss(N_CLASSES, cfg.loss.focal_gamma, cfg.loss.focal_weight,
                           cfg.loss.dice_weight, weights, cfg.loss.ambiguity_weighting)

    if args.overfit:
        loader = make_loader(train_set, cfg.data.batch_size, shuffle=False, num_workers=0)
        batch = next(iter(loader))
        print(f"\n  overfitting one batch of {batch['views'].shape[0]} events")
        final, reduction = overfit_one_batch(model, loss_fn, device, batch,
                                             args.overfit_steps)
        if final < 0.15 and reduction > 0.90:
            print("\n  PASS - the model can memorise this batch, so the labels, the "
                  "loss and the head agree")
        else:
            print("\n  FAIL - loss did not collapse. Either a real bug (labels, loss "
                  "masking, the head) or too little capacity / too few steps: retry "
                  "with --set model.base_width=32 --overfit-steps 700 before "
                  "suspecting the code.")
        return

    train_loader = make_loader(train_set, cfg.data.batch_size, True,
                               cfg.data.num_workers, drop_last=True)
    val_loader = make_loader(val_set, cfg.data.batch_size, False, cfg.data.num_workers)

    run = Run(cfg.run.out_dir, cfg, cfg.run.name, tag="train")
    run.save_json("class_weights.json", {"scheme": cfg.loss.class_weighting,
                                         "weights": weights.tolist()})
    print(f"  run -> {run.dir}\n")

    if args.resume:
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"])
        print(f"  resumed weights from {args.resume}")

    trainer = Trainer(cfg, model, loss_fn, train_loader, val_loader, device, run)
    result = trainer.fit()
    run.close()

    print(f"\nbest val mIoU {result['best_miou']:.4f}")
    print(f"checkpoint     {run.dir / 'best.pt'}")
    print(f"next: python scripts/test.py {run.dir / 'best.pt'} --data data/l2_noise/test")


if __name__ == "__main__":
    main()
