#!/usr/bin/env python3
"""PHASE 1 - data generation.

Builds the whole difficulty ladder, with train/val/test splits where a level is
trainable and a single split where it is validation-only.

Splits are separate *seeds*, not slices of one dataset: since (seed, index)
determines an event, different seeds cannot overlap, so leakage is impossible by
construction rather than by careful bookkeeping.

    python scripts/generate.py --all -j 8
    python scripts/generate.py --levels l0,l2 -j 8
    python scripts/generate.py --all -n 2000 -j 8        # small smoke ladder
"""
from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

from patt_reco.cliutil import apply_overrides
from patt_reco.config import config_hash, load_yaml
from patt_reco.dataset.generate import generate_dataset

# level -> (config file, is it trainable?)
LADDER = {
    "l0": ("l0_single.yaml", True),
    "l1": ("l1_multi.yaml", True),
    "l2": ("l2_noise.yaml", True),
    "l3": ("l3_busy.yaml", False),
    "l4_ood": ("l4_ood_shape.yaml", False),
    "l4_shift": ("l4_domain_shift.yaml", False),
}

# split -> (fraction of n_events, seed offset)
SPLITS = {"train": (0.80, 0), "val": (0.10, 1), "test": (0.10, 2)}


def build_level(level: str, config_dir: Path, out_root: Path, workers: int,
                n_events: int | None, overrides, force: bool) -> list[dict]:
    filename, trainable = LADDER[level]
    cfg = apply_overrides(load_yaml(config_dir / filename), overrides)
    if n_events is not None:
        cfg = replace(cfg, n_events=n_events)

    plan = SPLITS if trainable else {"all": (1.0, 0)}
    made = []
    for split, (fraction, seed_offset) in plan.items():
        count = max(1, int(round(fraction * cfg.n_events)))
        shard = max(1, min(cfg.shard_size, -(-count // max(workers, 1))))
        split_cfg = replace(cfg, n_events=count, seed=cfg.seed + seed_offset,
                            shard_size=shard)
        out = out_root / cfg.name / split

        if (out / "manifest.json").exists() and not force:
            print(f"  {cfg.name}/{split:<5} exists, skipping (use --force to rebuild)")
            made.append({"level": level, "split": split, "path": str(out), "skipped": True})
            continue

        print(f"  {cfg.name}/{split:<5} {count:>7} events  seed {split_cfg.seed}  "
              f"config {config_hash(split_cfg)}")
        generate_dataset(split_cfg, out, workers=workers, verbose=False)
        size = sum(p.stat().st_size for p in out.glob("*.h5"))
        print(f"          -> {size / 1e6:8.1f} MB  ({size / count / 1024:.1f} kB/event)")
        made.append({"level": level, "split": split, "path": str(out),
                     "n_events": count, "bytes": size})
    return made


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--levels", default=None,
                        help=f"comma-separated subset of {','.join(LADDER)}")
    parser.add_argument("--all", action="store_true", help="build every level")
    parser.add_argument("-o", "--out", type=Path, default=Path("data"))
    parser.add_argument("-c", "--configs", type=Path, default=Path("configs/data"))
    parser.add_argument("-j", "--workers", type=int, default=1)
    parser.add_argument("-n", "--n-events", type=int, default=None,
                        help="override the event count per level (before splitting)")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="PATH=VALUE", help="config override, repeatable")
    parser.add_argument("--force", action="store_true", help="rebuild existing splits")
    args = parser.parse_args()

    if args.all:
        levels = list(LADDER)
    elif args.levels:
        levels = [x.strip() for x in args.levels.split(",")]
    else:
        parser.error("pass --all or --levels")

    unknown = [x for x in levels if x not in LADDER]
    if unknown:
        parser.error(f"unknown level(s) {unknown}; choose from {list(LADDER)}")

    print(f"PHASE 1  data generation -> {args.out}")
    started, made = time.time(), []
    for level in levels:
        made += build_level(level, args.configs, args.out, args.workers,
                            args.n_events, args.overrides, args.force)

    total = sum(m.get("bytes", 0) for m in made)
    events = sum(m.get("n_events", 0) for m in made)
    print(f"\n{events} events across {len(made)} splits, {total / 1e9:.2f} GB, "
          f"{time.time() - started:.0f} s")
    print("next: python scripts/train.py configs/train/base.yaml")


if __name__ == "__main__":
    main()
