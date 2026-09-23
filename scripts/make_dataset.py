#!/usr/bin/env python3
"""Build a labelled dataset from one or more shape configs.

    python scripts/make_dataset.py configs/data/tracks.yaml configs/data/showers.yaml -o data/out
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from patt_reco.cliutil import apply_overrides
from patt_reco.config import CLASS_NAMES, load_yaml
from patt_reco.dataset.build import SPLITS, build_dataset


def _dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("configs", type=Path, nargs="+",
                        help="one YAML config per shape source, e.g. "
                             "configs/data/tracks.yaml configs/data/showers.yaml")
    parser.add_argument("-o", "--out", type=Path, default=None,
                        help="output directory (default: data/<first config name>)")
    parser.add_argument("--shuffle-seed", type=int, default=0,
                        help="seed for the within-split shuffle")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="PATH=VALUE",
                        help="override a config field, e.g. --set render.height=32. "
                             "Repeatable. Applied to EVERY source: a SourceConfig "
                             "always carries both a track and a shower block, so "
                             "--set shower.max_nodes=7 simply has no effect on a "
                             "track source. This is how the difficulty sweep is built.")
    args = parser.parse_args()

    sources = [apply_overrides(load_yaml(path), args.overrides)
               for path in args.configs]
    out = args.out or Path("data") / sources[0].name

    for cfg, path in zip(sources, args.configs):
        overrides = f"  --set {' --set '.join(args.overrides)}" if args.overrides else ""
        print(f"{cfg.name:<12} {cfg.kind:<7} label {cfg.label}  seed {cfg.seed}  "
              f"{cfg.n_train}/{cfg.n_val}/{cfg.n_test}  ({path}){overrides}")

    start = time.time()
    build_dataset(sources, out, shuffle_seed=args.shuffle_seed)
    elapsed = time.time() - start

    total = 0
    for split in SPLITS:
        with np.load(out / f"{split}.npz") as data:
            labels, images = data["labels"], data["images"]
            total += len(labels)
            balance = "  ".join(
                f"{CLASS_NAMES.get(int(c), c)} {n}"
                for c, n in zip(*np.unique(labels, return_counts=True)))
            shape = "x".join(str(d) for d in images.shape[1:]) if len(images) else "-"
            print(f"  {split:<5} {len(labels):>7} images  {shape:<9} {balance}")

    size = _dir_size(out)
    print(f"\n{total} images in {out}, {size / 1e6:.1f} MB, in {elapsed:.1f} s "
          f"({1000 * elapsed / max(total, 1):.1f} ms/image)")
    print(f"next: python scripts/preview.py {args.configs[0]} -n 8")


if __name__ == "__main__":
    main()
