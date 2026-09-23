#!/usr/bin/env python3
"""Build a labelled dataset from one or more shape configs.

    python scripts/make_dataset.py configs/data/tracks.yaml configs/data/showers.yaml -o data/out
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

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
    args = parser.parse_args()

    sources = [load_yaml(path) for path in args.configs]
    out = args.out or Path("data") / sources[0].name

    for cfg, path in zip(sources, args.configs):
        print(f"{cfg.name:<12} {cfg.kind:<7} label {cfg.label}  seed {cfg.seed}  "
              f"{cfg.n_train}/{cfg.n_val}/{cfg.n_test}  ({path})")

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
    print(f"next: python scripts/preview.py {out}")


if __name__ == "__main__":
    main()
