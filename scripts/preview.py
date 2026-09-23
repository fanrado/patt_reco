#!/usr/bin/env python3
"""Render sample images from one shape config, to eyeball it.

    python scripts/preview.py configs/data/tracks.yaml -n 8 -o previews/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

from patt_reco.config import load_yaml
from patt_reco.dataset.generate import generate_event
from patt_reco.viz.display import plot_grid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path, help="a YAML shape config")
    parser.add_argument("-n", "--n-images", type=int, default=8)
    parser.add_argument("-o", "--out", type=Path, default=Path("previews"))
    parser.add_argument("--ncols", type=int, default=8)
    parser.add_argument("--dpi", type=int, default=110)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    pairs = [generate_event(cfg, i) for i in range(args.n_images)]
    images = [image for image, _ in pairs]
    labels = [label for _, label in pairs]

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{cfg.name}.png"

    fig = plot_grid(images, labels, ncols=args.ncols)
    fig.savefig(path, dpi=args.dpi)
    fig.clf()

    filled = sum(float((image > 0).mean()) for image in images) / max(len(images), 1)
    print(f"  {path}  ({len(images)} {cfg.kind} images, "
          f"{cfg.render.height}x{cfg.render.width}, mean fill {filled:.1%})")


if __name__ == "__main__":
    main()
