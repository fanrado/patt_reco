#!/usr/bin/env python3
"""Render event displays from a config or a generated dataset.

    python scripts/preview.py configs/data/l3_busy.yaml -n 4 -o /tmp/l3
    python scripts/preview.py data/l2_noise -n 4 -o /tmp/l2      # from disk
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

from patt_reco.config import load_yaml
from patt_reco.dataset.generate import generate_event
from patt_reco.dataset.io_hdf5 import DatasetReader
from patt_reco.viz.event_display import plot_event


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, help="a YAML config or a dataset directory")
    parser.add_argument("-n", "--n-events", type=int, default=4)
    parser.add_argument("-i", "--start", type=int, default=0)
    parser.add_argument("-o", "--out", type=Path, default=Path("previews"))
    parser.add_argument("--panels", default="adc,semantic,instance")
    parser.add_argument("--dpi", type=int, default=110)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    panels = tuple(args.panels.split(","))

    if args.source.is_dir():
        from patt_reco.config import DatasetConfig, _build
        reader = DatasetReader(args.source)
        # a dataset on disk carries the config it was made with
        noise_cfg = _build(DatasetConfig, reader.manifest["config"]).noise
        records = [reader[i] for i in range(args.start,
                                            min(args.start + args.n_events, len(reader)))]
    else:
        cfg = load_yaml(args.source)
        noise_cfg = cfg.noise
        records = [generate_event(cfg, i) for i in range(args.start, args.start + args.n_events)]

    for rec in records:
        fig = plot_event(rec, noise_cfg=noise_cfg, panels=panels)
        path = args.out / f"event_{rec.index:05d}.png"
        fig.savefig(path, dpi=args.dpi)
        fig.clf()
        print(f"  {path}  ({rec.n_objects} objects, occupancy {rec.meta.get('occupancy', 0):.1%})")


if __name__ == "__main__":
    main()
