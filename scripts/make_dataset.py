#!/usr/bin/env python3
"""Generate a dataset from a YAML config.

    python scripts/make_dataset.py configs/data/l2_noise.yaml -o data/l2_noise -j 8
    python scripts/make_dataset.py configs/data/l0_single.yaml -n 200   # quick look
"""
from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

from patt_reco.cliutil import apply_override
from patt_reco.config import config_hash, load_yaml
from patt_reco.dataset.generate import generate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path, help="YAML config, e.g. configs/data/l2_noise.yaml")
    parser.add_argument("-o", "--out", type=Path, default=None,
                        help="output directory (default: data/<config name>)")
    parser.add_argument("-n", "--n-events", type=int, default=None,
                        help="override the event count, for a quick look")
    parser.add_argument("-j", "--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None, help="override the dataset seed")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="PATH=VALUE",
                        help="override any config field, e.g. --set event.n_objects='[5,5]'. "
                             "Repeatable; this is how the stress sweeps are built.")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    if args.n_events is not None:
        # keep enough shards to occupy every worker -- otherwise a small -n run
        # collapses to a single shard and `-j` does nothing
        per_worker = max(1, -(-args.n_events // max(args.workers, 1)))
        cfg = replace(cfg, n_events=args.n_events,
                      shard_size=max(1, min(cfg.shard_size, per_worker)))
    if args.seed is not None:
        cfg = replace(cfg, seed=args.seed)
    for override in args.overrides:
        cfg = apply_override(cfg, override)

    out = args.out or Path("data") / cfg.name
    print(f"{cfg.name}: {cfg.n_events} events, seed {cfg.seed}, config {config_hash(cfg)}")

    start = time.time()
    generate_dataset(cfg, out, workers=args.workers)
    elapsed = time.time() - start
    print(f"done in {elapsed:.1f} s ({1000 * elapsed / max(cfg.n_events, 1):.1f} ms/event)")


if __name__ == "__main__":
    main()
