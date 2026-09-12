"""Dataset generation: seed in, events out.

`(config.seed, event_index)` fully determines an event -- the readout sampling,
the composition and the rendering all draw from one `Generator` seeded from that
pair. Generation is therefore embarrassingly parallel and bit-identical no
matter how many workers run it.
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ..config import DatasetConfig, config_hash, load_yaml, to_dict
from ..detector.projection import render_event
from ..detector.readout import sample_readout
from ..geometry.compose import compose_event
from .io_hdf5 import ShardWriter
from .schema import EventRecord


def event_rng(seed: int, index: int) -> np.random.Generator:
    """The one place event randomness comes from."""
    return np.random.default_rng([seed, index])


def generate_event(cfg: DatasetConfig, index: int) -> EventRecord:
    """Sample, render and package a single event.

    The order of draws (readout, then composition, then rendering) is part of
    the reproducibility contract: changing it changes every existing dataset.
    """
    rng = event_rng(cfg.seed, index)
    readout = sample_readout(rng, cfg.detector)
    event = compose_event(rng, cfg.geometry, cfg.event, readout.volume)
    rendered = render_event(rng, event, readout, cfg.detector)
    return EventRecord.from_rendered(rendered, index=index, seed=cfg.seed)


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def _write_shard(args) -> tuple[str, int]:
    cfg, out_dir, shard_id, start, stop = args
    path = Path(out_dir) / f"shard_{shard_id:04d}.h5"
    shape = (len(cfg.detector.view_angles_deg), cfg.detector.n_channels, cfg.detector.n_ticks)
    with ShardWriter(path, shape, np.asarray(cfg.detector.view_angles_deg)) as writer:
        for index in range(start, stop):
            writer.append(generate_event(cfg, index))
    return path.name, stop - start


def generate_dataset(cfg: DatasetConfig, out_dir, workers: int = 1,
                     verbose: bool = True) -> Path:
    """Generate a full dataset directory: shards plus a manifest."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bounds = list(range(0, cfg.n_events, cfg.shard_size)) + [cfg.n_events]
    jobs = [(cfg, out_dir, i, bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_write_shard, jobs))
    else:
        results = []
        for job in jobs:
            results.append(_write_shard(job))
            if verbose:
                print(f"  wrote {results[-1][0]} ({results[-1][1]} events)", flush=True)

    manifest = {
        "name": cfg.name,
        "config": to_dict(cfg),
        "config_hash": config_hash(cfg),
        "n_events": cfg.n_events,
        "shards": [name for name, _ in results],
        "git_sha": _git_sha(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    if verbose:
        total = sum(p.stat().st_size for p in out_dir.glob("*.h5"))
        print(f"{cfg.n_events} events -> {out_dir}  "
              f"({total / 1e6:.1f} MB, {total / max(cfg.n_events, 1) / 1024:.1f} kB/event)")
    return out_dir


def generate_from_yaml(path, out_dir=None, workers: int = 1) -> Path:
    cfg = load_yaml(path)
    out_dir = Path(out_dir) if out_dir else Path("data") / cfg.name
    return generate_dataset(cfg, out_dir, workers=workers)
