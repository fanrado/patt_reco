"""Dataset generation: seed in, one labelled image out.

`(config.seed, index)` fully determines an image -- the shape sampling, the
deposition and the rendering all draw from one `Generator` seeded from that
pair. Generation is therefore embarrassingly parallel and bit-identical no
matter how many workers run it.
"""
from __future__ import annotations

import numpy as np

from ..config import SourceConfig
from ..geometry.project import project
from ..geometry.shower import Shower
from ..geometry.track import Track
from ..geometry.volume import Volume


def event_rng(seed: int, index: int) -> np.random.Generator:
    """The one place event randomness comes from."""
    return np.random.default_rng([seed, index])


def generate_event(cfg: SourceConfig, index: int) -> tuple[np.ndarray, int]:
    """Sample one object, rasterise it, and return (image, label)."""
    rng = event_rng(cfg.seed, index)
    vol = Volume.cube()

    if cfg.kind == "track":
        primitive = Track.sample(rng, cfg.track, vol)
        shape_cfg = cfg.track
    elif cfg.kind == "shower":
        primitive = Shower.sample(rng, cfg.shower, vol)
        shape_cfg = cfg.shower
    else:
        raise ValueError(
            f"unknown kind {cfg.kind!r}: expected 'track' or 'shower'")

    points, values = primitive.deposit(rng, shape_cfg)
    image = project(points, values, cfg.render)
    return image, int(cfg.label)
