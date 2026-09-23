"""Configuration dataclasses for the patt_reco generator.

Every knob lives here. Nothing in the generator has a hidden default: a config
object plus a seed fully determines an event.

One config file describes exactly one shape source. `tracks.yaml` and
`showers.yaml` are independent files, so a `SourceConfig` must stand on its own
-- there is no shared parent config to inherit from.

Configs are plain frozen dataclasses so they hash cleanly; `load_yaml` builds a
nested config from a YAML file and `to_dict`/`config_hash` serialise it back for
the dataset manifest.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, fields, is_dataclass, asdict
from typing import Any, get_type_hints

import yaml

# --------------------------------------------------------------------------- #
# classes
# --------------------------------------------------------------------------- #

TRACK = 0
SHOWER = 1

N_CLASSES = 2

CLASS_NAMES = {
    TRACK: "track",
    SHOWER: "shower",
}


# --------------------------------------------------------------------------- #
# shape sources
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class TrackConfig:
    """A single track: one continuous, optionally curved line of points."""

    length: tuple = (5.0, 60.0)      # arc length, sampled per event
    curvature: tuple = (0.0, 0.02)   # 1/radius; 0.0 is a perfectly straight line
    step: float = 0.05               # deposition step along the trajectory
    value: float = 1.0               # constant intensity carried by every point


@dataclass(frozen=True)
class ShowerConfig:
    """A branching cascade, deposited as scattered points rather than lines."""

    max_nodes: int = 400
    seg_len: tuple = (2.0, 6.0)        # length of one segment between splits
    open_angle: tuple = (0.05, 0.30)   # half-opening angle at a split, radians
    split_frac: float = 2.0            # Beta(a, a) split ratio; a = this
    spread: tuple = (0.1, 0.6)         # transverse jitter applied per point
    value: tuple = (0.5, 1.5)          # per-point intensity range, sampled for
                                       # variation only -- carries no physical
                                       # interpretation
    step: float = 0.05                 # deposition step along a segment


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class RenderConfig:
    """Orthographic rasterisation of the 3D points into one grayscale image."""

    height: int = 128
    width: int = 128
    margin: float = 0.05             # empty border kept around the object, as a
                                     # fraction of the shorter image side
    view: tuple = (0.0, 0.0, 1.0)    # the single viewing direction


# --------------------------------------------------------------------------- #
# top level
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class SourceConfig:
    """One shape source, fully self-contained."""

    name: str = "unnamed"
    kind: str = "track"              # "track" | "shower"
    label: int = TRACK
    seed: int = 0

    n_train: int = 8000
    n_val: int = 1000
    n_test: int = 1000

    track: TrackConfig = field(default_factory=TrackConfig)
    shower: ShowerConfig = field(default_factory=ShowerConfig)
    render: RenderConfig = field(default_factory=RenderConfig)


# --------------------------------------------------------------------------- #
# (de)serialisation
# --------------------------------------------------------------------------- #

def _build(cls, data: dict):
    """Recursively construct a (nested) dataclass from a plain dict.

    Annotations are resolved with `get_type_hints`, which evaluates them in the
    *defining module's* namespace, so nested dataclasses declared in another
    module still resolve instead of silently staying plain dicts.
    """
    try:
        type_by_name = get_type_hints(cls)
    except Exception:                       # unresolvable forward reference
        type_by_name = {f.name: f.type for f in fields(cls)}

    kwargs = {}
    known = {f.name for f in fields(cls)}
    for key, value in data.items():
        if key not in known:
            raise KeyError(f"{cls.__name__} has no field {key!r}")
        target = type_by_name.get(key)
        if isinstance(target, str):
            target = globals().get(target, target)
        if is_dataclass(target) and isinstance(value, dict):
            kwargs[key] = _build(target, value)
        elif isinstance(value, list):
            kwargs[key] = tuple(value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def load_yaml(path) -> SourceConfig:
    with open(path) as handle:
        data = yaml.safe_load(handle) or {}
    return _build(SourceConfig, data)


def to_dict(cfg) -> dict[str, Any]:
    return asdict(cfg)


def config_hash(cfg) -> str:
    """Stable 12-hex-character digest of a config, used to name datasets."""
    blob = json.dumps(to_dict(cfg), sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]
