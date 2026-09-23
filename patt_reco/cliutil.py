"""Dotted-path overrides for the frozen config dataclasses.

    --set track.curvature='[0.0, 0.5]' --set render.height=32

Values are parsed as YAML, so numbers, booleans, strings and lists all work,
and lists become tuples because the config dataclasses use tuples for ranges.
The config is rebuilt with `dataclasses.replace` rather than mutated, so it
stays frozen and hashable.
"""
from __future__ import annotations

from dataclasses import fields, is_dataclass, replace

import yaml


def _replace_at(obj, path: list[str], value, spec: str):
    """Rebuild `obj` with the field at `path` set to `value`."""
    name = path[0]
    if not is_dataclass(obj):
        raise SystemExit(
            f"--set {spec}: cannot descend into {name!r}, "
            f"{type(obj).__name__} is not a config block")

    known = [f.name for f in fields(obj)]
    if name not in known:
        raise SystemExit(
            f"--set {spec}: {type(obj).__name__} has no field {name!r} "
            f"(available: {', '.join(known)})")

    if len(path) == 1:
        return replace(obj, **{name: value})
    return replace(obj, **{name: _replace_at(getattr(obj, name), path[1:], value, spec)})


def apply_override(cfg, spec: str):
    """Apply one `dotted.path=value` override, returning a new config."""
    if "=" not in spec:
        raise SystemExit(f"--set {spec}: expected 'dotted.path=value', no '=' found")

    path, _, raw = spec.partition("=")
    value = yaml.safe_load(raw)
    if isinstance(value, list):
        value = tuple(value)        # the config dataclasses use tuples for ranges

    if not path:
        raise SystemExit(f"--set {spec}: the field path is empty")
    return _replace_at(cfg, path.split("."), value, spec)


def apply_overrides(cfg, specs):
    """Apply every override in turn, returning the final config."""
    for spec in specs:
        cfg = apply_override(cfg, spec)
    return cfg
