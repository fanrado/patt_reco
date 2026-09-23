"""Turn one or more source configs into a labelled dataset on disk.

Each source is an independent config file, so the sources are validated
against each other before anything is generated: they must render at the same
size and must not claim the same label. That check is what makes separate
tracks.yaml and showers.yaml files safe to combine.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..config import CLASS_NAMES, SourceConfig, to_dict
from .generate import generate_event

SPLITS = ("train", "val", "test")


def _validate(sources: list[SourceConfig]) -> None:
    if not sources:
        raise ValueError("no sources given: nothing to build")

    first = sources[0]
    shape = (first.render.height, first.render.width)
    for cfg in sources[1:]:
        if (cfg.render.height, cfg.render.width) != shape:
            raise ValueError(
                f"sources disagree on image size: {first.name!r} renders "
                f"{shape[0]}x{shape[1]} but {cfg.name!r} renders "
                f"{cfg.render.height}x{cfg.render.width}")

    seen: dict[int, str] = {}
    for cfg in sources:
        if cfg.label in seen:
            raise ValueError(
                f"sources {seen[cfg.label]!r} and {cfg.name!r} both claim "
                f"label {cfg.label}")
        seen[cfg.label] = cfg.name


def build_dataset(sources: list[SourceConfig], out_dir, shuffle_seed: int = 0) -> Path:
    """Generate every split, shuffle it, and write the npz files and meta."""
    sources = list(sources)
    _validate(sources)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    counts = {}
    for split in SPLITS:
        images, labels = [], []
        for cfg in sources:
            n = getattr(cfg, f"n_{split}")
            # a distinct index range per split, so no two splits ever share an
            # (seed, index) pair and therefore never share an image
            offset = sum(getattr(cfg, f"n_{s}") for s in SPLITS[:SPLITS.index(split)])
            for i in range(n):
                image, label = generate_event(cfg, offset + i)
                images.append(image)
                labels.append(label)

        counts[split] = len(images)
        if images:
            images = np.stack(images).astype(np.float32)
            labels = np.asarray(labels, dtype=np.uint8)
            order = np.random.default_rng(shuffle_seed).permutation(len(images))
            images, labels = images[order], labels[order]
        else:
            height, width = sources[0].render.height, sources[0].render.width
            images = np.empty((0, height, width), dtype=np.float32)
            labels = np.empty(0, dtype=np.uint8)

        np.savez_compressed(out_dir / f"{split}.npz", images=images, labels=labels)

    meta = {
        "sources": [to_dict(cfg) for cfg in sources],
        "class_names": {str(k): v for k, v in CLASS_NAMES.items()},
        "shuffle_seed": shuffle_seed,
        "counts": counts,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    return out_dir
