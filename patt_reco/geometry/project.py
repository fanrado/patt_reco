"""Orthographic rasterisation of a 3D point cloud into one image.

The minimal replacement for the deleted detector projection: no charge
transport, no response, no digitisation. Points are flattened onto the plane
perpendicular to the viewing direction, scaled to fit the frame, and summed
into pixels.
"""
from __future__ import annotations

import numpy as np

from .base import orthonormal_basis


def project(points: np.ndarray, values: np.ndarray, cfg) -> np.ndarray:
    """Rasterise `points` weighted by `values` into a [height, width] image.

    The object is fitted to the frame with a `cfg.margin` fractional border,
    preserving aspect ratio, and the result is peak-normalised to 1.0.
    """
    height, width = int(cfg.height), int(cfg.width)
    image = np.zeros(height * width, dtype=np.float64)

    points = np.asarray(points, dtype=float).reshape(-1, 3)
    values = np.asarray(values, dtype=float).ravel()
    if points.size == 0:
        return image.reshape(height, width).astype(np.float32)

    # flatten onto the plane perpendicular to the viewing direction
    u, v = orthonormal_basis(np.asarray(cfg.view, dtype=float))
    xy = np.stack([points @ u, points @ v], axis=1)

    # fit into the frame, same scale on both axes so shapes are not stretched
    border = float(cfg.margin) * min(height, width)
    usable = np.array([width - 2.0 * border, height - 2.0 * border])
    usable = np.maximum(usable, 1.0)
    span = np.maximum(xy.max(axis=0) - xy.min(axis=0), 1e-12)
    scale = float(np.min(usable / span))

    centre = 0.5 * (xy.max(axis=0) + xy.min(axis=0))
    cols = np.rint((xy[:, 0] - centre[0]) * scale + 0.5 * width).astype(int)
    rows = np.rint((xy[:, 1] - centre[1]) * scale + 0.5 * height).astype(int)
    np.clip(cols, 0, width - 1, out=cols)
    np.clip(rows, 0, height - 1, out=rows)

    flat = np.bincount(rows * width + cols, weights=values, minlength=height * width)
    image = flat[: height * width]

    peak = image.max()
    if peak > 0:
        image = image / peak
    return image.reshape(height, width).astype(np.float32)
