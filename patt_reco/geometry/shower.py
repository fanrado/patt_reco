"""The shower primitive: a branching spray of scattered points.

A cascade grown from an apex: each branch walks a short segment, scatters its
points transversely, then splits in two. The result is a cone-like cloud whose
shape -- not any pattern of intensities -- is what distinguishes it from a
track.

Each branch carries a dimensionless weight, purely a shape control: it scales
the branch's segment length so the two sides of a split run different
distances and the cascade looks lopsided rather than like a symmetric fan.
"""
from __future__ import annotations

import numpy as np

from ..config import SHOWER
from .base import Primitive, orthonormal_basis, random_direction, rotate_towards, unit
from .volume import Volume


class Shower(Primitive):
    """Recursively branching cloud of scattered points."""

    CLASS = SHOWER
    ANCHOR = "origin"

    @classmethod
    def sample(cls, rng, cfg, vol: Volume, **ov) -> "Shower":
        return cls(
            origin=np.asarray(ov.get("origin", vol.sample_point(rng)[0]), dtype=float),
            direction=unit(np.asarray(
                ov.get("direction", random_direction(rng)), dtype=float)),
        )

    def deposit(self, rng, cfg) -> tuple[np.ndarray, np.ndarray]:
        stack = [(self.anchor, unit(self.direction), 1.0)]
        all_points, all_values = [], []
        n_nodes = 0

        while stack and n_nodes < cfg.max_nodes:
            pos, d, weight = stack.pop()
            n_nodes += 1

            length = rng.uniform(*cfg.seg_len) * weight
            n = max(2, int(round(length / cfg.step)))
            s = np.linspace(0.0, length, n)
            points = pos + s[:, None] * d

            # scatter each point off the segment axis, so the branch reads as a
            # spray of points rather than a clean line
            u, v = orthonormal_basis(d)
            sigma = rng.uniform(*cfg.spread)
            offsets = rng.normal(0.0, sigma, size=(n, 2))
            points = points + offsets[:, :1] * u + offsets[:, 1:] * v

            all_points.append(points)
            all_values.append(rng.uniform(cfg.value[0], cfg.value[1], size=n))

            end = pos + length * d
            frac = rng.beta(cfg.split_frac, cfg.split_frac)
            for child_weight in (weight * frac, weight * (1.0 - frac)):
                theta = rng.uniform(*cfg.open_angle)
                child_d = rotate_towards(d, theta, rng.uniform(0.0, 2.0 * np.pi))
                stack.append((end, child_d, child_weight))

        return np.concatenate(all_points), np.concatenate(all_values)
