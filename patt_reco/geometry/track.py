"""The track primitive: a single line, straight or gently curved.

This replaces the old LineTrack and Helix. Curvature is a plain geometric
parameter -- the radius of the arc -- not a constant-field physics model, and
the intensity along the line is deliberately flat: a track is recognisable by
its *shape*, never by a pattern of values along it.
"""
from __future__ import annotations

import numpy as np

from ..config import TRACK
from .base import Primitive, orthonormal_basis, random_direction, unit
from .volume import Volume

# below this curvature the arc is indistinguishable from a straight segment,
# and 1/curvature would overflow
_STRAIGHT = 1e-12


class Track(Primitive):
    """A line of constant-value points, bending in a single plane."""

    CLASS = TRACK
    ANCHOR = "origin"

    @classmethod
    def sample(cls, rng, cfg, vol: Volume, **ov) -> "Track":
        direction = unit(np.asarray(
            ov.get("direction", random_direction(rng)), dtype=float))

        if "bend_axis" in ov:
            bend_axis = unit(np.asarray(ov["bend_axis"], dtype=float))
        else:
            # a uniformly random unit vector in the plane perpendicular to
            # `direction` -- the direction the arc bends towards
            u, v = orthonormal_basis(direction)
            phi = rng.uniform(0.0, 2.0 * np.pi)
            bend_axis = unit(np.cos(phi) * u + np.sin(phi) * v)

        return cls(
            origin=np.asarray(ov.get("origin", vol.sample_point(rng)[0]), dtype=float),
            direction=direction,
            length=float(ov.get("length", rng.uniform(*cfg.length))),
            curvature=float(ov.get("curvature", rng.uniform(*cfg.curvature))),
            bend_axis=bend_axis,
        )

    def deposit(self, rng, cfg) -> tuple[np.ndarray, np.ndarray]:
        origin = self.anchor
        direction = unit(self.direction)
        bend_axis = unit(np.asarray(self.params["bend_axis"], dtype=float))
        length = float(self.params["length"])
        curvature = float(self.params["curvature"])

        step = float(cfg.step)
        n = max(2, int(np.floor(length / step)) + 1)
        s = np.arange(n, dtype=float) * step          # arc length along the line

        if abs(curvature) < _STRAIGHT:
            points = origin + s[:, None] * direction
        else:
            # arc of radius 1/curvature, starting at `origin`, tangent to
            # `direction`, curving towards `bend_axis`
            angle = curvature * s
            along = np.sin(angle) / curvature
            across = (1.0 - np.cos(angle)) / curvature
            points = origin + along[:, None] * direction + across[:, None] * bend_axis

        values = np.full(n, float(cfg.value), dtype=float)
        return points, values
