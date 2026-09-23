"""The 3D region objects are generated in.

A plain axis-aligned box, configured directly rather than derived from anything
else. It bounds where an object may start and how far it may run before the
rasteriser projects it to an image.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Volume:
    """Axis-aligned box. `x`, `y` and `z` are just axes; none is special."""

    xlo: float
    xhi: float
    ylo: float
    yhi: float
    zlo: float
    zhi: float

    @classmethod
    def cube(cls, size: float = 1.0) -> "Volume":
        """Cube of side `size` centred on the origin -- the default box."""
        half = 0.5 * float(size)
        return cls(-half, half, -half, half, -half, half)

    @property
    def lo(self) -> np.ndarray:
        return np.array([self.xlo, self.ylo, self.zlo])

    @property
    def hi(self) -> np.ndarray:
        return np.array([self.xhi, self.yhi, self.zhi])

    @property
    def span(self) -> np.ndarray:
        return self.hi - self.lo

    @property
    def diagonal(self) -> float:
        return float(np.linalg.norm(self.span))

    def shrink(self, frac: float) -> "Volume":
        """Concentric box scaled by `frac` -- used for object start points."""
        mid = 0.5 * (self.lo + self.hi)
        half = 0.5 * self.span * frac
        return Volume(*np.stack([mid - half, mid + half], axis=1).T.ravel(order="F"))

    def sample_point(self, rng: np.random.Generator, n: int = 1) -> np.ndarray:
        return rng.uniform(self.lo, self.hi, size=(n, 3))

    def contains(self, points: np.ndarray) -> np.ndarray:
        return np.all((points >= self.lo) & (points <= self.hi), axis=-1)
