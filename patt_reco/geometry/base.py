"""Shared base for the 3D primitives.

Every primitive emits a *point cloud with per-point values*, not a rasterised
image: rendering resolution belongs to the renderer, not to the geometry.

Each image holds exactly one object, so there is no composition machinery here
-- a primitive only has to know how to sample itself and how to deposit itself.
"""
from __future__ import annotations

import numpy as np

from ..config import TRACK
from .volume import Volume

# --------------------------------------------------------------------------- #
# small vector helpers
# --------------------------------------------------------------------------- #


def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-12)


def random_direction(rng: np.random.Generator) -> np.ndarray:
    """Uniform on the unit sphere."""
    cos_theta = rng.uniform(-1.0, 1.0)
    phi = rng.uniform(0.0, 2.0 * np.pi)
    sin_theta = np.sqrt(1.0 - cos_theta**2)
    return np.array([sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta])


def orthonormal_basis(d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors spanning the plane perpendicular to `d`."""
    d = unit(np.asarray(d, dtype=float))
    helper = np.array([1.0, 0.0, 0.0]) if abs(d[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = unit(np.cross(d, helper))
    v = np.cross(d, u)
    return u, v


def rotate_towards(d: np.ndarray, theta: float, azimuth: float) -> np.ndarray:
    """Tilt `d` by polar angle `theta` at the given azimuth around it."""
    u, v = orthonormal_basis(d)
    return unit(np.cos(theta) * d + np.sin(theta) * (np.cos(azimuth) * u + np.sin(azimuth) * v))


# --------------------------------------------------------------------------- #
# base class
# --------------------------------------------------------------------------- #


class Primitive:
    """Base class. Subclasses fill in CLASS, ANCHOR, `sample` and `deposit`."""

    CLASS: int = TRACK
    ANCHOR: str = "origin"

    def __init__(self, **params):
        self.params = dict(params)
        self.cls = self.CLASS

    # -- introspection -----------------------------------------------------
    def __repr__(self) -> str:
        keys = ", ".join(f"{k}={np.round(v, 3) if np.ndim(v) else round(float(v), 3)}"
                         for k, v in self.params.items() if k != "origin")
        return f"{type(self).__name__}({keys})"

    @property
    def anchor(self) -> np.ndarray:
        return np.asarray(self.params[self.ANCHOR], dtype=float)

    @property
    def direction(self) -> np.ndarray:
        return np.asarray(self.params.get("direction", [0.0, 0.0, 1.0]), dtype=float)

    # -- placement ---------------------------------------------------------
    def translate(self, delta: np.ndarray) -> "Primitive":
        self.params[self.ANCHOR] = self.anchor + np.asarray(delta, dtype=float)
        return self

    def set_direction(self, d: np.ndarray) -> "Primitive":
        self.params["direction"] = unit(np.asarray(d, dtype=float))
        return self

    # -- interface ---------------------------------------------------------
    @classmethod
    def sample(cls, rng, cfg, vol: Volume, **overrides) -> "Primitive":
        raise NotImplementedError

    def deposit(self, rng, cfg) -> tuple[np.ndarray, np.ndarray]:
        """Return (points[N, 3] float, values[N] float).

        `values` are plain per-point intensities: no charge, no energy, no
        units of any kind.
        """
        raise NotImplementedError
