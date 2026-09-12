"""Out-of-distribution shapes, used only for validation.

These are deliberately absent from every training config. Each is labelled with
its *nearest in-distribution class*, so the question they pose is sharp: given a
geometry it has never seen, does the model degrade gracefully into the plausible
class, or does it hallucinate something confident and wrong?

    spiral      -> helix      (curvature that grows instead of staying constant)
    zigzag      -> track      (sharp kinks instead of smooth scattering)
    double_ring -> ring       (two concentric rings instead of one)
"""
from __future__ import annotations

import numpy as np

from ..config import HELIX, RING, TRACK
from .primitives import (PRIMITIVES, Primitive, landau_charge, orthonormal_basis,
                         random_direction, rotate_towards, unit)


class Spiral(Primitive):
    """Archimedean spiral: radius grows linearly with angle."""

    CLASS = HELIX
    ANCHOR = "origin"

    @classmethod
    def sample(cls, rng, gcfg, vol, **ov):
        return cls(
            origin=ov.get("origin", vol.sample_point(rng)[0]),
            direction=unit(np.asarray(ov.get("direction", random_direction(rng)), float)),
            normal=unit(np.asarray(ov.get("normal", random_direction(rng)), float)),
            r0=float(ov.get("r0", rng.uniform(1.0, 5.0))),
            growth=float(ov.get("growth", rng.uniform(0.5, 3.0))),   # cm per radian
            arc=float(ov.get("arc", rng.uniform(4.0, 12.0))),
        )

    def _curve(self, gcfg):
        u, v = orthonormal_basis(self.params["normal"])
        r0, growth, arc = self.params["r0"], self.params["growth"], self.params["arc"]
        # step in phi chosen so the arc-length step stays near gcfg.step_cm
        n = max(8, int(round(arc * (r0 + 0.5 * growth * arc) / gcfg.step_cm)))
        phi = np.linspace(0.0, arc, n)
        r = r0 + growth * phi
        points = (self.anchor - r0 * u
                  + r[:, None] * (np.cos(phi)[:, None] * u + np.sin(phi)[:, None] * v))
        ds = np.gradient(np.concatenate([[0.0], np.cumsum(
            np.linalg.norm(np.diff(points, axis=0), axis=1))]))
        return points, np.abs(ds)

    def deposit(self, rng, gcfg):
        points, ds = self._curve(gcfg)
        return points, landau_charge(rng, gcfg.dqdx_mip * ds, gcfg.landau_scale)

    def point_at(self, frac):
        return self.anchor


class Zigzag(Primitive):
    """Piecewise-linear track with sharp kinks -- not smooth scattering."""

    CLASS = TRACK
    ANCHOR = "origin"

    @classmethod
    def sample(cls, rng, gcfg, vol, **ov):
        return cls(
            origin=ov.get("origin", vol.sample_point(rng)[0]),
            direction=unit(np.asarray(ov.get("direction", random_direction(rng)), float)),
            length=float(ov.get("length", rng.uniform(15.0, 60.0))),
            n_kinks=int(ov.get("n_kinks", rng.integers(2, 7))),
            kink_rad=float(ov.get("kink_rad", rng.uniform(0.3, 1.0))),
        )

    def deposit(self, rng, gcfg):
        n_seg = self.params["n_kinks"] + 1
        seg_len = self.params["length"] / n_seg
        n_per = max(2, int(round(seg_len / gcfg.step_cm)))
        ds = seg_len / n_per

        position, direction = self.anchor, self.direction
        chunks = []
        for _ in range(n_seg):
            s = (np.arange(n_per) + 0.5) * ds
            chunks.append(position + s[:, None] * direction)
            position = position + seg_len * direction
            direction = rotate_towards(direction, self.params["kink_rad"],
                                       rng.uniform(0.0, 2.0 * np.pi))

        points = np.concatenate(chunks)
        mpv = np.full(len(points), gcfg.dqdx_mip * ds)
        return points, landau_charge(rng, mpv, gcfg.landau_scale)


class DoubleRing(Primitive):
    """Two concentric rings sharing a centre and a normal."""

    CLASS = RING
    ANCHOR = "centre"
    DIRECTIONAL = False

    @classmethod
    def sample(cls, rng, gcfg, vol, **ov):
        return cls(
            centre=ov.get("origin", ov.get("centre", vol.sample_point(rng)[0])),
            normal=unit(np.asarray(ov.get("normal", random_direction(rng)), float)),
            radius=float(ov.get("radius", rng.uniform(*gcfg.ring_radius_cm))),
            ratio=float(ov.get("ratio", rng.uniform(0.35, 0.7))),
            thickness=float(ov.get("thickness", rng.uniform(*gcfg.ring_thickness_cm))),
        )

    def deposit(self, rng, gcfg):
        u, v = orthonormal_basis(self.params["normal"])
        normal = np.asarray(self.params["normal"], float)
        points, charges = [], []
        for radius in (self.params["radius"], self.params["radius"] * self.params["ratio"]):
            n = max(8, int(round(2 * np.pi * radius / gcfg.step_cm)))
            phi = np.linspace(0.0, 2.0 * np.pi, n)
            ds = 2.0 * np.pi * radius / n
            r = radius + rng.normal(0.0, self.params["thickness"], n)
            points.append(self.anchor
                          + r[:, None] * (np.cos(phi)[:, None] * u + np.sin(phi)[:, None] * v)
                          + rng.normal(0.0, self.params["thickness"], n)[:, None] * normal)
            charges.append(landau_charge(rng, np.full(n, gcfg.dqdx_mip * ds),
                                         gcfg.landau_scale))
        return np.concatenate(points), np.concatenate(charges)

    def point_at(self, frac):
        return self.anchor


OOD_PRIMITIVES: dict[str, type[Primitive]] = {
    "spiral": Spiral,
    "zigzag": Zigzag,
    "double_ring": DoubleRing,
}

# registered so a config can name them; excluded from every training config's
# class weights, which is the only thing that keeps them out of training
PRIMITIVES.update(OOD_PRIMITIVES)
