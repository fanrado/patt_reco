"""3D primitives.

Every primitive emits a *point cloud with charge weights*, not a rasterised
image: rendering resolution is a detector parameter, not a geometry parameter.

The physics knobs here (Landau fluctuations, Bragg peaks, Highland scattering,
fractal shower branching) are present because they change the visible *pattern*.
Nothing is tuned to a particular detector.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import moyal

from ..config import (BLOB, GeometryConfig, HELIX, RING, SCATTERED, SHOWER,
                      TRACK)
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


def landau_charge(rng, mpv: np.ndarray, scale_frac: float) -> np.ndarray:
    """Moyal-distributed charge with the given most-probable value.

    Moyal is the standard cheap stand-in for Landau; its mode sits exactly at
    `loc`, which is what makes `mpv` mean what it says.
    """
    mpv = np.atleast_1d(np.asarray(mpv, dtype=float))
    scale = np.maximum(scale_frac * mpv, 1e-9)
    dq = moyal.rvs(loc=mpv, scale=scale, size=mpv.shape, random_state=rng)
    return np.maximum(dq, 0.0)


# --------------------------------------------------------------------------- #
# base class
# --------------------------------------------------------------------------- #


class Primitive:
    """Base class. Subclasses fill in CLASS, ANCHOR, `sample` and `deposit`."""

    CLASS: int = TRACK
    ANCHOR: str = "origin"
    DIRECTIONAL: bool = True   # can compose.py re-aim it (for forced crossings)?

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
        if self.DIRECTIONAL:
            self.params["direction"] = unit(np.asarray(d, dtype=float))
        return self

    def point_at(self, frac: float) -> np.ndarray:
        """A point on the object, `frac` of the way along it. Used to build
        deliberate crossings without having to rasterise first."""
        length = float(self.params.get("length", 0.0))
        return self.anchor + frac * length * self.direction

    # -- interface ---------------------------------------------------------
    @classmethod
    def sample(cls, rng, gcfg: GeometryConfig, vol: Volume, **overrides) -> "Primitive":
        raise NotImplementedError

    def deposit(self, rng, gcfg: GeometryConfig) -> tuple[np.ndarray, np.ndarray]:
        """Return (points[N, 3] in cm, dq[N] in ke-)."""
        raise NotImplementedError

    def truth_params(self) -> dict:
        """Analytic parameters stored as per-object truth."""
        return {k: np.asarray(v, dtype=np.float32).ravel() for k, v in self.params.items()
                if not isinstance(v, bool)}


# --------------------------------------------------------------------------- #
# straight track
# --------------------------------------------------------------------------- #


def _bragg_scale(residual: np.ndarray, gcfg: GeometryConfig) -> np.ndarray:
    """dE/dx enhancement vs residual range for a stopping particle."""
    with np.errstate(divide="ignore"):
        scale = (gcfg.bragg_ref_cm / np.maximum(residual, 1e-3)) ** gcfg.bragg_exponent
    return np.clip(scale, 1.0, gcfg.bragg_max)


class LineTrack(Primitive):
    """Straight ionising track, optionally stopping (Bragg peak at the end)."""

    CLASS = TRACK
    ANCHOR = "origin"

    @classmethod
    def sample(cls, rng, gcfg, vol, **ov):
        return cls(
            origin=ov.get("origin", vol.sample_point(rng)[0]),
            direction=unit(np.asarray(ov.get("direction", random_direction(rng)), float)),
            length=float(ov.get("length", rng.uniform(*gcfg.track_len_cm))),
            stopping=bool(ov.get("stopping", rng.random() < gcfg.p_stopping)),
        )

    def deposit(self, rng, gcfg):
        length = self.params["length"]
        n = max(2, int(round(length / gcfg.step_cm)))
        ds = length / n
        s = (np.arange(n) + 0.5) * ds
        pts = self.anchor + s[:, None] * self.direction

        mpv = np.full(n, gcfg.dqdx_mip * ds)
        if self.params["stopping"]:
            mpv = mpv * _bragg_scale(length - s, gcfg)
        return pts, landau_charge(rng, mpv, gcfg.landau_scale)


# --------------------------------------------------------------------------- #
# multiple-scattered track
# --------------------------------------------------------------------------- #


class ScatteredTrack(Primitive):
    """Track with Highland multiple Coulomb scattering -- the realistic 'curve'.

    The walk is integrated in the *initial* frame using the small-angle
    approximation, which keeps it fully vectorised. That is accurate to a few
    percent for total deflections up to ~0.5 rad; beyond that the trajectory is
    still a plausible wiggly track, which is all the label requires.
    """

    CLASS = SCATTERED
    ANCHOR = "origin"

    @classmethod
    def sample(cls, rng, gcfg, vol, **ov):
        return cls(
            origin=ov.get("origin", vol.sample_point(rng)[0]),
            direction=unit(np.asarray(ov.get("direction", random_direction(rng)), float)),
            length=float(ov.get("length", rng.uniform(*gcfg.scat_len_cm))),
            momentum=float(ov.get("momentum", rng.uniform(*gcfg.scat_momentum_mev))),
            stopping=bool(ov.get("stopping", rng.random() < gcfg.p_stopping)),
        )

    def deposit(self, rng, gcfg):
        length, p = self.params["length"], self.params["momentum"]
        n = max(2, int(round(length / gcfg.step_cm)))
        ds = length / n

        x_over_x0 = ds / gcfg.rad_len_cm
        theta0 = (13.6 / p) * np.sqrt(x_over_x0) * max(0.1, 1.0 + 0.038 * np.log(x_over_x0))

        d0 = self.direction
        u, v = orthonormal_basis(d0)
        theta_u = np.cumsum(rng.normal(0.0, theta0, n))
        theta_v = np.cumsum(rng.normal(0.0, theta0, n))
        dirs = unit(d0 + theta_u[:, None] * u + theta_v[:, None] * v)

        pts = self.anchor + np.cumsum(dirs * ds, axis=0)

        mpv = np.full(n, gcfg.dqdx_mip * ds)
        if self.params["stopping"]:
            mpv = mpv * _bragg_scale(length - (np.arange(n) + 0.5) * ds, gcfg)
        return pts, landau_charge(rng, mpv, gcfg.landau_scale)


# --------------------------------------------------------------------------- #
# helix
# --------------------------------------------------------------------------- #


class Helix(Primitive):
    """Constant-field helix: an arc in the bending plane, a wiggle elsewhere.

    `origin` is the *start* of the trajectory (not the circle centre), so that
    placement behaves like every other primitive even for a 120 cm radius.
    """

    CLASS = HELIX
    ANCHOR = "origin"

    @classmethod
    def sample(cls, rng, gcfg, vol, **ov):
        return cls(
            origin=ov.get("origin", vol.sample_point(rng)[0]),
            direction=unit(np.asarray(ov.get("direction", random_direction(rng)), float)),
            axis=unit(np.asarray(ov.get("axis", random_direction(rng)), float)),
            radius=float(ov.get("radius", rng.uniform(*gcfg.helix_radius_cm))),
            pitch=float(ov.get("pitch", rng.uniform(*gcfg.helix_pitch_cm))),
            arc=float(ov.get("arc", rng.uniform(*gcfg.helix_arc_rad))),
            sense=float(ov.get("sense", rng.choice([-1.0, 1.0]))),
        )

    def _frame(self):
        """Circle basis: `u` points from the centre to the start point."""
        axis = unit(self.params["axis"])
        d0 = self.direction
        # tangent at the start, projected perpendicular to the axis
        t = unit(d0 - np.dot(d0, axis) * axis)
        if np.linalg.norm(t) < 1e-6:
            t, _ = orthonormal_basis(axis)
        sense = self.params["sense"]
        u = unit(np.cross(t, axis)) * sense   # centre -> start
        v = np.cross(axis, u)
        centre = self.anchor - self.params["radius"] * u
        return centre, u, v, axis

    def deposit(self, rng, gcfg):
        radius, pitch, arc = (self.params["radius"], self.params["pitch"], self.params["arc"])
        centre, u, v, axis = self._frame()

        ds_dphi = np.hypot(radius, pitch)
        n = max(2, int(round(arc * ds_dphi / gcfg.step_cm)))
        phi = np.linspace(0.0, arc, n)
        ds = arc * ds_dphi / n

        pts = (centre
               + radius * np.cos(phi)[:, None] * u
               + radius * np.sin(phi)[:, None] * v
               + pitch * phi[:, None] * axis)

        mpv = np.full(n, gcfg.dqdx_mip * ds)
        return pts, landau_charge(rng, mpv, gcfg.landau_scale)

    def point_at(self, frac: float) -> np.ndarray:
        centre, u, v, axis = self._frame()
        phi = frac * self.params["arc"]
        return (centre + self.params["radius"] * (np.cos(phi) * u + np.sin(phi) * v)
                + self.params["pitch"] * phi * axis)


# --------------------------------------------------------------------------- #
# ring
# --------------------------------------------------------------------------- #


class Ring(Primitive):
    """Cherenkov-ring-like circle with a finite thickness; projects to an ellipse."""

    CLASS = RING
    ANCHOR = "centre"
    DIRECTIONAL = False

    @classmethod
    def sample(cls, rng, gcfg, vol, **ov):
        return cls(
            centre=ov.get("origin", ov.get("centre", vol.sample_point(rng)[0])),
            normal=unit(np.asarray(ov.get("normal", random_direction(rng)), float)),
            radius=float(ov.get("radius", rng.uniform(*gcfg.ring_radius_cm))),
            thickness=float(ov.get("thickness", rng.uniform(*gcfg.ring_thickness_cm))),
            arc_frac=float(ov.get("arc_frac", rng.uniform(*gcfg.ring_arc_frac))),
            phi0=float(ov.get("phi0", rng.uniform(0.0, 2.0 * np.pi))),
        )

    def deposit(self, rng, gcfg):
        radius, thickness = self.params["radius"], self.params["thickness"]
        arc = 2.0 * np.pi * self.params["arc_frac"]
        u, v = orthonormal_basis(self.params["normal"])

        n = max(8, int(round(radius * arc / gcfg.step_cm)))
        phi = self.params["phi0"] + np.linspace(0.0, arc, n)
        ds = radius * arc / n

        radial = rng.normal(0.0, thickness, n)
        out_of_plane = rng.normal(0.0, thickness, n)
        r = radius + radial
        pts = (self.anchor
               + r[:, None] * (np.cos(phi)[:, None] * u + np.sin(phi)[:, None] * v)
               + out_of_plane[:, None] * np.asarray(self.params["normal"], float))

        mpv = np.full(n, gcfg.dqdx_mip * ds)
        return pts, landau_charge(rng, mpv, gcfg.landau_scale)

    def point_at(self, frac: float) -> np.ndarray:
        return self.anchor


# --------------------------------------------------------------------------- #
# shower
# --------------------------------------------------------------------------- #


class Shower(Primitive):
    """Recursively branching cone.

    Energy is conserved: each segment deposits `E * (1 - exp(-L / X0))` and the
    remainder is split between two children, so the total charge converges to
    `E0 * ke_per_mev` as the cascade terminates.
    """

    CLASS = SHOWER
    ANCHOR = "origin"

    @classmethod
    def sample(cls, rng, gcfg, vol, **ov):
        return cls(
            origin=ov.get("origin", vol.sample_point(rng)[0]),
            direction=unit(np.asarray(ov.get("direction", random_direction(rng)), float)),
            energy=float(ov.get("energy", rng.uniform(*gcfg.shower_e0_mev))),
        )

    def deposit(self, rng, gcfg):
        stack = [(self.anchor, self.direction, self.params["energy"])]
        all_pts, all_dq = [], []
        n_nodes = 0

        while stack and n_nodes < gcfg.shower_max_nodes:
            pos, d, energy = stack.pop()
            if energy < gcfg.shower_emin_mev:
                continue
            n_nodes += 1

            length = gcfg.rad_len_cm * rng.uniform(*gcfg.shower_seg_x0)
            e_dep = energy * (1.0 - np.exp(-length / gcfg.rad_len_cm))

            n = max(2, int(round(length / gcfg.step_cm)))
            s = (np.arange(n) + 0.5) * (length / n)
            all_pts.append(pos + s[:, None] * d)
            all_dq.append(landau_charge(
                rng, np.full(n, e_dep * gcfg.shower_ke_per_mev / n), gcfg.landau_scale))

            end = pos + length * d
            frac = rng.beta(gcfg.shower_split_frac, gcfg.shower_split_frac)
            remaining = energy - e_dep
            for child_frac in (frac, 1.0 - frac):
                theta = rng.uniform(*gcfg.shower_open_rad)
                child_d = rotate_towards(d, theta, rng.uniform(0.0, 2.0 * np.pi))
                stack.append((end, child_d, remaining * child_frac))

        return np.concatenate(all_pts), np.concatenate(all_dq)


# --------------------------------------------------------------------------- #
# blob
# --------------------------------------------------------------------------- #


class Blob(Primitive):
    """Compact 3D Gaussian deposit: a vertex, a nuclear interaction, a stub."""

    CLASS = BLOB
    ANCHOR = "centre"
    DIRECTIONAL = False

    @classmethod
    def sample(cls, rng, gcfg, vol, **ov):
        sigma = ov.get("sigma")
        if sigma is None:
            base = rng.uniform(*gcfg.blob_sigma_cm)
            sigma = base * rng.uniform(0.7, 1.3, 3)   # mildly anisotropic
        return cls(
            centre=ov.get("origin", ov.get("centre", vol.sample_point(rng)[0])),
            sigma=np.asarray(sigma, dtype=float),
            charge=float(ov.get("charge", rng.uniform(*gcfg.blob_charge_ke))),
        )

    def deposit(self, rng, gcfg):
        n = gcfg.blob_points
        pts = self.anchor + rng.normal(0.0, 1.0, (n, 3)) * self.params["sigma"]
        mpv = np.full(n, self.params["charge"] / n)
        return pts, landau_charge(rng, mpv, gcfg.landau_scale)

    def point_at(self, frac: float) -> np.ndarray:
        return self.anchor


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #

PRIMITIVES: dict[str, type[Primitive]] = {
    "track": LineTrack,
    "scattered": ScatteredTrack,
    "helix": Helix,
    "ring": Ring,
    "shower": Shower,
    "blob": Blob,
}
