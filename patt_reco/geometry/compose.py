"""Event composition: place primitives in the volume and make them interact.

Random placement almost never produces the topologies that are actually hard --
shared vertices and small-angle crossings -- so both are injected deliberately
and at a controlled rate. That is what makes the crossing-angle sweep in the
validation plan a real measurement rather than a fishing expedition.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import COSMIC, EventConfig, GeometryConfig, TRACK
from .primitives import (PRIMITIVES, Primitive, LineTrack, ScatteredTrack,
                         random_direction, rotate_towards, unit)
from .volume import Volume


@dataclass
class DepositedObject:
    """One object, its 3D charge deposits, and the truth that goes with it."""

    obj_id: int
    cls: int
    points: np.ndarray          # [N, 3] cm
    dq: np.ndarray              # [N] ke-
    primitive: Primitive
    parent_id: int = -1         # delta rays / vertex siblings point at their parent

    @property
    def total_charge(self) -> float:
        return float(self.dq.sum())


@dataclass
class Event3D:
    objects: list[DepositedObject]
    volume: Volume
    meta: dict = field(default_factory=dict)

    @property
    def n_objects(self) -> int:
        return len(self.objects)


def _choose_classes(rng, n: int, weights: dict[str, float]) -> list[str]:
    names = list(weights)
    probs = np.array([weights[k] for k in names], dtype=float)
    probs = probs / probs.sum()
    return list(rng.choice(names, size=n, p=probs))


def _make_cosmic(rng, gcfg: GeometryConfig, vol: Volume) -> Primitive:
    """A straight track that traverses the entire volume, cosmic-ray style."""
    # zenith distribution ~ cos^2, measured from -y ("down")
    cos_z = rng.uniform(0.0, 1.0) ** (1.0 / 3.0)
    sin_z = np.sqrt(max(0.0, 1.0 - cos_z**2))
    azi = rng.uniform(0.0, 2.0 * np.pi)
    d = unit(np.array([sin_z * np.cos(azi), -cos_z, sin_z * np.sin(azi)]))

    through = vol.sample_point(rng)[0]
    reach = vol.diagonal
    track = LineTrack(origin=through - reach * d, direction=d,
                      length=2.0 * reach, stopping=False)
    track.cls = COSMIC
    return track


def _attach_delta(rng, gcfg: GeometryConfig, parent: Primitive) -> Primitive:
    """Short secondary branching off a parent track at a random arc-length."""
    frac = rng.uniform(0.1, 0.9)
    origin = parent.point_at(frac)
    # deltas come off at a wide angle to the parent
    d = rotate_towards(parent.direction, rng.uniform(0.6, 1.6), rng.uniform(0, 2 * np.pi))
    delta = ScatteredTrack(
        origin=origin, direction=d,
        length=float(rng.uniform(*gcfg.delta_len_cm)),
        momentum=float(rng.uniform(20.0, 120.0)),
        stopping=True,
    )
    delta.cls = TRACK
    return delta


def compose_event(rng, gcfg: GeometryConfig, ecfg: EventConfig, vol: Volume) -> Event3D:
    """Sample a full 3D event: primitives, vertices, crossings, deltas, cosmics."""
    fiducial = vol.shrink(ecfg.fiducial_frac)
    n = int(rng.integers(ecfg.n_objects[0], ecfg.n_objects[1] + 1))
    class_names = _choose_classes(rng, n, ecfg.class_weights)

    prims: list[Primitive] = []
    parents = [-1] * n

    # --- shared-vertex group (interaction-like; k = 2 is a vee) ------------
    vertex_members: list[int] = []
    if n >= 2 and rng.random() < ecfg.p_vertex:
        k = min(n, int(rng.integers(ecfg.vertex_size[0], ecfg.vertex_size[1] + 1)))
        vertex_members = list(range(k))
        vertex_point = fiducial.sample_point(rng)[0]
    else:
        vertex_point = None

    for i, name in enumerate(class_names):
        overrides = {}
        if i in vertex_members:
            overrides["origin"] = vertex_point
        prims.append(PRIMITIVES[name].sample(rng, gcfg, fiducial, **overrides))

    if vertex_members:
        for i in vertex_members[1:]:
            parents[i] = vertex_members[0]

    # --- forced crossing at a controlled angle -----------------------------
    movable = [i for i, p in enumerate(prims)
               if p.DIRECTIONAL and i not in vertex_members[1:]]
    if len(movable) >= 2 and rng.random() < ecfg.p_crossing:
        i, j = rng.choice(movable, size=2, replace=False)
        angle = np.deg2rad(rng.uniform(*ecfg.crossing_angle_deg))
        new_dir = rotate_towards(prims[i].direction, angle, rng.uniform(0, 2 * np.pi))
        prims[j].set_direction(new_dir)
        target = prims[i].point_at(rng.uniform(0.2, 0.8))
        prims[j].translate(target - prims[j].point_at(rng.uniform(0.2, 0.8)))

    # --- delta rays --------------------------------------------------------
    n_primary = len(prims)
    for i in range(n_primary):
        if prims[i].cls in (TRACK,) or isinstance(prims[i], (LineTrack, ScatteredTrack)):
            if rng.random() < ecfg.p_delta:
                prims.append(_attach_delta(rng, gcfg, prims[i]))
                parents.append(i)

    # --- cosmic background -------------------------------------------------
    n_cos = int(rng.integers(ecfg.n_cosmics[0], ecfg.n_cosmics[1] + 1))
    for _ in range(n_cos):
        prims.append(_make_cosmic(rng, gcfg, vol))
        parents.append(-1)

    # --- deposit -----------------------------------------------------------
    objects = []
    for obj_id, prim in enumerate(prims):
        points, dq = prim.deposit(rng, gcfg)
        objects.append(DepositedObject(obj_id=obj_id, cls=prim.cls, points=points,
                                       dq=dq, primitive=prim, parent_id=parents[obj_id]))

    return Event3D(objects=objects, volume=vol,
                   meta={"n_primary": n_primary, "n_cosmics": n_cos,
                         "has_vertex": bool(vertex_members)})
