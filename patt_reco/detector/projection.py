"""Rasterise a 3D event into 2D views, and build the per-pixel truth with it.

Two decisions here shape every downstream metric, so they are stated explicitly:

1. **Diffusion is applied once, in 3D, as a stochastic jitter of sub-deposits**
   -- not as a blur of the finished image. The electron cloud diffuses once and
   is then seen by every plane, so all views stay mutually consistent, and each
   deposit keeps its object identity (a blur would mix objects and destroy the
   instance labels).

2. **Labels are defined on the deposited charge, before the response
   convolution.** That is the physical truth (ionisation), and it means the
   digitised image legitimately carries signal -- including the negative lobes of
   a bipolar response -- at pixels labelled empty. Undoing that is part of the
   task, and it is exactly what has to transfer when the response changes.

The overlap rule: a pixel fed by several objects is labelled with its
*highest-charge* contributor, `n_contrib` records how many objects touched it,
and the full contribution list is kept for those pixels so evaluation can
exclude, down-weight or fully score the ambiguous ones.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import DetectorConfig, EMPTY
from ..geometry.compose import Event3D
from .readout import Readout

PARAM_NAMES = ("x", "y", "z", "dx", "dy", "dz", "length", "radius", "charge", "energy")
N_PARAMS = len(PARAM_NAMES)


def param_vector(prim) -> np.ndarray:
    """Fixed-width analytic truth for an object; NaN where a field is n/a."""
    out = np.full(N_PARAMS, np.nan, dtype=np.float32)
    anchor = prim.anchor
    out[0:3] = anchor
    if prim.DIRECTIONAL and "direction" in prim.params:
        out[3:6] = prim.direction
    for name, slot in (("length", 6), ("radius", 7), ("charge", 8), ("energy", 9)):
        if name in prim.params:
            out[slot] = float(prim.params[name])
    return out


@dataclass
class ObjectRecord:
    obj_id: int
    cls: int
    parent_id: int
    params: np.ndarray                      # [N_PARAMS]
    total_charge: float
    n_pixels_dominant: np.ndarray           # [n_views] pixels this object owns
    n_pixels_touched: np.ndarray            # [n_views] pixels it deposits in

    @property
    def visible(self) -> np.ndarray:
        return self.n_pixels_dominant > 0


@dataclass
class RenderedEvent:
    """Everything one event contains, before noise."""

    charge: np.ndarray          # [V, C, T] float32, deposited charge in ke-
    adc: np.ndarray             # [V, C, T] float32, after response + digitisation
    semantic: np.ndarray        # [V, C, T] uint8
    instance: np.ndarray        # [V, C, T] int16, -1 where empty
    n_contrib: np.ndarray       # [V, C, T] uint8
    contrib: dict               # sparse truth for ambiguous pixels only
    objects: list[ObjectRecord]
    readout: Readout
    meta: dict = field(default_factory=dict)

    @property
    def n_views(self) -> int:
        return self.charge.shape[0]

    @property
    def occupancy(self) -> float:
        return float((self.charge > 0).mean())


def _transport(rng, points: np.ndarray, dq: np.ndarray, readout: Readout,
               substeps: int) -> tuple[np.ndarray, np.ndarray]:
    """Attenuate and diffuse one object's deposits, once, in 3D."""
    if substeps > 1:
        points = np.repeat(points, substeps, axis=0)
        dq = np.repeat(dq, substeps) / substeps

    drift = np.clip(points[:, 0], 0.0, None)
    charge = dq * np.exp(-drift / readout.attenuation_cm)

    root = np.sqrt(drift)
    sigma_l = readout.diffusion_l * root
    sigma_t = readout.diffusion_t * root

    jittered = points.copy()
    jittered[:, 0] += rng.normal(0.0, 1.0, len(points)) * sigma_l
    jittered[:, 1] += rng.normal(0.0, 1.0, len(points)) * sigma_t
    jittered[:, 2] += rng.normal(0.0, 1.0, len(points)) * sigma_t
    return jittered, charge


def render_event(rng, event: Event3D, readout: Readout,
                 cfg: DetectorConfig) -> RenderedEvent:
    n_views = len(readout.views)
    n_ch, n_tk = readout.shape
    n_pix = n_ch * n_tk
    objects = event.objects

    # transport happens once per object, shared by all views
    transported = [_transport(rng, o.points, o.dq, readout, cfg.diffusion_substeps)
                   for o in objects]

    charge = np.zeros((n_views, n_ch, n_tk), dtype=np.float32)
    semantic = np.zeros((n_views, n_ch, n_tk), dtype=np.uint8)
    instance = np.full((n_views, n_ch, n_tk), -1, dtype=np.int16)
    n_contrib = np.zeros((n_views, n_ch, n_tk), dtype=np.uint8)

    n_dominant = np.zeros((len(objects), n_views), dtype=np.int32)
    n_touched = np.zeros((len(objects), n_views), dtype=np.int32)

    contrib_view, contrib_pixel, contrib_obj, contrib_q = [], [], [], []

    for v, view in enumerate(readout.views):
        best_q = np.zeros(n_pix, dtype=np.float32)
        best_k = np.full(n_pix, -1, dtype=np.int32)
        total = np.zeros(n_pix, dtype=np.float32)
        counts = np.zeros(n_pix, dtype=np.uint8)

        per_obj_pixels, per_obj_q = [], []

        for k, (jittered, q) in enumerate(transported):
            chan, tick = view.project(jittered)
            ci = np.floor(chan).astype(np.int64)
            ti = np.floor(tick).astype(np.int64)
            inside = (ci >= 0) & (ci < n_ch) & (ti >= 0) & (ti < n_tk)
            if not inside.any():
                per_obj_pixels.append(np.empty(0, dtype=np.int64))
                per_obj_q.append(np.empty(0, dtype=np.float32))
                continue

            flat = ci[inside] * n_tk + ti[inside]
            img = np.bincount(flat, weights=q[inside], minlength=n_pix).astype(np.float32)
            hit = np.flatnonzero(img)

            per_obj_pixels.append(hit)
            per_obj_q.append(img[hit])

            counts[hit] += 1
            total[hit] += img[hit]
            better = img[hit] > best_q[hit]
            winners = hit[better]
            best_q[winners] = img[winners]
            best_k[winners] = k
            n_touched[k, v] = hit.size

        charge[v] = total.reshape(n_ch, n_tk)
        n_contrib[v] = counts.reshape(n_ch, n_tk)

        filled = best_k >= 0
        inst_flat = np.full(n_pix, -1, dtype=np.int16)
        sem_flat = np.zeros(n_pix, dtype=np.uint8)
        if filled.any():
            winner = best_k[filled]
            inst_flat[filled] = np.array([objects[k].obj_id for k in winner], dtype=np.int16)
            sem_flat[filled] = np.array([objects[k].cls for k in winner], dtype=np.uint8)
            for k in range(len(objects)):
                n_dominant[k, v] = int(np.count_nonzero(winner == k))
        instance[v] = inst_flat.reshape(n_ch, n_tk)
        semantic[v] = sem_flat.reshape(n_ch, n_tk)

        # keep the full contribution list only where it adds information,
        # i.e. pixels that more than one object deposited in
        ambiguous = counts > 1
        for k in range(len(objects)):
            pixels, values = per_obj_pixels[k], per_obj_q[k]
            if pixels.size == 0:
                continue
            keep = ambiguous[pixels]
            if keep.any():
                contrib_view.append(np.full(int(keep.sum()), v, dtype=np.uint8))
                contrib_pixel.append(pixels[keep].astype(np.int32))
                contrib_obj.append(np.full(int(keep.sum()), objects[k].obj_id, dtype=np.int16))
                contrib_q.append(values[keep].astype(np.float32))

    contrib = {
        "view": np.concatenate(contrib_view) if contrib_view else np.empty(0, np.uint8),
        "pixel": np.concatenate(contrib_pixel) if contrib_pixel else np.empty(0, np.int32),
        "obj_id": np.concatenate(contrib_obj) if contrib_obj else np.empty(0, np.int16),
        "charge": np.concatenate(contrib_q) if contrib_q else np.empty(0, np.float32),
    }

    records = [
        ObjectRecord(obj_id=o.obj_id, cls=o.cls, parent_id=o.parent_id,
                     params=param_vector(o.primitive), total_charge=o.total_charge,
                     n_pixels_dominant=n_dominant[k], n_pixels_touched=n_touched[k])
        for k, o in enumerate(objects)
    ]

    from .digitize import digitise
    adc = digitise(charge, readout)

    return RenderedEvent(charge=charge, adc=adc, semantic=semantic, instance=instance,
                         n_contrib=n_contrib, contrib=contrib, objects=records,
                         readout=readout, meta=dict(event.meta))
