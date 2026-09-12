"""The on-disk event record, and how it inflates back into images.

Storage strategy (the reason the datasets stay small):

* only the **sparse deposited charge** is stored, not the dense ADC image;
* `semantic` is *derived* from `instance` via the object table rather than
  stored, which makes the two labels consistent by construction;
* the ADC image is *recomputed* at load time from the stored readout state.
  That costs ~1 ms, and it buys the domain-shift test for free: re-render the
  same event through a different response kernel and nothing else changes.
* noise is realised from a stored seed, not baked in, so every epoch sees a
  fresh realisation of the same physical noise model.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import EMPTY, NoiseConfig
from ..detector.digitize import digitise
from ..detector.projection import RenderedEvent
from ..detector.readout import PlaneView, Readout
from ..noise import realise_noise

READOUT_SCALARS = ("pitch_cm", "tick_cm", "diffusion_t", "diffusion_l",
                   "attenuation_cm", "gain", "pedestal", "saturation")


@dataclass
class EventRecord:
    """One event, in the form it is written to and read from disk."""

    index: int
    seed: int
    shape: tuple[int, int, int]            # (views, channels, ticks)

    pixel: np.ndarray                      # int32 [K], flat index into [V, C, T]
    charge: np.ndarray                     # float32 [K], ke-
    instance: np.ndarray                   # int16 [K], dominant contributor
    n_contrib: np.ndarray                  # uint8 [K]

    contrib_pixel: np.ndarray              # ambiguous pixels only
    contrib_obj: np.ndarray
    contrib_charge: np.ndarray

    obj_id: np.ndarray                     # int16 [n_obj]
    obj_cls: np.ndarray                    # uint8 [n_obj]
    obj_parent: np.ndarray                 # int16 [n_obj]
    obj_params: np.ndarray                 # float32 [n_obj, N_PARAMS]
    obj_charge: np.ndarray                 # float32 [n_obj]
    obj_ndom: np.ndarray                   # int32 [n_obj, V]
    obj_ntouch: np.ndarray                 # int32 [n_obj, V]

    kernel: np.ndarray                     # float32 [L]
    channel_gain: np.ndarray               # float32 [V, C]
    dead_mask: np.ndarray                  # bool [V, C]
    view_angles: np.ndarray                # float32 [V]
    scalars: dict                          # READOUT_SCALARS + flags
    meta: dict = field(default_factory=dict)

    # -- construction ------------------------------------------------------
    @classmethod
    def from_rendered(cls, rendered: RenderedEvent, index: int, seed: int) -> "EventRecord":
        flat_charge = rendered.charge.reshape(-1)
        hit = np.flatnonzero(flat_charge)
        readout = rendered.readout
        v0 = readout.views[0]

        return cls(
            index=index,
            seed=seed,
            shape=rendered.charge.shape,
            pixel=hit.astype(np.int32),
            charge=flat_charge[hit].astype(np.float32),
            instance=rendered.instance.reshape(-1)[hit].astype(np.int16),
            n_contrib=rendered.n_contrib.reshape(-1)[hit].astype(np.uint8),
            contrib_pixel=_absolute_contrib_pixel(rendered),
            contrib_obj=rendered.contrib["obj_id"],
            contrib_charge=rendered.contrib["charge"],
            obj_id=np.array([o.obj_id for o in rendered.objects], dtype=np.int16),
            obj_cls=np.array([o.cls for o in rendered.objects], dtype=np.uint8),
            obj_parent=np.array([o.parent_id for o in rendered.objects], dtype=np.int16),
            obj_params=np.stack([o.params for o in rendered.objects]).astype(np.float32),
            obj_charge=np.array([o.total_charge for o in rendered.objects], dtype=np.float32),
            obj_ndom=np.stack([o.n_pixels_dominant for o in rendered.objects]).astype(np.int32),
            obj_ntouch=np.stack([o.n_pixels_touched for o in rendered.objects]).astype(np.int32),
            kernel=readout.kernel.astype(np.float32),
            channel_gain=readout.channel_gain.astype(np.float32),
            dead_mask=readout.dead_mask,
            view_angles=np.array([v.angle_deg for v in readout.views], dtype=np.float32),
            scalars={
                "pitch_cm": v0.pitch_cm, "tick_cm": v0.tick_cm,
                "diffusion_t": readout.diffusion_t, "diffusion_l": readout.diffusion_l,
                "attenuation_cm": readout.attenuation_cm, "gain": readout.gain,
                "pedestal": readout.pedestal, "saturation": readout.saturation,
                "bipolar": float(readout.bipolar), "quantise": float(readout.quantise),
            },
            meta=dict(rendered.meta, occupancy=rendered.occupancy),
        )

    # -- inflation ---------------------------------------------------------
    @property
    def n_objects(self) -> int:
        return len(self.obj_id)

    def build_readout(self, kernel: np.ndarray | None = None) -> Readout:
        """Rebuild the sampled detector state. Pass `kernel` to re-render this
        event through a *different* response -- that is the domain-shift test."""
        n_v, n_ch, n_tk = self.shape
        s = self.scalars
        views = [PlaneView(angle_deg=float(a), pitch_cm=s["pitch_cm"], tick_cm=s["tick_cm"],
                           n_channels=n_ch, n_ticks=n_tk) for a in self.view_angles]
        return Readout(
            views=views, diffusion_t=s["diffusion_t"], diffusion_l=s["diffusion_l"],
            attenuation_cm=s["attenuation_cm"],
            kernel=self.kernel if kernel is None else np.asarray(kernel, dtype=np.float32),
            bipolar=bool(s["bipolar"]), gain=s["gain"], channel_gain=self.channel_gain,
            dead_mask=self.dead_mask, pedestal=s["pedestal"], saturation=s["saturation"],
            quantise=bool(s["quantise"]),
        )

    def dense_charge(self) -> np.ndarray:
        out = np.zeros(int(np.prod(self.shape)), dtype=np.float32)
        out[self.pixel] = self.charge
        return out.reshape(self.shape)

    def dense_instance(self) -> np.ndarray:
        out = np.full(int(np.prod(self.shape)), -1, dtype=np.int16)
        out[self.pixel] = self.instance
        return out.reshape(self.shape)

    def dense_semantic(self) -> np.ndarray:
        """Derived from `instance`, so the two can never disagree."""
        lookup = np.zeros(int(self.obj_id.max()) + 2, dtype=np.uint8) if self.n_objects else \
            np.zeros(1, dtype=np.uint8)
        lookup[self.obj_id] = self.obj_cls
        out = np.full(int(np.prod(self.shape)), EMPTY, dtype=np.uint8)
        out[self.pixel] = lookup[self.instance]
        return out.reshape(self.shape)

    def dense_n_contrib(self) -> np.ndarray:
        out = np.zeros(int(np.prod(self.shape)), dtype=np.uint8)
        out[self.pixel] = self.n_contrib
        return out.reshape(self.shape)

    def adc(self, noise_cfg: NoiseConfig | None = None, noise_seed: int | None = None,
            noise_scale: float = 1.0, kernel: np.ndarray | None = None) -> np.ndarray:
        """Digitised image, with a fresh noise realisation if asked for."""
        readout = self.build_readout(kernel=kernel)
        image = digitise(self.dense_charge(), readout)
        if noise_cfg is not None and noise_cfg.enabled and noise_scale > 0:
            rng = np.random.default_rng([self.seed, self.index, 1]
                                        if noise_seed is None else noise_seed)
            image = image + realise_noise(rng, self.shape, noise_cfg,
                                          dead_mask=readout.dead_mask, scale=noise_scale)
            np.clip(image, -readout.saturation, readout.saturation, out=image)
        return image.astype(np.float32)


def _absolute_contrib_pixel(rendered: RenderedEvent) -> np.ndarray:
    """Contribution pixels are recorded per view; store them as flat [V,C,T] indices."""
    n_v, n_ch, n_tk = rendered.charge.shape
    if rendered.contrib["pixel"].size == 0:
        return np.empty(0, dtype=np.int32)
    return (rendered.contrib["view"].astype(np.int64) * (n_ch * n_tk)
            + rendered.contrib["pixel"].astype(np.int64)).astype(np.int32)
