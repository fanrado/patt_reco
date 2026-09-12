"""Augmentation.

Three kinds, in descending order of value:

1. **Fresh noise realisation** -- free, physically exact, and the reason noise is
   stored as a seed rather than as bytes.
2. **Event mixing** -- overlay two clean events and add noise once. This is the
   mechanism that lets a model trained on 2-5 objects cope with 40, *without*
   training on the busy set that it will be validated against.
3. **Geometric** -- channel flips and small tick shifts only. Rotating a rendered
   view is physically wrong: the response kernel acts along ticks and is not
   isotropic, so a rotated image is not an image this detector could produce.
   Real geometric augmentation belongs at the 3D stage, i.e. in the generator.
"""
from __future__ import annotations

import numpy as np

from ..config import EMPTY, NoiseConfig
from ..noise import realise_noise


class MixedEvent:
    """The overlay of two events, exposing the same surface as an EventRecord."""

    def __init__(self, adc: np.ndarray, semantic: np.ndarray, instance: np.ndarray,
                 n_contrib: np.ndarray, n_objects: int):
        self._adc, self._semantic = adc, semantic
        self._instance, self._n_contrib = instance, n_contrib
        self.n_objects = n_objects
        self.shape = adc.shape

    def adc(self, *args, **kwargs) -> np.ndarray:
        return self._adc

    def dense_semantic(self) -> np.ndarray:
        return self._semantic

    def dense_instance(self) -> np.ndarray:
        return self._instance

    def dense_n_contrib(self) -> np.ndarray:
        return self._n_contrib


def mix_events(rec_a, rec_b, noise_cfg: NoiseConfig | None, rng: np.random.Generator,
               noise_scale: float = 1.0) -> MixedEvent:
    """Overlay two events: charges add, the larger contributor wins each pixel.

    Both events are digitised *without* noise and summed, then one noise
    realisation is added -- overlaying two noisy images would give a busy event
    with sqrt(2) times the noise, which no detector produces.
    """
    charge_a, charge_b = rec_a.dense_charge(), rec_b.dense_charge()
    adc = rec_a.adc(noise_cfg=None) + rec_b.adc(noise_cfg=None)

    semantic_a, semantic_b = rec_a.dense_semantic(), rec_b.dense_semantic()
    instance_a, instance_b = rec_a.dense_instance(), rec_b.dense_instance()

    offset = int(instance_a.max()) + 1 if (instance_a >= 0).any() else 0
    b_present = instance_b >= 0
    take_b = b_present & ((charge_b > charge_a) | (instance_a < 0))

    semantic = np.where(take_b, semantic_b, semantic_a).astype(np.uint8)
    instance = np.where(take_b, instance_b.astype(np.int32) + offset,
                        instance_a).astype(np.int16)
    # the two object sets are disjoint, so contributor counts simply add
    n_contrib = np.clip(rec_a.dense_n_contrib().astype(np.int32)
                        + rec_b.dense_n_contrib().astype(np.int32), 0, 255).astype(np.uint8)

    if noise_cfg is not None and noise_cfg.enabled and noise_scale > 0:
        readout = rec_a.build_readout()
        adc = adc + realise_noise(rng, adc.shape, noise_cfg,
                                  dead_mask=readout.dead_mask, scale=noise_scale)
        np.clip(adc, -readout.saturation, readout.saturation, out=adc)

    n_objects = rec_a.n_objects + rec_b.n_objects
    return MixedEvent(adc.astype(np.float32), semantic, instance, n_contrib, n_objects)


def flip_channels(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Mirror along the channel axis. Legal: it is a relabelling of the wires."""
    return {k: np.ascontiguousarray(v[..., ::-1, :]) for k, v in arrays.items()}


def shift_ticks(arrays: dict[str, np.ndarray], shift: int) -> dict[str, np.ndarray]:
    """Roll along the drift axis. Legal: it is a different trigger time."""
    return {k: np.roll(v, shift, axis=-1) for k, v in arrays.items()}


def apply_geometric(arrays: dict[str, np.ndarray], rng: np.random.Generator,
                    p_flip: float = 0.5, max_shift: int = 8) -> dict[str, np.ndarray]:
    if rng.random() < p_flip:
        arrays = flip_channels(arrays)
    if max_shift > 0:
        shift = int(rng.integers(-max_shift, max_shift + 1))
        if shift:
            arrays = shift_ticks(arrays, shift)
    return arrays
