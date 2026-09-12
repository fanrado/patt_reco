"""Noise realisation.

Noise is *not* baked into stored datasets (except frozen test sets): it is
realised at load time from the event's stored noise seed, so every epoch sees a
fresh realisation of the same physical noise model. That makes the dataset small
and gives a physically correct augmentation for free.
"""
from __future__ import annotations

import numpy as np

from ..config import NoiseConfig
from .coherent import coherent
from .incoherent import blips, pink, white

__all__ = ["realise_noise", "coherent", "white", "pink", "blips"]


def realise_noise(rng: np.random.Generator, shape: tuple[int, int, int],
                  cfg: NoiseConfig, dead_mask: np.ndarray | None = None,
                  scale: float = 1.0) -> np.ndarray:
    """Return a [V, C, T] noise image in ADC units.

    `scale` multiplies every amplitude at once, which is how the SNR sweep in
    the busy-event validation is driven without touching the config.
    """
    n_views, n_ch, n_tk = shape
    out = np.zeros(shape, dtype=np.float32)
    if not cfg.enabled or scale <= 0:
        return out

    for v in range(n_views):
        plane = (n_ch, n_tk)
        out[v] += white(rng, plane, scale * rng.uniform(*cfg.incoherent_adc))
        out[v] += coherent(rng, plane, scale * rng.uniform(*cfg.coherent_adc),
                           cfg.coherent_group)
        out[v] += pink(rng, plane, scale * rng.uniform(*cfg.pink_adc))
        if rng.random() < cfg.p_blip:
            n = int(rng.integers(cfg.n_blips[0], cfg.n_blips[1] + 1))
            out[v] += blips(rng, plane, n, tuple(scale * a for a in cfg.blip_adc))

    if dead_mask is not None:
        out[dead_mask] = 0.0
    return out
