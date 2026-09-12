"""Per-channel noise: white, 1/f, and hit-like artefacts.

These deliberately do *not* call `signal_processing_toolkit.utils.signal_gen`:
those helpers draw from numpy's global RNG, which would break the guarantee that
`(dataset_seed, event_index)` reproduces an event bit-for-bit regardless of
worker count. Everything here takes an explicit `Generator`.
"""
from __future__ import annotations

import numpy as np


def white(rng: np.random.Generator, shape: tuple[int, ...], sigma: float) -> np.ndarray:
    if sigma <= 0:
        return np.zeros(shape, dtype=np.float32)
    return rng.normal(0.0, sigma, shape).astype(np.float32)


def pink(rng: np.random.Generator, shape: tuple[int, int], sigma: float) -> np.ndarray:
    """1/f noise along the tick axis, independent per channel."""
    if sigma <= 0:
        return np.zeros(shape, dtype=np.float32)
    n_ch, n_tk = shape
    spectrum = rng.normal(0.0, 1.0, (n_ch, n_tk // 2 + 1)) \
        + 1j * rng.normal(0.0, 1.0, (n_ch, n_tk // 2 + 1))
    freq = np.arange(spectrum.shape[1], dtype=float)
    freq[0] = 1.0
    spectrum /= np.sqrt(freq)
    spectrum[:, 0] = 0.0                      # no DC offset
    series = np.fft.irfft(spectrum, n=n_tk, axis=1)
    scale = series.std()
    if scale > 0:
        series *= sigma / scale
    return series.astype(np.float32)


def blips(rng: np.random.Generator, shape: tuple[int, int], n_blips: int,
          amplitude: tuple[float, float]) -> np.ndarray:
    """Small hit-like artefacts: the kind of thing that fools a naive threshold."""
    out = np.zeros(shape, dtype=np.float32)
    if n_blips <= 0:
        return out
    n_ch, n_tk = shape
    chan = rng.integers(0, n_ch, n_blips)
    tick = rng.integers(0, n_tk, n_blips)
    amp = rng.uniform(*amplitude, n_blips)
    width = rng.uniform(0.8, 2.5, n_blips)
    for c, t, a, w in zip(chan, tick, amp, width):
        t0, t1 = max(0, int(t - 3 * w)), min(n_tk, int(t + 3 * w) + 1)
        if t1 <= t0:
            continue
        span = np.arange(t0, t1)
        out[c, t0:t1] += a * np.exp(-0.5 * ((span - t) / w) ** 2)
    return out
