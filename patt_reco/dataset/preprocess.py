"""Input normalisation, shared by the models and the classical baselines.

Gain is randomised per event, so raw ADC is not comparable between events. Every
consumer therefore divides by the *noise* level estimated from the image itself.
That is what a real experiment does, and it makes the network's input unit equal
to one standard deviation of noise -- so a threshold of "3" means 3 sigma
whatever the gain, and SNR sweeps mean the same thing across the dataset.

The estimate is a median absolute deviation, which ignores the few percent of
pixels carrying signal. A noise-free image (L0, L1) has MAD = 0, hence the floor.
"""
from __future__ import annotations

import numpy as np

MAD_TO_SIGMA = 1.4826


def noise_sigma(adc: np.ndarray, floor: float = 1.0) -> np.ndarray:
    """Robust per-view noise estimate, shape [..., 1, 1] for broadcasting."""
    flat = adc.reshape(adc.shape[0], -1) if adc.ndim == 3 else adc.reshape(1, -1)
    median = np.median(flat, axis=1, keepdims=True)
    mad = np.median(np.abs(flat - median), axis=1, keepdims=True)
    sigma = np.maximum(MAD_TO_SIGMA * mad, floor)
    return sigma.reshape((-1,) + (1,) * (adc.ndim - 1)) if adc.ndim == 3 else sigma.reshape(1, 1)


def normalise(adc: np.ndarray, floor: float = 1.0, clip: float | None = 50.0) -> np.ndarray:
    """ADC -> units of noise sigma, median-subtracted per view."""
    adc = np.asarray(adc, dtype=np.float32)
    flat = adc.reshape(adc.shape[0], -1) if adc.ndim == 3 else adc.reshape(1, -1)
    median = np.median(flat, axis=1).reshape((-1,) + (1,) * (adc.ndim - 1))
    out = (adc - median) / noise_sigma(adc, floor)
    if clip is not None:
        np.clip(out, -clip, clip, out=out)
    return out.astype(np.float32)
