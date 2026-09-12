"""Coherent noise: one waveform shared by a group of neighbouring channels.

This is the component that actually hurts. It is not removable by per-pixel
thresholding, it looks like a real horizontal feature, and every detector with
shared electronics has it.
"""
from __future__ import annotations

import numpy as np


def coherent(rng: np.random.Generator, shape: tuple[int, int], sigma: float,
             group: int) -> np.ndarray:
    if sigma <= 0 or group <= 0:
        return np.zeros(shape, dtype=np.float32)
    n_ch, n_tk = shape
    n_groups = int(np.ceil(n_ch / group))
    waveforms = rng.normal(0.0, sigma, (n_groups, n_tk)).astype(np.float32)
    # per-channel coupling to its group's waveform
    coupling = rng.normal(1.0, 0.15, n_ch).astype(np.float32)
    expanded = np.repeat(waveforms, group, axis=0)[:n_ch]
    return expanded * coupling[:, None]
