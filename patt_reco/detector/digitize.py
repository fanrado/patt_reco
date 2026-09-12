"""Charge image -> ADC image: response convolution, gain, defects, quantisation."""
from __future__ import annotations

import numpy as np

from .readout import Readout
from .response import apply_response


def digitise(charge: np.ndarray, readout: Readout) -> np.ndarray:
    """[V, C, T] charge in ke- -> [V, C, T] ADC (noise-free)."""
    out = np.empty_like(charge, dtype=np.float32)
    for v in range(charge.shape[0]):
        shaped = apply_response(charge[v].astype(np.float64), readout.kernel)
        shaped = shaped * readout.gain * readout.channel_gain[v][:, None]
        shaped[readout.dead_mask[v]] = 0.0
        out[v] = shaped.astype(np.float32)

    out += readout.pedestal
    np.clip(out, -readout.saturation, readout.saturation, out=out)
    if readout.quantise:
        np.round(out, out=out)
    return out
