"""1-D field + electronics response, convolved along the drift (tick) axis.

Two families:

* **collection-like (unipolar)** -- a charge-preserving positive pulse;
* **induction-like (bipolar)** -- a derivative-shaped pulse with a negative
  lobe, so nearby deposits partially cancel.

The bipolar case is what makes the task genuinely hard: the digitised image has
signal where no charge was deposited, and two close tracks can erase each other.
Truth labels stay defined on the *deposited charge* (see `projection.py`), so a
model has to undo this transformation rather than memorise one detector's shape.
"""
from __future__ import annotations

import numpy as np


def response_kernel(width_ticks: float, bipolar: bool, asym: float = 0.7,
                    n_sigma: float = 4.0) -> np.ndarray:
    """Build a normalised 1-D response kernel.

    Unipolar kernels sum to 1 (charge preserving). Bipolar kernels are
    normalised so the *positive lobe* sums to 1, which keeps peak ADC amplitudes
    comparable between the two families.
    """
    width = max(float(width_ticks), 0.3)
    half = max(2, int(np.ceil(n_sigma * width)))
    t = np.arange(-half, half + 1, dtype=float)
    gauss = np.exp(-0.5 * (t / width) ** 2)

    if not bipolar:
        return gauss / gauss.sum()

    kernel = -(t / width) * gauss           # derivative of a Gaussian
    kernel[t > 0] *= asym                   # asymmetric lobes
    positive = kernel[kernel > 0].sum()
    return kernel / max(positive, 1e-12)


def apply_response(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolve every channel (row) of a [channels, ticks] image along ticks."""
    pad = len(kernel) // 2
    padded = np.pad(image, ((0, 0), (pad, pad)), mode="constant")
    # sliding-window dot product: exact 'same' convolution, vectorised
    windows = np.lib.stride_tricks.sliding_window_view(padded, len(kernel), axis=1)
    return windows @ kernel[::-1]
