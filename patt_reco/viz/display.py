"""Greyscale display of generated images.

Nearly every generator bug is obvious in an image and invisible in a loss
curve, so this module exists from day one rather than being bolted on.

Intensities are shown on a fixed 0..1 scale rather than autoscaled per panel:
`project` already peak-normalises each image, and letting matplotlib rescale
each axes independently would hide real differences in how much of an image is
actually lit.
"""
from __future__ import annotations

import numpy as np

from ..config import CLASS_NAMES


def plot_image(image, label=None, ax=None):
    """Show one [H, W] image in greyscale. Returns the Axes."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(3.0, 3.0))

    image = np.asarray(image, dtype=float)
    ax.imshow(image, origin="lower", interpolation="nearest",
              cmap="gray", vmin=0.0, vmax=1.0)

    if label is not None:
        ax.set_title(CLASS_NAMES.get(int(label), str(label)), fontsize=10)

    # pixel indices carry no meaning here; ticks would only add clutter
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return ax


def plot_grid(images, labels=None, ncols: int = 8, scale: float = 1.6):
    """Show a batch of images as a grid of greyscale panels. Returns the Figure."""
    import matplotlib.pyplot as plt

    images = list(images)
    n = len(images)
    ncols = max(1, min(ncols, n)) if n else 1
    nrows = max(1, -(-n // ncols))

    fig, axes = plt.subplots(nrows, ncols, squeeze=False,
                             figsize=(scale * ncols, scale * nrows))
    for i, ax in enumerate(axes.ravel()):
        if i < n:
            plot_image(images[i], None if labels is None else labels[i], ax=ax)
        else:
            ax.set_axis_off()
    fig.tight_layout()
    return fig
