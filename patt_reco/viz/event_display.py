"""Three-view event displays.

Nearly every generator bug is obvious in an event display and invisible in a
loss curve, so this module exists from day one rather than being bolted on.
"""
from __future__ import annotations

import numpy as np

from ..config import CLASS_NAMES, N_CLASSES, NoiseConfig

_CLASS_COLOURS = [
    "#ffffff",  # empty
    "#1f77b4",  # track
    "#2ca02c",  # scattered
    "#9467bd",  # helix
    "#ff7f0e",  # ring
    "#d62728",  # shower
    "#8c564b",  # blob
    "#7f7f7f",  # cosmic
]


def _class_cmap():
    from matplotlib.colors import ListedColormap
    return ListedColormap(_CLASS_COLOURS[:N_CLASSES])


def _instance_image(instance: np.ndarray, rng_seed: int = 0) -> np.ndarray:
    """Map instance ids to distinguishable colours; empty pixels stay white."""
    import matplotlib.pyplot as plt
    palette = plt.get_cmap("tab20")(np.linspace(0, 1, 20))[:, :3]
    rng = np.random.default_rng(rng_seed)
    order = rng.permutation(20)
    out = np.ones(instance.shape + (3,), dtype=float)
    filled = instance >= 0
    out[filled] = palette[order[instance[filled] % 20]]
    return out


def plot_event(rec, noise_cfg: NoiseConfig | None = None, noise_scale: float = 1.0,
               panels: tuple[str, ...] = ("adc", "semantic", "instance"),
               figsize_scale: float = 3.2, adc_percentile: float = 99.5,
               title: str | None = None):
    """Render one `EventRecord` as a (panels x views) grid of images."""
    import matplotlib.pyplot as plt

    adc = rec.adc(noise_cfg=noise_cfg, noise_scale=noise_scale)
    semantic = rec.dense_semantic()
    instance = rec.dense_instance()
    charge = rec.dense_charge()
    n_views = rec.shape[0]

    images = {"adc": adc, "semantic": semantic, "instance": instance, "charge": charge,
              "n_contrib": rec.dense_n_contrib()}

    fig, axes = plt.subplots(len(panels), n_views, squeeze=False,
                             figsize=(figsize_scale * n_views, figsize_scale * len(panels)))

    for row, panel in enumerate(panels):
        data = images[panel]
        for v in range(n_views):
            ax = axes[row][v]
            plane = data[v].T                      # ticks on y, channels on x
            if panel == "instance":
                ax.imshow(_instance_image(plane), origin="lower", interpolation="nearest",
                          aspect="auto")
            elif panel == "semantic":
                ax.imshow(plane, origin="lower", interpolation="nearest", aspect="auto",
                          cmap=_class_cmap(), vmin=0, vmax=N_CLASSES - 1)
            elif panel == "adc":
                # percentile, not max: one Bragg peak or shower core is often
                # 10x brighter than everything else and would wash the rest out
                limit = max(1.0, float(np.percentile(np.abs(plane), adc_percentile)))
                ax.imshow(plane, origin="lower", interpolation="nearest", aspect="auto",
                          cmap="RdBu_r", vmin=-limit, vmax=limit)
            else:
                ax.imshow(np.ma.masked_less_equal(plane, 0), origin="lower",
                          interpolation="nearest", aspect="auto", cmap="viridis")
            if row == 0:
                ax.set_title(f"view {v}  ({rec.view_angles[v]:+.0f}$\\degree$)", fontsize=9)
            if v == 0:
                ax.set_ylabel(f"{panel}\ntick", fontsize=9)
            if row == len(panels) - 1:
                ax.set_xlabel("channel", fontsize=9)
            ax.tick_params(labelsize=7)

    if title is None:
        bipolar = "bipolar" if rec.scalars["bipolar"] else "unipolar"
        title = (f"event {rec.index} (seed {rec.seed}) | {rec.n_objects} objects | "
                 f"occupancy {100 * (charge > 0).mean():.1f}% | {bipolar} response | "
                 f"pitch {rec.scalars['pitch_cm']:.2f} cm")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig


def class_legend_handles():
    """Patches for a semantic-panel legend."""
    from matplotlib.patches import Patch
    return [Patch(facecolor=_CLASS_COLOURS[c], label=CLASS_NAMES[c])
            for c in range(1, N_CLASSES)]
