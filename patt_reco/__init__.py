"""patt_reco -- synthetic images of lines and shower-like shapes.

A small, self-contained generator for binary pattern recognition: every image
holds exactly one object, either a track (a straight or curved line) or a
shower (a branching spray of scattered points). The two classes differ by
shape alone -- there is no detector, no noise, and no intensity pattern to
learn instead.

    from patt_reco import SourceConfig, generate_event, plot_image

    image, label = generate_event(SourceConfig(kind="track"), index=0)
    plot_image(image, label)
"""
from .config import (CLASS_NAMES, N_CLASSES, RenderConfig, SHOWER, ShowerConfig,
                     SourceConfig, TRACK, TrackConfig, load_yaml)
from .dataset.build import build_dataset
from .dataset.generate import generate_event
from .viz.display import plot_image

__version__ = "0.0.1"

__all__ = [
    "CLASS_NAMES", "N_CLASSES", "TRACK", "SHOWER",
    "RenderConfig", "ShowerConfig", "SourceConfig", "TrackConfig",
    "load_yaml", "build_dataset", "generate_event", "plot_image",
]
