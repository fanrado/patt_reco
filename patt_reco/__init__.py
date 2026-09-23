"""patt_reco -- synthetic images of lines and shower-like shapes, and a
classifier for them.

A small, self-contained project in two halves. The generator makes images
holding exactly one object, either a track (a straight or curved line) or a
shower (a branching spray of scattered points); the two classes differ by
shape alone, with no detector, no noise and no intensity pattern to learn
instead. The training half fits a deliberately minimal CNN to them.

    from patt_reco import SourceConfig, generate_event, plot_image

    image, label = generate_event(SourceConfig(kind="track"), index=0)
    plot_image(image, label)

The generator needs only numpy, matplotlib and pyyaml. `CNN`, `Trainer`,
`TrainConfig` and `load_train_yaml` additionally need PyTorch, and are absent
from this namespace when it is not installed.
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

# The training half is optional: a generator-only install has no torch, and
# importing patt_reco must still work there.
_TORCH_ONLY = ("CNN", "TrainConfig", "Trainer", "load_train_yaml")

try:
    from .models import CNN
    from .train import TrainConfig, Trainer, load_train_yaml
except ImportError as exc:          # pragma: no cover -- depends on the install
    _TORCH_IMPORT_ERROR = exc

    def __getattr__(name):
        if name in _TORCH_ONLY:
            raise ImportError(
                f"patt_reco.{name} requires PyTorch, which is not installed. "
                f'Install it with: pip install "patt_reco[dl]"'
            ) from _TORCH_IMPORT_ERROR
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
else:
    __all__ += list(_TORCH_ONLY)
