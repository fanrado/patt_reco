"""patt_reco -- an experiment-independent pattern-recognition benchmark.

Synthetic 3D events (tracks, scattered tracks, helices, rings, showers, blobs)
rendered into 2D stereo readout views with realistic charge transport, detector
response and noise, together with exact per-pixel semantic and instance truth.

    from patt_reco import DatasetConfig, generate_event, plot_event

    rec = generate_event(DatasetConfig(), index=0)
    plot_event(rec)
"""
from .config import (CLASS_NAMES, DatasetConfig, DetectorConfig, EventConfig,
                     GeometryConfig, NoiseConfig, N_CLASSES, load_yaml)
from .dataset.generate import generate_dataset, generate_event, generate_from_yaml
from .dataset.io_hdf5 import DatasetReader, ShardReader
from .dataset.schema import EventRecord
from .viz.event_display import plot_event

__version__ = "0.0.1"

__all__ = [
    "CLASS_NAMES", "N_CLASSES", "DatasetConfig", "DetectorConfig", "EventConfig",
    "GeometryConfig", "NoiseConfig", "load_yaml", "generate_dataset",
    "generate_event", "generate_from_yaml", "DatasetReader", "ShardReader",
    "EventRecord", "plot_event",
]
