import numpy as np
import pytest

from patt_reco.config import DatasetConfig, DetectorConfig, EventConfig


@pytest.fixture
def small_cfg():
    """A cheap but non-trivial config: a handful of objects on a small readout."""
    return DatasetConfig(
        name="test", n_events=8, seed=1234,
        event=EventConfig(n_objects=(2, 5), n_cosmics=(0, 1)),
        detector=DetectorConfig(n_channels=64, n_ticks=64),
    )


@pytest.fixture
def rng():
    return np.random.default_rng(0)
