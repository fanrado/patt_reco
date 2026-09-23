"""Shared fixtures for the track/shower generator tests.

Everything here is deliberately tiny: a 16x16 frame and a handful of events.
The generator is exercised for its *contracts*, not for image quality, so
paying for 128x128 renders in a unit test buys nothing.
"""
import numpy as np
import pytest

from patt_reco.config import (RenderConfig, SHOWER, ShowerConfig, SourceConfig,
                              TRACK, TrackConfig)
from patt_reco.geometry.volume import Volume


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def vol():
    return Volume.cube()


@pytest.fixture
def render_cfg():
    return RenderConfig(height=16, width=16, margin=0.1, view=(0.0, 0.0, 1.0))


@pytest.fixture
def track_cfg():
    """Coarse `step`, so a deposit is tens of points rather than hundreds."""
    return TrackConfig(length=(0.5, 1.0), curvature=(0.0, 2.0), step=0.05, value=1.0)


@pytest.fixture
def shower_cfg():
    return ShowerConfig(max_nodes=7, seg_len=(0.1, 0.2), open_angle=(0.2, 0.5),
                        split_frac=2.0, spread=(0.01, 0.03), value=(0.5, 1.5),
                        step=0.02)


@pytest.fixture
def straight_shower_cfg():
    """A shower collapsed to a single straight ray.

    Zero opening angle and zero spread remove every source of randomness in
    the branch *geometry*, and a huge Beta shape parameter pins every split to
    an even one. What survives is the part worth pinning down: how far the
    cascade reaches, which is exactly what the breadth-first, non-compounding
    length rule decides.
    """
    return ShowerConfig(max_nodes=7, seg_len=(1.0, 1.0), open_angle=(0.0, 0.0),
                        split_frac=1e6, spread=(0.0, 0.0), value=(1.0, 1.0),
                        step=0.5)


@pytest.fixture
def track_source(render_cfg, track_cfg):
    return SourceConfig(name="tracks", kind="track", label=TRACK, seed=11,
                        n_train=6, n_val=4, n_test=2,
                        track=track_cfg, render=render_cfg)


@pytest.fixture
def shower_source(render_cfg, shower_cfg):
    return SourceConfig(name="showers", kind="shower", label=SHOWER, seed=22,
                        n_train=6, n_val=4, n_test=2,
                        shower=shower_cfg, render=render_cfg)
