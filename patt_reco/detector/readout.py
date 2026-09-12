"""Per-event readout geometry and transport parameters.

Every quantity here is *sampled per event* inside the ranges given by
`DetectorConfig`. That is deliberate: a model trained at one pitch, one drift
velocity and one response shape has learned a detector, not a pattern.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import DetectorConfig
from ..geometry.volume import Volume
from .response import response_kernel


@dataclass(frozen=True)
class PlaneView:
    """One wire-plane-like projection.

        channel = (y cos(theta) + z sin(theta)) / pitch + n_channels / 2
        tick    = x / (v_drift * dt)
    """

    angle_deg: float
    pitch_cm: float
    tick_cm: float
    n_channels: int
    n_ticks: int

    @property
    def span_factor(self) -> float:
        """|cos| + |sin|: how much transverse reach this view needs."""
        a = np.deg2rad(self.angle_deg)
        return float(abs(np.cos(a)) + abs(np.sin(a)))

    def project(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """[N,3] cm -> (channel, tick) in continuous pixel units."""
        a = np.deg2rad(self.angle_deg)
        w = points[:, 1] * np.cos(a) + points[:, 2] * np.sin(a)
        channel = w / self.pitch_cm + 0.5 * self.n_channels
        tick = points[:, 0] / self.tick_cm
        return channel, tick


@dataclass
class Readout:
    """The complete sampled detector state for one event."""

    views: list[PlaneView]
    diffusion_t: float          # sigma_T [cm] = this * sqrt(drift [cm])
    diffusion_l: float
    attenuation_cm: float
    kernel: np.ndarray
    bipolar: bool
    gain: float                 # ADC per ke-
    channel_gain: np.ndarray    # [n_views, n_channels] relative
    dead_mask: np.ndarray       # [n_views, n_channels] bool, True = dead
    pedestal: float
    saturation: float
    quantise: bool

    @property
    def shape(self) -> tuple[int, int]:
        return (self.views[0].n_channels, self.views[0].n_ticks)

    @property
    def volume(self) -> Volume:
        """Fiducial box guaranteed to be visible in *every* view.

        The transverse half-size is set by the most demanding view: a 60-degree
        stereo plane reaches (|cos| + |sin|) = 1.37 times further than a
        vertical one for the same box.
        """
        v0 = self.views[0]
        worst = max(view.span_factor for view in self.views)
        half = 0.5 * v0.n_channels * v0.pitch_cm / worst
        drift = v0.n_ticks * v0.tick_cm
        return Volume(0.0, drift, -half, half, -half, half)


def sample_readout(rng: np.random.Generator, cfg: DetectorConfig) -> Readout:
    pitch = float(rng.uniform(*cfg.pitch_cm))
    tick = float(rng.uniform(*cfg.tick_cm))
    views = [PlaneView(angle_deg=float(a), pitch_cm=pitch, tick_cm=tick,
                       n_channels=cfg.n_channels, n_ticks=cfg.n_ticks)
             for a in cfg.view_angles_deg]

    bipolar = bool(rng.random() < cfg.p_bipolar)
    kernel = response_kernel(
        width_ticks=float(rng.uniform(*cfg.response_width_ticks)),
        bipolar=bipolar,
        asym=float(rng.uniform(*cfg.bipolar_asym)),
    )

    n_v, n_c = len(views), cfg.n_channels
    channel_gain = rng.normal(1.0, cfg.gain_spread, (n_v, n_c)).clip(0.1, None)
    dead_frac = float(rng.uniform(*cfg.frac_dead_channels))
    dead_mask = rng.random((n_v, n_c)) < dead_frac

    return Readout(
        views=views,
        diffusion_t=float(rng.uniform(*cfg.diffusion_t_cm_sqrt)),
        diffusion_l=float(rng.uniform(*cfg.diffusion_l_cm_sqrt)),
        attenuation_cm=float(rng.uniform(*cfg.attenuation_cm)),
        kernel=kernel,
        bipolar=bipolar,
        gain=float(rng.uniform(*cfg.gain_adc_per_ke)),
        channel_gain=channel_gain,
        dead_mask=dead_mask,
        pedestal=cfg.pedestal_adc,
        saturation=cfg.adc_saturation,
        quantise=cfg.quantise,
    )
