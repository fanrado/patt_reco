"""Configuration dataclasses for the patt_reco generator.

Every knob lives here. Nothing in the generator has a hidden default: a config
object plus a seed fully determines an event (see `patt_reco.dataset.generate`).

Configs are plain frozen dataclasses so they hash cleanly; `load_yaml` builds a
nested config from a YAML file and `to_dict`/`config_hash` serialise it back for
the dataset manifest.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, fields, is_dataclass, asdict
from typing import Any

import yaml

# --------------------------------------------------------------------------- #
# semantic classes
# --------------------------------------------------------------------------- #

EMPTY = 0
TRACK = 1
SCATTERED = 2
HELIX = 3
RING = 4
SHOWER = 5
BLOB = 6
COSMIC = 7

N_CLASSES = 8

CLASS_NAMES = {
    EMPTY: "empty",
    TRACK: "track",
    SCATTERED: "scattered",
    HELIX: "helix",
    RING: "ring",
    SHOWER: "shower",
    BLOB: "blob",
    COSMIC: "cosmic",
}

# classes that can be sampled as primary event objects (cosmics are background,
# added separately by the composer)
PRIMARY_CLASSES = (TRACK, SCATTERED, HELIX, RING, SHOWER, BLOB)


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class GeometryConfig:
    """Physics knobs for the 3D primitives. Lengths in cm, charge in ke-."""

    step_cm: float = 0.05            # deposition step along a trajectory
    dqdx_mip: float = 55.0           # ke/cm for a minimum-ionising particle
    landau_scale: float = 0.15       # Moyal scale as a fraction of the MPV

    # --- straight track ---
    track_len_cm: tuple = (5.0, 60.0)
    p_stopping: float = 0.35         # probability the track ends in a Bragg peak
    bragg_ref_cm: float = 5.0        # residual range at which dE/dx = MIP
    bragg_exponent: float = 0.42     # dE/dx ~ R^-0.42
    bragg_max: float = 6.0           # cap, in units of MIP

    # --- multiple-scattered track ---
    scat_len_cm: tuple = (5.0, 60.0)
    scat_momentum_mev: tuple = (80.0, 2000.0)
    rad_len_cm: float = 14.0         # X0 of liquid argon

    # --- helix ---
    helix_radius_cm: tuple = (8.0, 120.0)
    helix_pitch_cm: tuple = (0.0, 15.0)   # advance along the axis per radian
    helix_arc_rad: tuple = (0.6, 4.5)

    # --- ring ---
    ring_radius_cm: tuple = (4.0, 20.0)
    ring_thickness_cm: tuple = (0.15, 0.8)
    ring_arc_frac: tuple = (0.6, 1.0)

    # --- shower ---
    shower_e0_mev: tuple = (50.0, 1000.0)
    shower_emin_mev: float = 15.0
    shower_seg_x0: tuple = (0.3, 0.9)      # segment length in units of X0
    shower_open_rad: tuple = (0.05, 0.30)  # half-opening angle at a split
    shower_split_frac: float = 2.0         # Beta(a,a) energy split; a = this
    shower_max_nodes: int = 400
    shower_ke_per_mev: float = 25.0        # charge yield

    # --- blob ---
    blob_sigma_cm: tuple = (0.4, 2.5)
    blob_charge_ke: tuple = (50.0, 800.0)
    blob_points: int = 400

    # --- delta ray ---
    delta_len_cm: tuple = (0.5, 4.0)


# --------------------------------------------------------------------------- #
# event composition
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class EventConfig:
    """How many objects, how they are arranged, and how hard we make it."""

    n_objects: tuple = (1, 1)              # inclusive range, uniform
    class_weights: dict = field(
        default_factory=lambda: {
            "track": 1.0, "scattered": 1.0, "helix": 1.0,
            "ring": 0.6, "shower": 0.8, "blob": 0.5,
        }
    )

    # shared-vertex groups (interaction-like topologies, incl. vees)
    p_vertex: float = 0.3
    vertex_size: tuple = (2, 4)

    # delta rays hanging off tracks
    p_delta: float = 0.15

    # deliberately forced 3D crossings -- random placement almost never makes
    # the small-angle crossings that are the actual hard case
    p_crossing: float = 0.3
    crossing_angle_deg: tuple = (5.0, 90.0)

    # cosmic-like background traversing the whole volume
    n_cosmics: tuple = (0, 0)

    # fraction of the readout span objects are allowed to start in
    fiducial_frac: float = 0.85


# --------------------------------------------------------------------------- #
# detector / readout
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class DetectorConfig:
    """3D -> 2D projection, charge transport and digitisation.

    Ranges are sampled *per event* (domain randomisation): a model that only
    works at one pitch or one response shape has not learned the pattern.
    """

    n_channels: int = 128
    n_ticks: int = 128
    view_angles_deg: tuple = (0.0, 60.0, -60.0)

    pitch_cm: tuple = (0.3, 0.5)           # cm per channel
    tick_cm: tuple = (0.3, 0.5)            # cm per tick (v_drift * dt)

    # charge transport
    diffusion_substeps: int = 4                 # sub-deposits per point, for smooth diffusion
    diffusion_t_cm_sqrt: tuple = (0.01, 0.05)   # sigma_T = k * sqrt(drift cm)
    diffusion_l_cm_sqrt: tuple = (0.01, 0.04)   # sigma_L = k * sqrt(drift cm)
    attenuation_cm: tuple = (200.0, 2000.0)     # charge ~ exp(-x / this)

    # electronics + field response, convolved along the tick axis
    response_width_ticks: tuple = (1.0, 3.0)
    p_bipolar: float = 0.5                  # induction-like (bipolar) vs collection
    bipolar_asym: tuple = (0.4, 0.9)        # negative-lobe amplitude fraction

    # digitisation
    gain_adc_per_ke: tuple = (1.5, 4.0)
    pedestal_adc: float = 0.0
    adc_saturation: float = 4095.0
    quantise: bool = True

    # defects
    frac_dead_channels: tuple = (0.0, 0.02)
    gain_spread: float = 0.05               # per-channel relative gain sigma


# --------------------------------------------------------------------------- #
# noise
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class NoiseConfig:
    """Noise is realised at load time from the stored seed, not baked in."""

    enabled: bool = True
    incoherent_adc: tuple = (1.0, 4.0)      # per-pixel Gaussian sigma
    coherent_adc: tuple = (0.0, 3.0)        # shared across a channel group
    coherent_group: int = 32
    pink_adc: tuple = (0.0, 2.0)            # 1/f, per channel along ticks
    p_blip: float = 0.3                     # hit-like artefacts
    n_blips: tuple = (1, 20)
    blip_adc: tuple = (5.0, 40.0)


# --------------------------------------------------------------------------- #
# top level
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class DatasetConfig:
    name: str = "unnamed"
    n_events: int = 100
    seed: int = 0
    shard_size: int = 5000
    freeze_noise: bool = False   # True -> bake noise into the stored ADC (test sets)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    event: EventConfig = field(default_factory=EventConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)


# --------------------------------------------------------------------------- #
# (de)serialisation
# --------------------------------------------------------------------------- #

def _build(cls, data: dict):
    """Recursively construct a (nested) dataclass from a plain dict."""
    kwargs = {}
    type_by_name = {f.name: f.type for f in fields(cls)}
    for key, value in data.items():
        if key not in type_by_name:
            raise KeyError(f"{cls.__name__} has no field {key!r}")
        target = type_by_name[key]
        # resolve string annotations from `from __future__ import annotations`
        if isinstance(target, str):
            target = globals().get(target, target)
        if is_dataclass(target) and isinstance(value, dict):
            kwargs[key] = _build(target, value)
        elif isinstance(value, list):
            kwargs[key] = tuple(value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def load_yaml(path) -> DatasetConfig:
    with open(path) as handle:
        data = yaml.safe_load(handle) or {}
    return _build(DatasetConfig, data)


def to_dict(cfg) -> dict[str, Any]:
    return asdict(cfg)


def config_hash(cfg) -> str:
    """Stable 12-hex-character digest of a config, used to name runs/datasets."""
    blob = json.dumps(to_dict(cfg), sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]
