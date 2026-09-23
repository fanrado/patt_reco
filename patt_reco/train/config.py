"""Configuration dataclasses for training.

Same contract as `patt_reco.config`: frozen nested dataclasses, built from a
YAML file, with no hidden defaults anywhere in the training code. The build,
serialisation and hashing helpers are reused from there rather than
reimplemented.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import yaml

from ..config import _build, config_hash, to_dict  # noqa: F401  -- re-exported


@dataclass(frozen=True)
class ModelConfig:
    n_filters: int = 16
    kernel_size: int = 3
    pool: int = 4
    hidden: int = 64
    dropout: float = 0.0      # 0.0: the baseline carries no hidden regularisation


@dataclass(frozen=True)
class DataConfig:
    # the directory build.py wrote, holding train.npz / val.npz / test.npz
    root: str = "data/tracks_vs_showers"
    batch_size: int = 64
    num_workers: int = 4


@dataclass(frozen=True)
class OptimConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 20
    grad_clip: float = 1.0

    # How many epochs a model needs depends on the dataset size, so an epoch
    # count that is correct at one n_train silently under-trains at another:
    # 8 epochs is 1000 steps at 4000 images/class and 24 at 100/class. This
    # expresses the budget in the unit that actually governs convergence.
    min_steps: int = 0       # 0 disables; otherwise a floor on total steps


@dataclass(frozen=True)
class RunConfig:
    out_dir: str = "runs"
    name: str = ""
    seed: int = 0
    log_every: int = 20


@dataclass(frozen=True)
class TrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    run: RunConfig = field(default_factory=RunConfig)


def load_train_yaml(path) -> TrainConfig:
    with open(path) as handle:
        data = yaml.safe_load(handle) or {}
    return _build(TrainConfig, data)
