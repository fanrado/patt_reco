"""Training configuration. Same pattern as the generator config: frozen
dataclasses, YAML in, no hidden defaults in code."""
from __future__ import annotations

from dataclasses import dataclass, field

import yaml

from ..config import _build, to_dict, config_hash  # noqa: F401  (re-exported)


@dataclass(frozen=True)
class ModelConfig:
    name: str = "unet"
    base_width: int = 32
    depth: int = 4
    in_channels: int = 1


@dataclass(frozen=True)
class DataConfig:
    train: str = "data/l2_noise/train"
    val: str = "data/l2_noise/val"
    batch_size: int = 8
    accum_steps: int = 4          # effective batch = batch_size * accum_steps
    num_workers: int = 4
    augment: bool = True
    mix_prob: float = 0.25        # event-mixing probability, see dataset/augment.py
    max_tick_shift: int = 8
    max_train_events: int = 0     # 0 = all; useful for smoke runs
    max_val_events: int = 2000


@dataclass(frozen=True)
class OptimConfig:
    lr: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 30
    warmup_frac: float = 0.03
    grad_clip: float = 1.0
    amp: bool = True


@dataclass(frozen=True)
class LossConfig:
    focal_gamma: float = 2.0
    focal_weight: float = 1.0
    dice_weight: float = 1.0
    ambiguity_weighting: bool = True
    class_weighting: str = "pixels_per_object"   # or inverse_freq / none
    class_weight_events: int = 200


@dataclass(frozen=True)
class RunConfig:
    out_dir: str = "runs"
    name: str = ""
    seed: int = 0
    preview_events: int = 3
    log_every: int = 50


@dataclass(frozen=True)
class TrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    run: RunConfig = field(default_factory=RunConfig)


def load_train_yaml(path) -> TrainConfig:
    with open(path) as handle:
        return _build(TrainConfig, yaml.safe_load(handle) or {})
