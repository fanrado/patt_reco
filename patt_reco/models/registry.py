"""Model registry, so models are swappable from a config file.

Every model takes the batch dict produced by `PattRecoDataset` and returns a
dict of outputs. Losses and metrics key off *which* outputs are present, so
adding an instance head or a query decoder later never touches the training loop.
"""
from __future__ import annotations

from typing import Callable

_REGISTRY: dict[str, Callable] = {}


def register(name: str):
    def decorator(factory):
        _REGISTRY[name] = factory
        return factory
    return decorator


def build_model(name: str, **kwargs):
    if name not in _REGISTRY:
        raise KeyError(f"unknown model {name!r}; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def available() -> list[str]:
    return sorted(_REGISTRY)


@register("unet")
def _unet(**kwargs):
    from .unet import UNet
    return UNet(**kwargs)
