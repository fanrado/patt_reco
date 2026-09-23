"""Classifier architectures. `CNN` is the minimal baseline later variants are compared against."""
from __future__ import annotations

import inspect
from dataclasses import fields

from .cnn import CNN

__all__ = ["CNN", "build_model"]


def build_model(model_cfg, height: int, width: int) -> CNN:
    """Construct the model from a ModelConfig, mapping every field.

    Call sites used to hand-enumerate the config's fields, so adding a knob
    to ModelConfig meant remembering to edit each one. Forgetting produced a
    model that trained happily under the wrong architecture and reported a
    confident, wrong number -- which is exactly what happened to `n_blocks`.

    Mapping the fields programmatically removes that failure mode: a field
    that the model cannot accept raises here instead of being dropped, so a
    new knob either reaches the model or stops the run.
    """
    accepted = set(inspect.signature(CNN.__init__).parameters) - {"self", "height", "width"}
    names = [f.name for f in fields(model_cfg)]

    unknown = [n for n in names if n not in accepted]
    if unknown:
        raise TypeError(
            f"{type(model_cfg).__name__} has field(s) {', '.join(sorted(unknown))} "
            f"that {CNN.__name__} does not accept. Either add the parameter to "
            f"{CNN.__name__}.__init__ or remove the field -- a config value that "
            f"cannot reach the model would otherwise be silently ignored.")

    return CNN(height, width, **{n: getattr(model_cfg, n) for n in names})
