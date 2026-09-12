"""Geometry layer: 3D primitives, event composition, the fiducial volume."""
from . import ood  # noqa: F401  -- registers the OOD shapes in PRIMITIVES
from .compose import DepositedObject, Event3D, compose_event
from .primitives import PRIMITIVES
from .volume import Volume

__all__ = ["PRIMITIVES", "Volume", "Event3D", "DepositedObject", "compose_event"]
