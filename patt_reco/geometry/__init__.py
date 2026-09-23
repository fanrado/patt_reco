"""Geometry layer: the 3D track and shower primitives, and the generation box.

Each image holds exactly one object, so there is no event composition here and
no out-of-distribution arm.
"""
from .base import Primitive
from .track import Track
from .shower import Shower
from .volume import Volume

PRIMITIVES = {"track": Track, "shower": Shower}

__all__ = ["Primitive", "Track", "Shower", "Volume", "PRIMITIVES"]
