"""CPU-friendly action-conditioned 3D masking prototype."""

from .data import ToyDynamicDataset
from .models import (
    ActMaskModel,
    ActionProximityMask,
    MotionMagnitudeMask,
    NoMask,
)

__all__ = [
    "ToyDynamicDataset",
    "ActMaskModel",
    "NoMask",
    "MotionMagnitudeMask",
    "ActionProximityMask",
]

__version__ = "0.1.0"
