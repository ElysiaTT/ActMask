"""Learned and analytic action-mask models."""

from .actmask_model import ActMaskModel
from .baselines import (
    ActionProximityMask,
    CurrentPositionProximity,
    FuturePositionProximity,
    MotionMagnitudeMask,
    NoMask,
    TimeAlignedTrajectoryProximity,
)
from .temporal_actmask import (
    ActionFrameRelativeTemporalActMask,
    ActMaskMLP,
    RotationInvariantTemporalActMask,
    TemporalActMask,
)
from .correspondence import (
    IndexCorrespondenceTemporalActMask,
    LegacyAxisBiasedNeighborhoodTemporalActMask,
    InvariantFeatureMotionTemporalActMask,
    LocalNeighborhoodTemporalActMask,
    MotionConsistentMutualTemporalActMask,
    SetHistoryTemporalActMask,
)
from .milestone2b_baselines import (
    ConstantVelocityKalmanProximity,
    EstimatedTimeAlignedTrajectoryProximity,
    EstimatedVelocityFutureProximity,
    ExactHiddenTrajectoryOracle,
)

__all__ = [
    "ActMaskModel",
    "ActMaskMLP",
    "TemporalActMask",
    "RotationInvariantTemporalActMask",
    "ActionFrameRelativeTemporalActMask",
    "IndexCorrespondenceTemporalActMask",
    "LegacyAxisBiasedNeighborhoodTemporalActMask",
    "InvariantFeatureMotionTemporalActMask",
    "MotionConsistentMutualTemporalActMask",
    "SetHistoryTemporalActMask",
    "LocalNeighborhoodTemporalActMask",
    "NoMask",
    "MotionMagnitudeMask",
    "ActionProximityMask",
    "CurrentPositionProximity",
    "FuturePositionProximity",
    "TimeAlignedTrajectoryProximity",
    "EstimatedVelocityFutureProximity",
    "EstimatedTimeAlignedTrajectoryProximity",
    "ConstantVelocityKalmanProximity",
    "ExactHiddenTrajectoryOracle",
]
