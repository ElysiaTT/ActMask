"""Feature-ablation models and deterministic evaluation-time shuffles."""

from __future__ import annotations

from enum import Enum
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import Dataset


class FeatureMode(str, Enum):
    FULL = "full"
    NO_VELOCITY = "no_velocity"
    NO_ACTION = "no_action"
    POSITION_ONLY = "position_only"


def canonical_feature_mode(value: FeatureMode | str) -> str:
    raw_value = value.value if isinstance(value, FeatureMode) else value
    normalized = str(raw_value).strip().lower().replace("-", "_")
    allowed = tuple(mode.value for mode in FeatureMode)
    if normalized not in allowed:
        raise ValueError(f"feature mode must be one of {allowed}, got {value!r}")
    return normalized


class AblatedActMask(nn.Module):
    """Bind an ActMaskV2-compatible model to one trained feature mode."""

    def __init__(self, model: nn.Module, feature_mode: FeatureMode | str) -> None:
        super().__init__()
        self.model = model
        self.feature_mode = canonical_feature_mode(feature_mode)

    def forward(self, points: Tensor, velocities: Tensor, action: Tensor) -> Tensor:
        return self.model(
            points, velocities, action, feature_mode=self.feature_mode
        )


# Descriptive alias retained for orchestration code.
FeatureAblationModel = AblatedActMask


def deterministic_derangement(length: int, seed: int) -> tuple[int, ...]:
    """Return one seeded global cycle mapping each recipient to a donor.

    A single cycle is a derangement for every ``length >= 2``.  The mapping is
    generated once for the full dataset, so it is independent of data-loader
    batching and iteration order.
    """

    if not isinstance(length, int) or isinstance(length, bool):
        raise TypeError("length must be an integer")
    if length < 2:
        raise ValueError("a shuffled evaluation dataset needs at least two samples")
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    rng = np.random.default_rng(int(seed))
    cycle = rng.permutation(length)
    donors = np.empty(length, dtype=np.int64)
    donors[cycle] = np.roll(cycle, -1)
    if np.any(donors == np.arange(length)):  # Defensive assertion of the contract.
        raise RuntimeError("internal error: cyclic mapping was not a derangement")
    return tuple(int(index) for index in donors.tolist())


def _python_scalar(value: Any) -> Any:
    if isinstance(value, Tensor):
        if value.numel() != 1:
            raise ValueError("sample_id tensors must contain exactly one value")
        return value.detach().cpu().item()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _clone_value(value: Any) -> Any:
    return value.clone() if isinstance(value, Tensor) else value


class ShuffledFeatureDataset(Dataset[dict[str, Any]]):
    """Replace one input feature from a globally deranged donor sample.

    Only ``velocities`` or ``action`` is replaced.  Labels, points, success,
    scenario, and every original identifier remain those of the recipient.
    Donor identity is added under both a generic and feature-specific metadata
    key for auditable evaluation records.
    """

    _FEATURES = {"velocities": "velocity", "action": "action"}

    def __init__(
        self,
        dataset: Dataset[Mapping[str, Any]],
        feature: str,
        *,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if feature not in self._FEATURES:
            raise ValueError("feature must be 'velocities' or 'action'")
        if len(dataset) < 2:
            raise ValueError("a shuffled evaluation dataset needs at least two samples")
        self.dataset = dataset
        self.feature = feature
        self.seed = int(seed)
        self.donor_indices = deterministic_derangement(len(dataset), self.seed)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if isinstance(index, Tensor):
            if index.numel() != 1:
                raise IndexError("dataset index tensor must contain one value")
            index = int(index.item())
        if not isinstance(index, (int, np.integer)) or isinstance(index, bool):
            raise TypeError("index must be an integer")
        recipient_index = int(index)
        if recipient_index < 0:
            recipient_index += len(self)
        if recipient_index < 0 or recipient_index >= len(self):
            raise IndexError(f"index {index} is outside a dataset of length {len(self)}")

        donor_index = self.donor_indices[recipient_index]
        recipient = self.dataset[recipient_index]
        donor = self.dataset[donor_index]
        if self.feature not in recipient or self.feature not in donor:
            raise KeyError(f"both recipient and donor must contain {self.feature!r}")

        recipient_feature = torch.as_tensor(recipient[self.feature])
        donor_feature = torch.as_tensor(donor[self.feature])
        if recipient_feature.shape != donor_feature.shape:
            raise ValueError(
                f"cannot shuffle {self.feature}: recipient shape "
                f"{tuple(recipient_feature.shape)} != donor shape {tuple(donor_feature.shape)}"
            )

        result = {key: _clone_value(value) for key, value in recipient.items()}
        result[self.feature] = _clone_value(donor[self.feature])
        donor_sample_id = _python_scalar(donor.get("sample_id", donor_index))
        result["shuffled_feature"] = self.feature
        result["shuffle_donor_sample_id"] = donor_sample_id
        prefix = self._FEATURES[self.feature]
        result[f"{prefix}_donor_sample_id"] = donor_sample_id
        return result


class ShuffledVelocityDataset(ShuffledFeatureDataset):
    def __init__(self, dataset: Dataset[Mapping[str, Any]], *, seed: int = 0) -> None:
        super().__init__(dataset, "velocities", seed=seed)


class ShuffledActionDataset(ShuffledFeatureDataset):
    def __init__(self, dataset: Dataset[Mapping[str, Any]], *, seed: int = 0) -> None:
        super().__init__(dataset, "action", seed=seed)


class ShuffledTimeDataset(Dataset[dict[str, Any]]):
    """Cross-sample shuffle of the action-duration field only.

    The Milestone-2 action schema stores execution duration at ``action[6]``.
    This view uses the same global seeded derangement as the velocity/action
    shuffles, but replaces *only* that scalar.  In particular, the action
    endpoints, radius, point inputs, labels, and recipient identifiers remain
    untouched.  The donor identity is carried in the returned metadata so a
    time-shuffled prediction can be audited independently of DataLoader batch
    size or iteration order.
    """

    def __init__(self, dataset: Dataset[Mapping[str, Any]], *, seed: int = 0) -> None:
        super().__init__()
        if len(dataset) < 2:
            raise ValueError("a shuffled evaluation dataset needs at least two samples")
        self.dataset = dataset
        self.seed = int(seed)
        self.donor_indices = deterministic_derangement(len(dataset), self.seed)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if isinstance(index, Tensor):
            if index.numel() != 1:
                raise IndexError("dataset index tensor must contain one value")
            index = int(index.item())
        if not isinstance(index, (int, np.integer)) or isinstance(index, bool):
            raise TypeError("index must be an integer")
        recipient_index = int(index)
        if recipient_index < 0:
            recipient_index += len(self)
        if recipient_index < 0 or recipient_index >= len(self):
            raise IndexError(f"index {index} is outside a dataset of length {len(self)}")

        donor_index = self.donor_indices[recipient_index]
        recipient = self.dataset[recipient_index]
        donor = self.dataset[donor_index]
        if "action" not in recipient or "action" not in donor:
            raise KeyError("both recipient and donor must contain 'action'")

        recipient_action = torch.as_tensor(recipient["action"])
        donor_action = torch.as_tensor(donor["action"])
        if recipient_action.shape != donor_action.shape:
            raise ValueError(
                "cannot shuffle action duration: recipient action shape "
                f"{tuple(recipient_action.shape)} != donor action shape "
                f"{tuple(donor_action.shape)}"
            )
        if recipient_action.ndim < 1 or recipient_action.shape[-1] < 7:
            raise ValueError(
                "action must have a final dimension containing duration at index 6"
            )

        result = {key: _clone_value(value) for key, value in recipient.items()}
        shuffled_action = _clone_value(recipient["action"])
        if not isinstance(shuffled_action, Tensor):
            shuffled_action = torch.as_tensor(shuffled_action).clone()
        donor_duration = donor_action[..., 6]
        shuffled_action[..., 6] = donor_duration.to(
            dtype=shuffled_action.dtype, device=shuffled_action.device
        )
        result["action"] = shuffled_action
        donor_sample_id = _python_scalar(donor.get("sample_id", donor_index))
        result["shuffled_feature"] = "action_duration"
        result["shuffle_donor_sample_id"] = donor_sample_id
        result["time_donor_sample_id"] = donor_sample_id
        result["action_duration_donor_sample_id"] = donor_sample_id
        return result


def shuffled_velocity_dataset(
    dataset: Dataset[Mapping[str, Any]], *, seed: int = 0
) -> ShuffledVelocityDataset:
    return ShuffledVelocityDataset(dataset, seed=seed)


def shuffled_action_dataset(
    dataset: Dataset[Mapping[str, Any]], *, seed: int = 0
) -> ShuffledActionDataset:
    return ShuffledActionDataset(dataset, seed=seed)


def shuffled_time_dataset(
    dataset: Dataset[Mapping[str, Any]], *, seed: int = 0
) -> ShuffledTimeDataset:
    """Return a deterministic view with only ``action[6]`` cross-shuffled."""

    return ShuffledTimeDataset(dataset, seed=seed)


__all__ = [
    "AblatedActMask",
    "FeatureAblationModel",
    "FeatureMode",
    "ShuffledActionDataset",
    "ShuffledFeatureDataset",
    "ShuffledTimeDataset",
    "ShuffledVelocityDataset",
    "canonical_feature_mode",
    "deterministic_derangement",
    "shuffled_action_dataset",
    "shuffled_time_dataset",
    "shuffled_velocity_dataset",
]
