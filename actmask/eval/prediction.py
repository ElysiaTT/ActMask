"""Prediction collection with CPU-resident Milestone-2 records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset


_MODEL_FIELDS = {"points", "velocities", "action", "mask"}
_LARGE_ORACLE_FIELDS = {"point_trajectory", "gripper_trajectory"}
_FIXED_METADATA_FIELDS = {
    "sample_id",
    "group_id",
    "base_scene_id",
    "geometry_seed",
    "scenario_seed",
    "pair_id",
    "variant_id",
    "scenario",
    "success",
    "counterfactual_type",
    "distribution",
    "domain",
    "ood_axis",
    "trajectory_family_id",
    "source_sample_id",
    "ranking_scene_id",
    "candidate_id",
    "candidate_type",
    "candidate_success",
    "candidate_utility",
    "oracle_candidate_id",
    "oracle_utility",
    "temporal_velocity_delay",
}


def _cpu_tensor(value: Any, *, dtype: torch.dtype | None = None) -> Tensor:
    tensor = value.detach().cpu() if isinstance(value, Tensor) else torch.as_tensor(value)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor.clone()


def _scalar(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, Tensor):
        if value.numel() != 1:
            return value.detach().cpu().clone()
        return value.detach().cpu().item()
    if isinstance(value, np.ndarray):
        if value.size != 1:
            return value.copy()
        return value.reshape(-1)[0].item()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _batch_item(value: Any, index: int, batch_size: int) -> Any:
    """Extract one sample from a default-collated batch value."""

    if isinstance(value, Tensor):
        if value.ndim > 0 and value.shape[0] == batch_size:
            return value[index]
        return value
    if isinstance(value, np.ndarray):
        if value.ndim > 0 and value.shape[0] == batch_size:
            return value[index]
        return value
    if isinstance(value, (list, tuple)) and len(value) == batch_size:
        return value[index]
    return value


def _metadata_value(value: Any) -> Any:
    if isinstance(value, Tensor):
        value = value.detach().cpu().clone()
        return value.item() if value.numel() == 1 else value
    if isinstance(value, np.ndarray):
        copied = value.copy()
        return copied.reshape(-1)[0].item() if copied.size == 1 else copied
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass
class PredictionRecord:
    """One point-cloud prediction and the identifiers needed for analysis."""

    probabilities: Tensor
    targets: Tensor
    sample_id: Any = None
    group_id: Any = None
    base_scene_id: Any = None
    geometry_seed: Any = None
    scenario_seed: Any = None
    pair_id: Any = None
    variant_id: Any = None
    scenario: str = "unknown"
    success: float = 0.0
    counterfactual_type: str = "none"
    distribution: str = "ID"
    ood_axis: str = "none"
    ranking_scene_id: Any = None
    candidate_id: Any = None
    candidate_type: str = ""
    candidate_success: float | None = None
    candidate_utility: float | None = None
    oracle_candidate_id: Any = None
    oracle_utility: float | None = None
    changed_input_mask: Tensor | None = None
    invariant_mask: Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    points: Tensor | None = None
    velocities: Tensor | None = None
    action: Tensor | None = None

    def __post_init__(self) -> None:
        self.probabilities = _cpu_tensor(self.probabilities, dtype=torch.float32).reshape(-1)
        self.targets = _cpu_tensor(self.targets).reshape(-1) >= 0.5
        if self.probabilities.shape != self.targets.shape:
            raise ValueError(
                "probabilities and targets must have identical point shapes, got "
                f"{tuple(self.probabilities.shape)} and {tuple(self.targets.shape)}"
            )
        if not torch.isfinite(self.probabilities).all():
            raise ValueError("probabilities must be finite")
        if bool(torch.any((self.probabilities < 0.0) | (self.probabilities > 1.0))):
            raise ValueError("probabilities must lie in [0, 1]")
        for name in ("changed_input_mask", "invariant_mask"):
            value = getattr(self, name)
            if value is not None:
                normalized = _cpu_tensor(value).reshape(-1) >= 0.5
                if normalized.shape != self.targets.shape:
                    raise ValueError(
                        f"{name} must match targets, got {tuple(normalized.shape)} "
                        f"and {tuple(self.targets.shape)}"
                    )
                setattr(self, name, normalized)
        self.scenario = str(self.scenario)
        self.counterfactual_type = str(self.counterfactual_type)
        self.distribution = str(self.distribution)
        self.ood_axis = str(self.ood_axis)
        self.success = float(self.success)
        self.candidate_type = str(self.candidate_type)
        if self.candidate_success is not None:
            self.candidate_success = float(self.candidate_success)
        if self.candidate_utility is not None:
            self.candidate_utility = float(self.candidate_utility)
        if self.oracle_utility is not None:
            self.oracle_utility = float(self.oracle_utility)

    @property
    def target(self) -> Tensor:
        """Compatibility alias for callers using the singular field name."""

        return self.targets

    @property
    def mean_mask_score(self) -> float:
        return float(self.probabilities.mean().item())

    @property
    def score(self) -> float:
        """Compatibility alias used by successful-action ranking."""

        return self.mean_mask_score

    def predictions(self, threshold: float) -> Tensor:
        if not 0.0 <= float(threshold) <= 1.0:
            raise ValueError(f"threshold must lie in [0, 1], got {threshold}")
        return self.probabilities >= float(threshold)


def _loader_from_data(
    data: DataLoader[Any] | Dataset[Mapping[str, Any]],
    *,
    batch_size: int,
    num_workers: int,
) -> Iterable[Mapping[str, Any]]:
    if isinstance(data, DataLoader):
        return data
    if not isinstance(data, Dataset):
        raise TypeError("data must be a torch Dataset or DataLoader")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")
    return DataLoader(
        data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )


def _model_logits(
    model: nn.Module,
    points: Tensor,
    velocities: Tensor,
    actions: Tensor,
    feature_mode: str | None,
) -> Tensor:
    output = (
        model(points, velocities, actions)
        if feature_mode is None
        else model(points, velocities, actions, feature_mode=feature_mode)
    )
    if isinstance(output, Mapping):
        if "logits" not in output:
            raise ValueError("mapping model output must contain a 'logits' field")
        output = output["logits"]
    if not isinstance(output, Tensor):
        raise TypeError("model must return a Tensor or a mapping containing logits")
    return output


def collect_predictions(
    model: nn.Module,
    data: DataLoader[Any] | Dataset[Mapping[str, Any]],
    device: torch.device | str = "cpu",
    feature_mode: str | None = None,
    *,
    batch_size: int = 32,
    num_workers: int = 0,
    distribution: str | None = None,
    ood_axis: str | None = None,
    retain_inputs: bool = False,
    retain_oracle_metadata: bool = False,
    from_logits: bool = True,
) -> list[PredictionRecord]:
    """Run a model and retain per-sample CPU probabilities, targets, and IDs.

    ``feature_mode`` is forwarded only when explicitly supplied, keeping the
    function compatible with both ActMaskV2 and the parameter-free baselines.
    Large exact trajectories are omitted from ``metadata`` unless
    ``retain_oracle_metadata`` is requested; all scalar/vector M2 metadata and
    counterfactual masks are retained.
    """

    loader = _loader_from_data(data, batch_size=batch_size, num_workers=num_workers)
    target_device = torch.device(device)
    model = model.to(target_device)
    was_training = model.training
    model.eval()
    records: list[PredictionRecord] = []
    source_offset = 0

    try:
        with torch.inference_mode():
            for batch in loader:
                if not isinstance(batch, Mapping):
                    raise TypeError("each data batch must be a mapping")
                missing = _MODEL_FIELDS.difference(batch)
                if missing:
                    raise KeyError(f"data batch is missing required fields: {sorted(missing)}")
                points = torch.as_tensor(batch["points"], dtype=torch.float32, device=target_device)
                velocities = torch.as_tensor(
                    batch["velocities"], dtype=torch.float32, device=target_device
                )
                actions = torch.as_tensor(
                    batch["action"], dtype=torch.float32, device=target_device
                )
                targets = torch.as_tensor(
                    batch["mask"], dtype=torch.float32, device=target_device
                )
                logits = _model_logits(model, points, velocities, actions, feature_mode)
                if logits.shape != targets.shape:
                    raise RuntimeError(
                        f"model returned {tuple(logits.shape)} for target "
                        f"{tuple(targets.shape)}"
                    )
                probabilities = torch.sigmoid(logits) if from_logits else logits
                probabilities = probabilities.detach().cpu()
                cpu_targets = targets.detach().cpu()
                batch_count = int(targets.shape[0])

                for index in range(batch_count):
                    def item(key: str, default: Any = None) -> Any:
                        if key not in batch:
                            return default
                        return _batch_item(batch[key], index, batch_count)

                    raw_distribution = (
                        distribution
                        if distribution is not None
                        else item("distribution", item("domain", "ID"))
                    )
                    normalized_distribution = str(_scalar(raw_distribution, "ID"))
                    if normalized_distribution.lower() == "id":
                        normalized_distribution = "ID"
                    elif normalized_distribution.lower() == "ood":
                        normalized_distribution = "OOD"
                    raw_ood_axis = ood_axis if ood_axis is not None else item("ood_axis", "none")

                    metadata: dict[str, Any] = {}
                    for key, value in batch.items():
                        if key in _MODEL_FIELDS or key in _FIXED_METADATA_FIELDS:
                            continue
                        if key in {"changed_input_mask", "invariant_mask"}:
                            continue
                        if key in _LARGE_ORACLE_FIELDS and not retain_oracle_metadata:
                            continue
                        metadata[key] = _metadata_value(
                            _batch_item(value, index, batch_count)
                        )

                    record = PredictionRecord(
                        probabilities=probabilities[index],
                        targets=cpu_targets[index],
                        sample_id=_scalar(item("sample_id"), source_offset + index),
                        group_id=_scalar(item("group_id")),
                        base_scene_id=_scalar(item("base_scene_id")),
                        geometry_seed=_scalar(item("geometry_seed")),
                        scenario_seed=_scalar(item("scenario_seed")),
                        pair_id=_scalar(item("pair_id")),
                        variant_id=_scalar(item("variant_id")),
                        scenario=str(_scalar(item("scenario"), "unknown")),
                        success=float(_scalar(item("success"), 0.0)),
                        counterfactual_type=str(
                            _scalar(item("counterfactual_type"), "none")
                        ),
                        distribution=normalized_distribution,
                        ood_axis=str(_scalar(raw_ood_axis, "none")),
                        ranking_scene_id=_scalar(item("ranking_scene_id")),
                        candidate_id=_scalar(item("candidate_id")),
                        candidate_type=str(_scalar(item("candidate_type"), "")),
                        candidate_success=(
                            float(_scalar(item("candidate_success")))
                            if "candidate_success" in batch
                            else None
                        ),
                        candidate_utility=(
                            float(_scalar(item("candidate_utility")))
                            if "candidate_utility" in batch
                            else None
                        ),
                        oracle_candidate_id=_scalar(item("oracle_candidate_id")),
                        oracle_utility=(
                            float(_scalar(item("oracle_utility")))
                            if "oracle_utility" in batch
                            else None
                        ),
                        changed_input_mask=(
                            _cpu_tensor(item("changed_input_mask"))
                            if "changed_input_mask" in batch
                            else None
                        ),
                        invariant_mask=(
                            _cpu_tensor(item("invariant_mask"))
                            if "invariant_mask" in batch
                            else None
                        ),
                        metadata=metadata,
                        points=_cpu_tensor(points[index]) if retain_inputs else None,
                        velocities=_cpu_tensor(velocities[index]) if retain_inputs else None,
                        action=_cpu_tensor(actions[index]) if retain_inputs else None,
                    )
                    records.append(record)
                source_offset += batch_count
    finally:
        model.train(was_training)

    if not records:
        raise ValueError("cannot collect predictions from an empty dataset")
    return records


__all__ = ["PredictionRecord", "collect_predictions"]
