"""Standalone CPU-only temporal and shortcut diagnostics for Milestone 2E.

The full 2E runner deliberately does not invoke this module.  It never trains
models, changes the frozen candidate worlds or labels, and writes one isolated
artifact: ``outputs/actmask/milestone2e/temporal_shortcut_diagnostic.json``.

The diagnostic loads only provenance-compatible checkpoints chosen by the
existing validation-only primary-selection artifact.  By default it evaluates
the selected verifier on the frozen C20 test worlds under Phase 4 temporal
views, then fills the missing Phase 5 ranking-head/shortcut reporting.  The
``--all-validation-promising`` mode expands *Phase 4 only* to every fair arm
that passes a fixed, validation-only five-seed proximity rule.  It never uses
test results to decide that set and does not change the primary selection.
In particular, learned-only scores are reported alongside the frozen hybrid:
the latter's analytic half can otherwise mask a learned verifier that ignores
history.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from actmask.data.milestone2b_dataset import _clone_tensor_tree, _estimate_velocity, validate_observable_mapping
from actmask.eval.calibration import reliability_metrics
from actmask.eval.statistics import paired_summary
from actmask.experiments.milestone2c import _loader, _metadata_rows
from actmask.experiments.milestone2e import (
    DEFAULT_CONFIG,
    OUTPUT_DIR,
    RANKING_METRICS,
    RankingBundle,
    _load_config,
    _write_json,
    paired_ranking_report,
    ranking_metrics_tie_aware,
    validate_checkpoint_metadata,
)
from actmask.experiments.milestone2e_full import (
    FULL_OUTPUT_DIR,
    PRIMARY_HEAD,
    DiagnosticIdentityDataset,
    FactorialPrediction,
    _fair_prediction,
    _hybrid_scores,
    _json_digest,
    _model,
    build_datasets,
)


DEFAULT_OUTPUT_PATH = OUTPUT_DIR / "temporal_shortcut_diagnostic.json"
DEFAULT_ALL_PROMISING_OUTPUT_PATH = OUTPUT_DIR / "temporal_shortcut_diagnostic_all_validation_promising.json"
PRIMARY_SELECTION_FILE = "validation_primary_selection.json"
PRIMARY_TRAINING_RECORDS_FILE = "training_records_primary.json"
CANDIDATE_COUNT = 20
DIAGNOSTIC_SEED = 20261271

# This is deliberately a code-level constant rather than a post-hoc threshold
# chosen after inspecting a temporal test view.  The best fair arm is the
# existing validation-primary arm (lowest mean five-seed C20 normalized regret,
# then highest mean five-seed C20 Top-1).  An arm must meet *both* limits.
VALIDATION_PROMISING_MAX_REGRET_ABOVE_BEST = 0.060
VALIDATION_PROMISING_MAX_TOP1_BELOW_BEST = 0.050
VALIDATION_PROMISING_CRITERION = (
    "A fair primary-head arm is Phase-4-promising iff its mean five-seed "
    "validation C20 normalized regret is no more than 0.060 above the "
    "validation-selected best fair arm and its mean validation C20 Top-1 is "
    "no more than 0.050 below that same arm."
)

# The first twelve names mirror Phase 4 exactly.  ``scene_only`` is the extra
# Phase 5 control; it uses the existing no-action feature path and also makes
# the analytic comparator action-blind at the dataset boundary.
TEMPORAL_SHORTCUT_VIEWS: tuple[str, ...] = (
    "correct_history",
    "reversed_history",
    "independently_permuted_frames",
    "duplicated_final_frame",
    "mismatched_history",
    "zeroed_motion",
    "shuffled_motion",
    "removed_timestamps",
    "shuffled_timestamps",
    "action_only",
    "static_geometry_only",
    "candidate_template_only",
    "scene_only",
)
_VIEW_ALIASES = {"permuted_frames": "independently_permuted_frames"}
_VIEW_DESCRIPTIONS = {
    "correct_history": "Unchanged frozen observable history.",
    "reversed_history": "History/visibility order reversed while timestamp slots remain ordered.",
    "independently_permuted_frames": "A deterministic non-identity frame permutation per candidate world.",
    "duplicated_final_frame": "Every history slot is the final frame; timestamps remain unchanged.",
    "mismatched_history": "History fields come from another frozen candidate world; action and labels stay local.",
    "zeroed_motion": "Raw and post-correspondence velocity/confidence are zeroed.",
    "shuffled_motion": "Raw and post-correspondence velocity/confidence are point-permuted within a world.",
    "removed_timestamps": "Timestamps become a fixed uniform grid ending at zero.",
    "shuffled_timestamps": "Original time intervals are deterministically reordered but stay increasing.",
    "action_only": "All scene/history/timing fields are constants; only action_command remains candidate-specific.",
    "static_geometry_only": "Final static geometry plus action, with the learned core's no-history path.",
    "candidate_template_only": "Action-only input with translation/rotation removed; length, duration, and radius remain.",
    "scene_only": "Action command/timing are constants and the learned core's no-action path is used.",
}
_POST_CANONICAL_MOTION_VIEWS = frozenset(("zeroed_motion", "shuffled_motion"))


def _canonical_view(value: str) -> str:
    value = _VIEW_ALIASES.get(str(value), str(value))
    if value not in TEMPORAL_SHORTCUT_VIEWS:
        raise ValueError(f"unknown temporal diagnostic view {value!r}")
    return value


def _group_rng(group_id: int, salt: int) -> np.random.Generator:
    words = [DIAGNOSTIC_SEED, int(group_id) & 0xFFFFFFFF, (int(group_id) >> 32) & 0xFFFFFFFF, int(salt)]
    return np.random.default_rng(np.random.SeedSequence(words))


def _nonidentity_permutation(size: int, *, group_id: int, salt: int) -> torch.Tensor:
    if size < 2:
        raise ValueError("a non-identity temporal permutation needs at least two entries")
    values = _group_rng(group_id, salt).permutation(size)
    if np.array_equal(values, np.arange(size)):
        values[[0, 1]] = values[[1, 0]]
    return torch.as_tensor(values, dtype=torch.long)


def _uniform_timestamps(timestamps: torch.Tensor) -> torch.Tensor:
    """Fixed 10 Hz grid ending at zero, removing scene-specific timing jitter."""

    frames = int(timestamps.numel())
    return (torch.arange(frames, dtype=timestamps.dtype, device=timestamps.device) - (frames - 1)) * 0.1


def _refresh_observable_motion(observable: Mapping[str, torch.Tensor]) -> None:
    """Recompute the public velocity estimate after a history/timestamp edit."""

    history = observable["points_history"]
    visibility = observable["visibility_history"]
    timestamps = observable["timestamps"]
    if history.device.type != "cpu" or visibility.device.type != "cpu" or timestamps.device.type != "cpu":
        raise RuntimeError("Milestone 2E temporal diagnostics are CPU-only")
    velocity, confidence = _estimate_velocity(
        history.detach().numpy(), visibility.detach().numpy(), timestamps.detach().numpy()
    )
    observable["estimated_velocity"] = torch.as_tensor(velocity, dtype=history.dtype)
    observable["velocity_confidence"] = torch.as_tensor(confidence, dtype=history.dtype)


def _identity_history(sample: Mapping[str, Any]) -> torch.Tensor | None:
    hidden = sample.get("hidden_state")
    if not isinstance(hidden, Mapping):
        return None
    value = hidden.get("evaluation_identity_history")
    return value if isinstance(value, torch.Tensor) else None


def _set_identity_history(sample: dict[str, Any], value: torch.Tensor | None) -> None:
    if value is not None:
        sample.setdefault("hidden_state", {})["evaluation_identity_history"] = value


def _copy_mismatched_scene(target: dict[str, Any], source: Mapping[str, Any]) -> None:
    """Copy only observable scene fields; target action/labels/metadata remain frozen."""

    target_observable = target["observable"]
    source_observable = source["observable"]
    for key in (
        "points_history",
        "visibility_history",
        "timestamps",
        "estimated_velocity",
        "velocity_confidence",
        "observation_delay",
    ):
        target_observable[key] = source_observable[key].clone()
    _set_identity_history(target, _identity_history(source).clone() if _identity_history(source) is not None else None)


def _blank_scene(observable: Mapping[str, torch.Tensor]) -> None:
    history = observable["points_history"]
    observable["points_history"] = torch.zeros_like(history)
    observable["visibility_history"] = torch.zeros_like(observable["visibility_history"], dtype=torch.bool)
    observable["timestamps"] = _uniform_timestamps(observable["timestamps"])
    observable["estimated_velocity"] = torch.zeros_like(observable["estimated_velocity"])
    observable["velocity_confidence"] = torch.zeros_like(observable["velocity_confidence"])
    observable["observation_delay"] = torch.zeros_like(observable["observation_delay"])
    # Nominal delay is an input separate from action_command.  Keeping it
    # would make this a timing-plus-action diagnostic rather than action-only.
    observable["nominal_action_delay"] = torch.zeros_like(observable["nominal_action_delay"])


def _canonical_template_action(action: torch.Tensor) -> torch.Tensor:
    """Remove global placement/orientation without consulting template metadata."""

    result = torch.zeros_like(action)
    length = torch.linalg.vector_norm(action[3:6] - action[:3])
    result[3] = length
    result[6:] = action[6:]
    return result


def _apply_dataset_view(
    sample: dict[str, Any], *, view: str, group_id: int, mismatched_source: Mapping[str, Any] | None
) -> None:
    """Mutate a private sample clone only; targets and metadata are untouched."""

    view = _canonical_view(view)
    observable = sample["observable"]
    history_changed = False
    if view == "mismatched_history":
        if mismatched_source is None:
            raise ValueError("mismatched_history needs another candidate world")
        _copy_mismatched_scene(sample, mismatched_source)
        observable = sample["observable"]
        history_changed = True
    elif view == "reversed_history":
        observable["points_history"] = observable["points_history"].flip(0)
        observable["visibility_history"] = observable["visibility_history"].flip(0)
        identity = _identity_history(sample)
        _set_identity_history(sample, identity.flip(0) if identity is not None else None)
        history_changed = True
    elif view == "independently_permuted_frames":
        permutation = _nonidentity_permutation(int(observable["points_history"].shape[0]), group_id=group_id, salt=31)
        observable["points_history"] = observable["points_history"][permutation]
        observable["visibility_history"] = observable["visibility_history"][permutation]
        identity = _identity_history(sample)
        _set_identity_history(sample, identity[permutation] if identity is not None else None)
        history_changed = True
    elif view in {"duplicated_final_frame", "static_geometry_only"}:
        frames = int(observable["points_history"].shape[0])
        observable["points_history"] = observable["points_history"][-1:].expand(frames, -1, -1).clone()
        observable["visibility_history"] = observable["visibility_history"][-1:].expand(frames, -1).clone()
        identity = _identity_history(sample)
        _set_identity_history(sample, identity[-1:].expand(frames, -1).clone() if identity is not None else None)
        history_changed = True
    elif view == "removed_timestamps":
        observable["timestamps"] = _uniform_timestamps(observable["timestamps"])
        history_changed = True
    elif view == "shuffled_timestamps":
        timestamps = observable["timestamps"]
        intervals = timestamps[1:] - timestamps[:-1]
        permutation = _nonidentity_permutation(int(intervals.numel()), group_id=group_id, salt=47).to(intervals.device)
        shuffled = intervals[permutation]
        observable["timestamps"] = torch.cat((timestamps[:1], timestamps[:1] + torch.cumsum(shuffled, dim=0)))
        history_changed = True
    elif view in {"action_only", "candidate_template_only"}:
        _blank_scene(observable)
        if view == "candidate_template_only":
            observable["action_command"] = _canonical_template_action(observable["action_command"])
    elif view == "scene_only":
        observable["action_command"] = torch.zeros_like(observable["action_command"])
        observable["nominal_action_delay"] = torch.zeros_like(observable["nominal_action_delay"])

    if history_changed:
        _refresh_observable_motion(observable)
    elif view == "zeroed_motion":
        observable["estimated_velocity"] = torch.zeros_like(observable["estimated_velocity"])
        observable["velocity_confidence"] = torch.zeros_like(observable["velocity_confidence"])
    elif view == "shuffled_motion":
        point_permutation = _nonidentity_permutation(
            int(observable["estimated_velocity"].shape[0]), group_id=group_id, salt=59
        ).to(observable["estimated_velocity"].device)
        observable["estimated_velocity"] = observable["estimated_velocity"][point_permutation]
        observable["velocity_confidence"] = observable["velocity_confidence"][point_permutation]
    validate_observable_mapping(observable)


class TemporalDiagnosticDataset(Dataset[dict[str, Any]]):
    """Deterministic observable-only temporal/shortcut view of complete worlds.

    Every candidate in a world receives exactly the same scene/history
    perturbation.  The wrapper clones before editing, so it cannot alter the
    frozen dataset, candidate labels, metadata, or hidden physical state.
    """

    def __init__(self, dataset: Dataset[dict[str, Any]], *, view: str) -> None:
        self.dataset = dataset
        self.view = _canonical_view(view)
        records = tuple(getattr(dataset, "group_records", ()))
        if not records:
            raise TypeError("temporal diagnostics require a complete grouped ranking dataset")
        if len(dataset) % len(records):
            raise ValueError("dataset does not contain complete candidate worlds")
        self.group_records = records
        self.candidate_count = len(dataset) // len(records)
        self._group_order = [int(record.group_id) for record in records]
        if len(set(self._group_order)) != len(self._group_order):
            raise ValueError("candidate world identifiers must be unique")
        self._group_position = {group_id: position for position, group_id in enumerate(self._group_order)}

    def __len__(self) -> int:
        return len(self.dataset)

    def _mismatched_source(self, group_id: int) -> Mapping[str, Any] | None:
        if self.view != "mismatched_history":
            return None
        if len(self._group_order) < 2:
            raise ValueError("mismatched-history diagnostic requires at least two candidate worlds")
        position = self._group_position[group_id]
        offset = int(_group_rng(group_id, 71).integers(1, len(self._group_order)))
        source_position = (position + offset) % len(self._group_order)
        return self.dataset[source_position * self.candidate_count]

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = _clone_tensor_tree(self.dataset[index])
        group_id = int(sample["metadata"]["candidate_set_id"])
        _apply_dataset_view(
            sample,
            view=self.view,
            group_id=group_id,
            mismatched_source=self._mismatched_source(group_id),
        )
        return sample


# The design name is retained for callers which use the more descriptive
# ``TemporalShortcutDataset`` spelling.
TemporalShortcutDataset = TemporalDiagnosticDataset


class _WorldPrefix(Dataset[dict[str, Any]]):
    """A complete leading set of frozen worlds; never a partial candidate set."""

    def __init__(self, dataset: Dataset[dict[str, Any]], *, groups: int) -> None:
        records = tuple(getattr(dataset, "group_records", ()))
        if groups < 2 or groups > len(records):
            raise ValueError("groups must be between two and the frozen test-world count")
        if len(dataset) % len(records):
            raise ValueError("dataset does not contain complete candidate worlds")
        self.dataset = dataset
        self.group_records = records[:groups]
        self._length = groups * (len(dataset) // len(records))

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += self._length
        if index < 0 or index >= self._length:
            raise IndexError(index)
        return self.dataset[index]


@dataclass(frozen=True)
class CheckpointSpec:
    correspondence: str
    backbone: str
    ranking_head: str
    seed: int
    path: Path
    calibration: dict[str, float]
    best_epoch: int


@dataclass(frozen=True)
class ValidationPromisingVariant:
    """One complete fair arm admitted by the fixed validation-only rule."""

    correspondence: str
    backbone: str
    seeds: tuple[int, ...]
    mean_validation_normalized_regret: float
    mean_validation_top1: float
    regret_above_best: float
    top1_below_best: float

    @property
    def key(self) -> tuple[str, str]:
        return self.correspondence, self.backbone

    @property
    def token(self) -> str:
        return f"{self.correspondence}__{self.backbone}"

    def report_row(self) -> dict[str, Any]:
        return {
            "correspondence": self.correspondence,
            "backbone": self.backbone,
            "training_seeds": list(self.seeds),
            "mean_validation_normalized_regret": self.mean_validation_normalized_regret,
            "mean_validation_top1": self.mean_validation_top1,
            "normalized_regret_above_best": self.regret_above_best,
            "top1_below_best": self.top1_below_best,
            "passes_fixed_validation_promising_rule": True,
        }


def _variant_key(selection: Mapping[str, str]) -> tuple[str, str]:
    return str(selection["correspondence"]), str(selection["backbone"])


def _fair_primary_variants(config: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    """All non-oracle factorial arms expected in the primary-record artifact."""

    correspondences = tuple(str(value) for value in config["correspondence"]["variants"])
    backbones = tuple(str(value) for value in config["backbone"]["variants"])
    fair_correspondences = tuple(value for value in correspondences if value != "GroundTruthIdentityCorrespondence")
    if not fair_correspondences or not backbones:
        raise ValueError("2E configuration has no fair correspondence/backbone factorial arms")
    return tuple((correspondence, backbone) for correspondence in fair_correspondences for backbone in backbones)


def _finite_record_float(record: Mapping[str, Any], field: str, *, label: str) -> float:
    try:
        value = float(record[field])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{label} lacks finite {field}") from error
    if not np.isfinite(value):
        raise ValueError(f"{label} has non-finite {field}")
    return value


def _validation_promising_from_records(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_seeds: Sequence[int],
    fair_variants: Sequence[tuple[str, str]],
    selected_primary: Mapping[str, str],
) -> tuple[list[ValidationPromisingVariant], dict[str, Any]]:
    """Derive all Phase-4 candidates from complete five-seed validation rows.

    This function is intentionally independent of datasets and test outputs.
    It refuses partial factorial records, non-primary-head rows, duplicate
    seeds, or a mismatch with the existing validation primary selection before
    a temporal test view can be evaluated.
    """

    expected_seed_tuple = tuple(int(seed) for seed in expected_seeds)
    if len(expected_seed_tuple) != 5 or len(set(expected_seed_tuple)) != 5:
        raise ValueError("validation-promising selection requires exactly five distinct configured seeds")
    expected_keys = tuple((str(correspondence), str(backbone)) for correspondence, backbone in fair_variants)
    if len(set(expected_keys)) != len(expected_keys):
        raise ValueError("fair factorial variants must be unique")
    expected_key_set = set(expected_keys)
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {key: [] for key in expected_keys}

    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"primary training record {index} is not a mapping")
        try:
            correspondence = str(record["correspondence"])
            backbone = str(record["backbone"])
        except KeyError as error:
            raise ValueError(f"primary training record {index} lacks architecture provenance") from error
        if correspondence == "GroundTruthIdentityCorrespondence":
            # The explicitly diagnostic oracle is neither a fair candidate nor
            # an eligibility reference for the temporal experiment.
            continue
        key = correspondence, backbone
        if key not in expected_key_set:
            raise ValueError(f"primary training record {index} has unknown fair arm {key}")
        if str(record.get("ranking_head")) != PRIMARY_HEAD:
            raise ValueError(f"primary training record {index} is not a {PRIMARY_HEAD} checkpoint")
        try:
            seed = int(record["seed"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"primary training record {index} lacks an integer seed") from error
        if seed not in expected_seed_tuple:
            raise ValueError(f"primary training record {index} has unexpected seed {seed}")
        _finite_record_float(record, "validation_normalized_regret", label=f"primary training record {index}")
        _finite_record_float(record, "validation_top1", label=f"primary training record {index}")
        grouped[key].append(record)

    summaries: list[dict[str, Any]] = []
    for key in expected_keys:
        rows = grouped[key]
        seeds = tuple(int(row["seed"]) for row in rows)
        if len(rows) != len(expected_seed_tuple) or set(seeds) != set(expected_seed_tuple):
            raise ValueError(
                f"primary training records for {key} are incomplete or duplicate; "
                f"expected exactly seeds {list(expected_seed_tuple)}, got {list(seeds)}"
            )
        summaries.append(
            {
                "correspondence": key[0],
                "backbone": key[1],
                "training_seeds": list(expected_seed_tuple),
                "mean_validation_normalized_regret": float(
                    np.mean([_finite_record_float(row, "validation_normalized_regret", label=str(key)) for row in rows])
                ),
                "mean_validation_top1": float(
                    np.mean([_finite_record_float(row, "validation_top1", label=str(key)) for row in rows])
                ),
            }
        )

    best = min(
        summaries,
        key=lambda row: (
            float(row["mean_validation_normalized_regret"]),
            -float(row["mean_validation_top1"]),
            str(row["correspondence"]),
            str(row["backbone"]),
        ),
    )
    selected_key = _variant_key(selected_primary)
    best_key = str(best["correspondence"]), str(best["backbone"])
    if selected_key != best_key:
        raise ValueError(
            "validation_primary_selection.json disagrees with complete primary training records: "
            f"selected {selected_key}, derived best fair arm {best_key}"
        )

    best_regret = float(best["mean_validation_normalized_regret"])
    best_top1 = float(best["mean_validation_top1"])
    all_rows: list[dict[str, Any]] = []
    promising: list[ValidationPromisingVariant] = []
    for row in sorted(
        summaries,
        key=lambda value: (
            float(value["mean_validation_normalized_regret"]),
            -float(value["mean_validation_top1"]),
            str(value["correspondence"]),
            str(value["backbone"]),
        ),
    ):
        regret_delta = float(row["mean_validation_normalized_regret"]) - best_regret
        top1_gap = best_top1 - float(row["mean_validation_top1"])
        qualifies = bool(
            regret_delta <= VALIDATION_PROMISING_MAX_REGRET_ABOVE_BEST + 1.0e-12
            and top1_gap <= VALIDATION_PROMISING_MAX_TOP1_BELOW_BEST + 1.0e-12
        )
        summary_row = {
            **row,
            "normalized_regret_above_best": regret_delta,
            "top1_below_best": top1_gap,
            "passes_fixed_validation_promising_rule": qualifies,
        }
        all_rows.append(summary_row)
        if qualifies:
            promising.append(
                ValidationPromisingVariant(
                    correspondence=str(row["correspondence"]),
                    backbone=str(row["backbone"]),
                    seeds=expected_seed_tuple,
                    mean_validation_normalized_regret=float(row["mean_validation_normalized_regret"]),
                    mean_validation_top1=float(row["mean_validation_top1"]),
                    regret_above_best=regret_delta,
                    top1_below_best=top1_gap,
                )
            )
    if selected_key not in {variant.key for variant in promising}:
        raise AssertionError("the validation-selected primary must satisfy its own fixed promising rule")
    evidence = {
        "source": PRIMARY_TRAINING_RECORDS_FILE,
        "selection_reference": {
            "correspondence": best_key[0],
            "backbone": best_key[1],
            "mean_validation_normalized_regret": best_regret,
            "mean_validation_top1": best_top1,
        },
        "fixed_criterion": VALIDATION_PROMISING_CRITERION,
        "maximum_normalized_regret_above_best": VALIDATION_PROMISING_MAX_REGRET_ABOVE_BEST,
        "maximum_top1_below_best": VALIDATION_PROMISING_MAX_TOP1_BELOW_BEST,
        "all_fair_primary_arms": all_rows,
        "promising_fair_arms": [variant.report_row() for variant in promising],
    }
    return promising, evidence


def _validation_promising_variants(
    *, config: Mapping[str, Any], output_root: Path, selected_primary: Mapping[str, str]
) -> tuple[list[ValidationPromisingVariant], dict[str, Any]]:
    """Load the validation-only factorial evidence and apply the fixed rule."""

    path = output_root / PRIMARY_TRAINING_RECORDS_FILE
    if not path.exists():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text())
    if not isinstance(value, list):
        raise ValueError(f"expected a list of primary training records: {path}")
    return _validation_promising_from_records(
        value,
        expected_seeds=tuple(int(seed) for seed in config["experiment"]["training_seeds"]),
        fair_variants=_fair_primary_variants(config),
        selected_primary=selected_primary,
    )


def _selection(output_root: Path) -> dict[str, str] | None:
    path = output_root / PRIMARY_SELECTION_FILE
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    if not isinstance(value, Mapping) or not isinstance(value.get("correspondence"), str) or not isinstance(value.get("backbone"), str):
        raise ValueError(f"invalid validation-only selection artifact: {path}")
    if value["correspondence"] == "GroundTruthIdentityCorrespondence":
        raise ValueError("the selected primary verifier may not be the diagnostic-only identity arm")
    return {"correspondence": str(value["correspondence"]), "backbone": str(value["backbone"])}


def _checkpoint_path(output_root: Path, correspondence: str, backbone: str, ranking_head: str, seed: int) -> Path:
    token = f"{correspondence}__{backbone}__{ranking_head}__seed{seed}"
    return output_root / "checkpoints" / f"{token}.pt"


def _load_checkpoint(
    *, config: Mapping[str, Any], output_root: Path, correspondence: str, backbone: str, ranking_head: str, seed: int
) -> tuple[Any, CheckpointSpec]:
    """Load a selected CPU checkpoint, rejecting stale/cross-arm provenance."""

    path = _checkpoint_path(output_root, correspondence, backbone, ranking_head, seed)
    if not path.exists():
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"checkpoint is not a mapping: {path}")
    validate_checkpoint_metadata(
        checkpoint,
        {
            "architecture": "FactorialTemporalActMask",
            "correspondence": correspondence,
            "backbone": backbone,
            "ranking_head": ranking_head,
            "seed": int(seed),
            "config_digest": _json_digest(config),
        },
    )
    calibration = checkpoint.get("calibration")
    if not isinstance(calibration, Mapping) or not {"coefficient", "intercept"}.issubset(calibration):
        raise ValueError(f"checkpoint lacks validation-only Platt calibration: {path}")
    model = _model(config, correspondence, backbone)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.cpu().eval()
    if any(parameter.device.type != "cpu" for parameter in model.parameters()):
        raise RuntimeError("temporal diagnostics attempted to load a non-CPU model")
    spec = CheckpointSpec(
        correspondence=correspondence,
        backbone=backbone,
        ranking_head=ranking_head,
        seed=int(seed),
        path=path,
        calibration={"coefficient": float(calibration["coefficient"]), "intercept": float(calibration["intercept"])},
        best_epoch=int(checkpoint.get("best_epoch", 0)),
    )
    return model, spec


def _load_head_models(
    *, config: Mapping[str, Any], output_root: Path, selection: Mapping[str, str], ranking_head: str
) -> tuple[dict[int, Any], dict[int, CheckpointSpec], list[str]]:
    models: dict[int, Any] = {}
    specs: dict[int, CheckpointSpec] = {}
    errors: list[str] = []
    for value in config["experiment"]["training_seeds"]:
        seed = int(value)
        try:
            model, spec = _load_checkpoint(
                config=config,
                output_root=output_root,
                correspondence=str(selection["correspondence"]),
                backbone=str(selection["backbone"]),
                ranking_head=ranking_head,
                seed=seed,
            )
        except (FileNotFoundError, ValueError, RuntimeError) as error:
            errors.append(f"seed {seed}: {error}")
        else:
            models[seed] = model
            specs[seed] = spec
    return models, specs, errors


def _feature_mode(view: str) -> str:
    if view == "static_geometry_only":
        return "no_history"
    if view == "scene_only":
        return "no_action"
    return "full"


def _batch_group_ids(batch: Mapping[str, Any]) -> list[int]:
    value = batch["metadata"]["candidate_set_id"]
    if not isinstance(value, torch.Tensor):
        return [int(item) for item in value]
    return [int(item) for item in value.detach().cpu().tolist()]


def _post_canonical_motion(canonical: dict[str, torch.Tensor], batch: Mapping[str, Any], *, view: str) -> None:
    """Apply motion ablations after correspondence has overwritten raw motion."""

    if view not in _POST_CANONICAL_MOTION_VIEWS:
        return
    if view == "zeroed_motion":
        canonical["estimated_velocity"] = torch.zeros_like(canonical["estimated_velocity"])
        canonical["velocity_confidence"] = torch.zeros_like(canonical["velocity_confidence"])
        return
    velocity = canonical["estimated_velocity"].clone()
    confidence = canonical["velocity_confidence"].clone()
    for row, group_id in enumerate(_batch_group_ids(batch)):
        permutation = _nonidentity_permutation(int(velocity.shape[1]), group_id=group_id, salt=83).to(velocity.device)
        velocity[row] = velocity[row, permutation]
        confidence[row] = confidence[row, permutation]
    canonical["estimated_velocity"] = velocity
    canonical["velocity_confidence"] = confidence


def _forward_diagnostic(model: Any, batch: Mapping[str, Any], *, view: str) -> dict[str, torch.Tensor]:
    """Forward with a view-specific post-correspondence motion intervention."""

    observable = batch["observable"]
    if model.correspondence == "GroundTruthIdentityCorrespondence":
        canonical = model.canonicalize(observable, identity_history=batch["hidden_state"]["evaluation_identity_history"])
    else:
        canonical = model.canonicalize(observable)
    _post_canonical_motion(canonical, batch, view=view)
    return model.core(canonical, feature_mode=_feature_mode(view))


@torch.no_grad()
def predict_temporal_diagnostic(model: Any, dataset: Dataset[Any], *, batch_size: int, view: str) -> FactorialPrediction:
    """Predict strict-observable learned scores without evaluating mask AP."""

    probabilities: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    success: list[np.ndarray] = []
    utility: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    model.eval()
    for batch in _loader(dataset, batch_size):
        output = _forward_diagnostic(model, batch, view=view)
        raw = output["utility_logits"].detach().cpu().numpy()
        logits.append(raw)
        probabilities.append(1.0 / (1.0 + np.exp(-raw)))
        success.append(batch["targets"]["success"].detach().cpu().numpy())
        utility.append(batch["targets"]["candidate_utility"].detach().cpu().numpy())
        metadata.extend(_metadata_rows(batch["metadata"]))
    if not probabilities:
        raise ValueError("temporal diagnostic received an empty dataset")
    return FactorialPrediction(
        utility_probability=np.concatenate(probabilities),
        utility_logits=np.concatenate(logits),
        success=np.concatenate(success),
        candidate_utility=np.concatenate(utility),
        metadata=metadata,
        mask_probability=np.empty(0, dtype=np.float64),
        mask_target=np.empty(0, dtype=np.float64),
    )


def _score_distribution(scores: np.ndarray, success: np.ndarray) -> dict[str, Any]:
    score = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(success, dtype=np.float64).reshape(-1) >= 0.5

    def describe(values: np.ndarray) -> dict[str, float | int | None]:
        if not values.size:
            return {"n": 0, "mean": None, "std": None, "p05": None, "p25": None, "p50": None, "p75": None, "p95": None}
        return {
            "n": int(values.size), "mean": float(values.mean()), "std": float(values.std(ddof=1 if values.size > 1 else 0)),
            "p05": float(np.quantile(values, 0.05)), "p25": float(np.quantile(values, 0.25)),
            "p50": float(np.quantile(values, 0.50)), "p75": float(np.quantile(values, 0.75)), "p95": float(np.quantile(values, 0.95)),
        }

    positive, negative = score[labels], score[~labels]
    if positive.size and negative.size:
        comparison = positive[:, None] - negative[None, :]
        separation: float | None = float(np.mean((comparison > 0.0) + 0.5 * (comparison == 0.0)))
    else:
        separation = None
    return {"successful": describe(positive), "failed": describe(negative), "success_over_failure_probability": separation}


def _metrics(prediction: FactorialPrediction, scores: np.ndarray, *, score_name: str, calibration: Mapping[str, float] | None) -> dict[str, Any]:
    report = ranking_metrics_tie_aware(
        RankingBundle(scores=np.asarray(scores, dtype=np.float64), success=prediction.success, utility=prediction.candidate_utility, metadata=prediction.metadata),
        score_name=score_name,
    )
    report["score_distribution"] = _score_distribution(np.asarray(scores), prediction.success)
    if calibration is not None:
        report["utility_calibration"] = reliability_metrics(np.asarray(scores), prediction.success, bins=15)
    return report


def _evaluate_view(
    model: Any, *, spec: CheckpointSpec, dataset: Dataset[Any], fair: FactorialPrediction, batch_size: int, view: str
) -> dict[str, Any]:
    learned = predict_temporal_diagnostic(model, dataset, batch_size=batch_size, view=view)
    hybrid, calibrated = _hybrid_scores(learned, fair, spec.calibration)
    return {
        "frozen_hybrid": _metrics(learned, hybrid, score_name="frozen_hybrid_alpha_0.5", calibration=None),
        "learned_only": _metrics(learned, calibrated, score_name="validation_calibrated_learned", calibration=spec.calibration),
        "fair_only": _metrics(fair, fair.utility_probability, score_name="frozen_analytic", calibration=None),
        "checkpoint": str(spec.path),
        "best_epoch": spec.best_epoch,
        "calibration": dict(spec.calibration),
    }


def _paired(changed: Mapping[str, Any], original: Mapping[str, Any], *, seed: int, config: Mapping[str, Any]) -> dict[str, Any]:
    resamples = int(config["statistics"]["bootstrap_resamples"])
    permutations = int(config["statistics"]["permutation_samples"])
    result: dict[str, Any] = {}
    for offset, name in enumerate(("frozen_hybrid", "learned_only", "fair_only")):
        result[name] = paired_ranking_report(
            changed[name], original[name], seed=seed + offset * 1009, resamples=resamples, permutations=permutations
        )
        left = [float(row["tie_fraction"]) for row in changed[name]["per_group"]]
        right = [float(row["tie_fraction"]) for row in original[name]["per_group"]]
        result[name]["tie_fraction"] = paired_summary(
            left, right, seed=seed + offset * 1009 + 809, resamples=resamples, permutations=permutations
        )
    return result


def _summary(rows: Sequence[Mapping[str, Any]], *, score_kind: str) -> dict[str, Any]:
    metrics = (*RANKING_METRICS, "tie_fraction")
    result: dict[str, Any] = {"seeds": [int(row["seed"]) for row in rows]}
    for metric in metrics:
        values = np.asarray([float(row[score_kind][metric]) for row in rows], dtype=np.float64)
        result[metric] = {"mean": float(values.mean()), "std": float(values.std(ddof=1 if values.size > 1 else 0)), "values": values.tolist()}
    if score_kind == "learned_only":
        for metric in ("ece", "brier", "average_precision", "pr_auc"):
            values = np.asarray([float(row[score_kind]["utility_calibration"][metric]) for row in rows], dtype=np.float64)
            result.setdefault("utility_calibration", {})[metric] = {
                "mean": float(values.mean()), "std": float(values.std(ddof=1 if values.size > 1 else 0)), "values": values.tolist(),
            }
    return result


def _view_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {name: _summary(rows, score_kind=name) for name in ("frozen_hybrid", "learned_only", "fair_only")}


def _temporal_view_inputs(
    frozen_dataset: Dataset[Any], *, batch_size: int
) -> dict[str, tuple[TemporalDiagnosticDataset, FactorialPrediction]]:
    """Construct each frozen Phase-4 view once, shared by every fair arm."""

    inputs: dict[str, tuple[TemporalDiagnosticDataset, FactorialPrediction]] = {}
    for view in TEMPORAL_SHORTCUT_VIEWS:
        dataset = TemporalDiagnosticDataset(frozen_dataset, view=view)
        inputs[view] = dataset, _fair_prediction(dataset, batch_size=batch_size)
    if tuple(inputs) != TEMPORAL_SHORTCUT_VIEWS:
        raise AssertionError("Phase 4 must retain its complete fixed 13-view order")
    return inputs


def _phase4_variant_views(
    *,
    models: Mapping[int, Any],
    specs: Mapping[int, CheckpointSpec],
    view_inputs: Mapping[str, tuple[Dataset[Any], FactorialPrediction]],
    batch_size: int,
    config: Mapping[str, Any],
    pairing_seed: int,
) -> dict[str, dict[str, Any]]:
    """Evaluate all 13 frozen views for one complete, provenance-checked arm."""

    if set(models) != set(specs) or not models:
        raise ValueError("each Phase-4 arm needs one model and checkpoint specification per seed")
    if tuple(view_inputs) != TEMPORAL_SHORTCUT_VIEWS:
        raise AssertionError("Phase 4 inputs must contain exactly the fixed 13 views in order")
    views: dict[str, dict[str, Any]] = {}
    correct_by_seed: dict[int, dict[str, Any]] = {}
    for view_offset, view in enumerate(TEMPORAL_SHORTCUT_VIEWS):
        dataset, fair = view_inputs[view]
        rows: list[dict[str, Any]] = []
        for seed in sorted(models):
            evaluation = _evaluate_view(
                models[seed], spec=specs[seed], dataset=dataset, fair=fair, batch_size=batch_size, view=view
            )
            row: dict[str, Any] = {
                "candidate_count": CANDIDATE_COUNT,
                "seed": seed,
                "view": view,
                "feature_mode": _feature_mode(view),
                **evaluation,
            }
            if view == "correct_history":
                correct_by_seed[seed] = row
            else:
                row["paired_view_minus_correct_history"] = _paired(
                    evaluation,
                    correct_by_seed[seed],
                    seed=pairing_seed + view_offset * 10000 + seed,
                    config=config,
                )
            rows.append(row)
        if len(rows) != len(models):
            raise AssertionError(f"Phase 4 lost a seed in view {view}")
        views[view] = {"description": _VIEW_DESCRIPTIONS[view], "per_seed": rows, "summary": _view_summary(rows)}
    if tuple(views) != TEMPORAL_SHORTCUT_VIEWS:
        raise AssertionError("Phase 4 result is missing a required temporal/shortcut view")
    return views


def _phase4_variant_report(
    *, variant: ValidationPromisingVariant, views: Mapping[str, Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    """Serializable complete Phase-4 result for one validation-defined fair arm."""

    if tuple(views) != TEMPORAL_SHORTCUT_VIEWS:
        raise AssertionError("per-variant Phase 4 reports require all 13 views")
    return {
        "variant": variant.report_row(),
        "view_order": list(TEMPORAL_SHORTCUT_VIEWS),
        "views": dict(views),
        "causality_assessment": _causality_assessment(
            views, gate=float(config["gates"]["history_or_action_ranking_drop"])
        ),
    }


def _causality_assessment(views: Mapping[str, Mapping[str, Any]], *, gate: float) -> dict[str, Any]:
    correct = views["correct_history"]["summary"]["learned_only"]
    important = (
        "reversed_history", "independently_permuted_frames", "duplicated_final_frame", "mismatched_history",
        "zeroed_motion", "shuffled_motion", "removed_timestamps", "shuffled_timestamps",
    )
    rows: list[dict[str, Any]] = []
    for view in important:
        current = views[view]["summary"]["learned_only"]
        top1_delta = float(current["top1_success"]["mean"] - correct["top1_success"]["mean"])
        regret_delta = float(current["normalized_regret"]["mean"] - correct["normalized_regret"]["mean"])
        rows.append({
            "view": view,
            "learned_only_top1_delta": top1_delta,
            "learned_only_normalized_regret_delta": regret_delta,
            "meaningful_degradation": bool(top1_delta <= -gate or regret_delta >= gate),
        })
    detected = any(bool(row["meaningful_degradation"]) for row in rows)
    return {
        "gate": float(gate),
        "criterion": "Top-1 drop >= gate OR normalized-regret increase >= gate on a causally important temporal/motion corruption.",
        "views": rows,
        "dynamics_aware_signal_detected": detected,
        "interpretation": "At least one relevant corruption degrades learned-only ranking." if detected else "No relevant corruption reaches the fixed gate; the learned verifier is not demonstrated dynamics-aware.",
    }


def _head_diagnostic(
    *, config: Mapping[str, Any], output_root: Path, selection: Mapping[str, str], clean_dataset: Dataset[Any], fair: FactorialPrediction,
    primary_rows: Sequence[Mapping[str, Any]], primary_head: str, batch_size: int,
) -> dict[str, Any]:
    """Evaluate already-trained head variants only; no listwise head is invented."""

    report: dict[str, Any] = {
        "selection": dict(selection),
        "listwise": "not evaluated: no clean listwise implementation is present in the frozen protocol",
        "heads": {},
    }
    primary_by_seed = {int(row["seed"]): row for row in primary_rows}
    for head in ("success_bce", "pairwise", primary_head):
        if head == primary_head:
            rows = list(primary_rows)
            errors: list[str] = []
        else:
            models, specs, errors = _load_head_models(config=config, output_root=output_root, selection=selection, ranking_head=head)
            rows = []
            if not errors:
                for seed in sorted(models):
                    evaluation = _evaluate_view(
                        models[seed], spec=specs[seed], dataset=clean_dataset, fair=fair, batch_size=batch_size, view="correct_history"
                    )
                    row: dict[str, Any] = {"seed": seed, "view": "correct_history", **evaluation}
                    row["paired_vs_primary_head"] = _paired(
                        evaluation, primary_by_seed[seed], seed=DIAGNOSTIC_SEED + 30000 + seed, config=config
                    )
                    rows.append(row)
        if errors:
            report["heads"][head] = {"status": "unavailable", "errors": errors}
        else:
            report["heads"][head] = {"status": "evaluated_existing_checkpoint", "per_seed": rows, "summary": _view_summary(rows)}
    return report


def _incomplete_report(
    *,
    config: Mapping[str, Any],
    output_root: Path,
    selection: Mapping[str, str] | None,
    errors: Sequence[str],
    all_validation_promising: bool = False,
) -> dict[str, Any]:
    return {
        "milestone": "2E", "status": "not_run_missing_or_incompatible_selected_checkpoints", "cpu_only": True,
        "output_root": str(output_root), "config_digest": _json_digest(config), "selected_variant": dict(selection) if selection else None,
        "all_validation_promising": bool(all_validation_promising),
        "errors": list(errors),
        "scope": (
            "No training or evaluation was started; provide complete validation-primary records and all provenance-compatible "
            "checkpoints for every validation-promising fair arm before running this frozen diagnostic."
            if all_validation_promising
            else "No training or evaluation was started; provide all validation-selected primary checkpoints to run this frozen diagnostic."
        ),
    }


def run(
    config_path: Path = DEFAULT_CONFIG,
    *,
    output_path: Path | None = None,
    output_root: Path = FULL_OUTPUT_DIR,
    groups: int | None = None,
    all_validation_promising: bool = False,
) -> dict[str, Any]:
    """Run the isolated frozen C20 diagnostic, never training or overwriting it.

    ``all_validation_promising`` changes only Phase 4.  It reads the complete
    primary-training records, fixes the eligible fair arms strictly from their
    validation metrics, verifies every selected checkpoint before evaluation,
    and then evaluates all 13 views for every eligible arm.  Phase 5 remains
    primary-only because its alternative ranking-head checkpoints were trained
    only for the validation-selected primary architecture.
    """

    output_root = Path(output_root)
    if output_path is None:
        output_path = DEFAULT_ALL_PROMISING_OUTPUT_PATH if all_validation_promising else DEFAULT_OUTPUT_PATH
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite temporal diagnostic artifact: {output_path}")
    config = _load_config(config_path)
    if str(config["experiment"]["device"]).lower() != "cpu":
        raise ValueError("Milestone 2E temporal diagnostics are CPU-only")
    torch.set_num_threads(int(config["training"]["num_threads"]))
    selection = _selection(output_root)
    if selection is None:
        report = _incomplete_report(
            config=config,
            output_root=output_root,
            selection=None,
            errors=[f"missing {output_root / PRIMARY_SELECTION_FILE}"],
            all_validation_promising=all_validation_promising,
        )
        _write_json(output_path, report)
        return report

    validation_promising: list[ValidationPromisingVariant] = []
    validation_evidence: dict[str, Any] | None = None
    if all_validation_promising:
        try:
            validation_promising, validation_evidence = _validation_promising_variants(
                config=config, output_root=output_root, selected_primary=selection
            )
        except (FileNotFoundError, ValueError, AssertionError) as error:
            report = _incomplete_report(
                config=config,
                output_root=output_root,
                selection=selection,
                errors=[str(error)],
                all_validation_promising=True,
            )
            _write_json(output_path, report)
            return report

    primary_models, primary_specs, errors = _load_head_models(
        config=config, output_root=output_root, selection=selection, ranking_head=PRIMARY_HEAD
    )
    expected_seeds = {int(seed) for seed in config["experiment"]["training_seeds"]}
    if errors or set(primary_models) != expected_seeds:
        report = _incomplete_report(
            config=config,
            output_root=output_root,
            selection=selection,
            errors=errors or ["primary checkpoint seed set is incomplete"],
            all_validation_promising=all_validation_promising,
        )
        _write_json(output_path, report)
        return report

    all_phase4_models: dict[tuple[str, str], tuple[dict[int, Any], dict[int, CheckpointSpec]]] = {
        _variant_key(selection): (primary_models, primary_specs)
    }
    if all_validation_promising:
        load_errors: list[str] = []
        for variant in validation_promising:
            if variant.key == _variant_key(selection):
                continue
            models, specs, arm_errors = _load_head_models(
                config=config,
                output_root=output_root,
                selection={"correspondence": variant.correspondence, "backbone": variant.backbone},
                ranking_head=PRIMARY_HEAD,
            )
            if arm_errors or set(models) != expected_seeds:
                load_errors.extend(
                    f"{variant.token}: {message}"
                    for message in (arm_errors or ["primary checkpoint seed set is incomplete"])
                )
            else:
                all_phase4_models[variant.key] = models, specs
        expected_promising_keys = {variant.key for variant in validation_promising}
        if load_errors or set(all_phase4_models) != expected_promising_keys:
            report = _incomplete_report(
                config=config,
                output_root=output_root,
                selection=selection,
                errors=load_errors or ["validation-promising checkpoint set is incomplete"],
                all_validation_promising=True,
            )
            _write_json(output_path, report)
            return report

    frozen_dataset: Dataset[Any] = build_datasets(config)[f"ranking_test/{CANDIDATE_COUNT}"]
    frozen_groups = len(getattr(frozen_dataset, "group_records", ()))
    if groups is not None and int(groups) != frozen_groups:
        frozen_dataset = _WorldPrefix(frozen_dataset, groups=int(groups))
    if len(getattr(frozen_dataset, "group_records", ())) < 2:
        raise ValueError("temporal diagnostics require at least two complete C20 worlds")
    batch_size = int(config["training"]["evaluation_batch_size"])
    view_inputs = _temporal_view_inputs(frozen_dataset, batch_size=batch_size)
    primary_views = _phase4_variant_views(
        models=primary_models,
        specs=primary_specs,
        view_inputs=view_inputs,
        batch_size=batch_size,
        config=config,
        pairing_seed=DIAGNOSTIC_SEED,
    )
    clean_dataset, clean_fair = view_inputs["correct_history"]
    phase5 = _head_diagnostic(
        config=config, output_root=output_root, selection=selection, clean_dataset=clean_dataset, fair=clean_fair,
        primary_rows=primary_views["correct_history"]["per_seed"], primary_head=PRIMARY_HEAD, batch_size=batch_size,
    )
    phase5.update(
        {
            "scope": {
                "evaluated_variant": dict(selection),
                "primary_only": True,
                "reason": (
                    "Phase 5 compares ranking-head checkpoints only for the validation-selected primary architecture; "
                    "--all-validation-promising expands Phase 4 temporal views only and cannot alter primary selection."
                ),
            },
            "tie_fraction": {
                view: primary_views[view]["summary"]["learned_only"]["tie_fraction"] for view in TEMPORAL_SHORTCUT_VIEWS
            },
            "candidate_template_shortcut_accuracy": {
                "definition": "scene-blind learned-only C20 Top-1 under translation/rotation-normalized candidate template input; no ID, slot, or family metadata is supplied.",
                "metrics": primary_views["candidate_template_only"]["summary"]["learned_only"],
            },
            "action_only_ranking_performance": primary_views["action_only"]["summary"],
            "scene_only_ranking_performance": primary_views["scene_only"]["summary"],
            "score_distribution_and_calibration": {
                "location": "Each per-seed learned_only/frozen_hybrid metric record contains successful-vs-failed score distributions; learned_only records also contain validation-calibration reliability metrics.",
            },
        }
    )
    phase4: dict[str, Any] = {
        "evaluation_mode": "all_validation_promising" if all_validation_promising else "selected_primary_only",
        "view_order": list(TEMPORAL_SHORTCUT_VIEWS),
        "views": primary_views,
        "causality_assessment": _causality_assessment(
            primary_views, gate=float(config["gates"]["history_or_action_ranking_drop"])
        ),
    }
    if all_validation_promising:
        assert validation_evidence is not None
        per_variant: dict[str, dict[str, Any]] = {}
        for offset, variant in enumerate(validation_promising):
            models, specs = all_phase4_models[variant.key]
            variant_views = primary_views if variant.key == _variant_key(selection) else _phase4_variant_views(
                models=models,
                specs=specs,
                view_inputs=view_inputs,
                batch_size=batch_size,
                config=config,
                pairing_seed=DIAGNOSTIC_SEED + (offset + 1) * 1000000,
            )
            per_variant[variant.token] = _phase4_variant_report(variant=variant, views=variant_views, config=config)
        if set(per_variant) != {variant.token for variant in validation_promising}:
            raise AssertionError("all validation-promising fair arms must receive a complete Phase 4 report")
        phase4.update(
            {
                "validation_promising_rule": validation_evidence,
                "per_variant": per_variant,
                "scope": (
                    "Every arm in per_variant passed the fixed validation-only five-seed criterion before frozen test "
                    "evaluation. Each contains all 13 Phase 4 temporal/shortcut views."
                ),
            }
        )
    report = {
        "milestone": "2E", "status": "completed_frozen_c20" if groups is None else "nondecision_complete_world_prefix",
        "cpu_only": True, "candidate_count": CANDIDATE_COUNT, "groups": len(getattr(frozen_dataset, "group_records", ())),
        "frozen_test_groups": frozen_groups, "config_digest": _json_digest(config), "selected_primary": dict(selection),
        "all_validation_promising": bool(all_validation_promising),
        "scope": "Read-only frozen C20 evaluation; no training, checkpoint selection, label change, or GPU use.",
        "phase4_temporal_shortcut": phase4,
        "phase5_ranking_reporting": phase5,
    }
    _write_json(output_path, report)
    return report


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=FULL_OUTPUT_DIR)
    parser.add_argument("--groups", type=int, default=None, help="optional complete C20 prefix; non-decision when supplied")
    parser.add_argument(
        "--all-validation-promising",
        action="store_true",
        help=(
            "evaluate all fair primary-head arms passing the fixed five-seed validation-only Phase-4 criterion; "
            "uses a separate default output path and leaves Phase 5 primary-only"
        ),
    )
    arguments = parser.parse_args(argv)
    report = run(
        output_path=arguments.output,
        output_root=arguments.output_root,
        groups=arguments.groups,
        all_validation_promising=arguments.all_validation_promising,
    )
    print(report["status"])
    return report


if __name__ == "__main__":
    main()
