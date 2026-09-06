"""Versioned NumPy schemas for binding datasets and verifier predictions.

The package intentionally does not import :mod:`actmask`: importing the research
prototype pulls in PyTorch, while a benchmark validator should run with NumPy
alone on a laptop or in an external method repository.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np


DATASET_SCHEMA = "actmask-binding-dataset-v1"
PREDICTION_SCHEMA = "actmask-binding-prediction-v1"
SUITE_SCHEMA = "actmask-binding-intervention-suite-v1"


def _text_scalar(value: np.ndarray, name: str) -> str:
    if value.shape != ():
        raise ValueError(f"{name} must be a scalar")
    return str(value.item())


def _integer_scalar(value: np.ndarray, name: str) -> int:
    if value.shape != () or not np.issubdtype(value.dtype, np.integer):
        raise ValueError(f"{name} must be an integer scalar")
    return int(value.item())


def _string_array(value: np.ndarray, name: str, shape: tuple[int, ...]) -> None:
    if value.shape != shape or value.dtype.kind not in "US":
        raise ValueError(f"{name} must be a string array with shape {shape}")


def _numeric_array(
    value: np.ndarray,
    name: str,
    shape: tuple[int, ...],
    *,
    finite: bool = True,
) -> None:
    if value.shape != shape or not np.issubdtype(value.dtype, np.number):
        raise ValueError(f"{name} must be numeric with shape {shape}")
    if finite and not np.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")


def _canonical_array_digest(arrays: Mapping[str, np.ndarray]) -> str:
    """Hash names, dtypes, shapes, and C-order bytes in stable key order."""

    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(value.dtype.str.encode("ascii") + b"\0")
        digest.update(json.dumps(value.shape).encode("ascii") + b"\0")
        if value.dtype.kind in "US":
            payload = json.dumps(value.tolist(), ensure_ascii=False, separators=(",", ":"))
            digest.update(payload.encode("utf-8"))
        else:
            digest.update(value.tobytes(order="C"))
        digest.update(b"\n")
    return digest.hexdigest()


@dataclass(frozen=True)
class BindingDataset:
    """One split containing matched two-branch twins.

    Shapes use ``T`` twins, exactly two branches, padded history length ``H``,
    ``K`` candidate actions, action dimension ``A``, and effect dimension ``E``.
    Candidate truth is evaluator-only and must never be exposed to a verifier.
    """

    dataset_id: str
    split: str
    history_actions: np.ndarray  # [T, 2, H, A]
    history_effects: np.ndarray  # [T, 2, H, E]
    history_times: np.ndarray  # [T, 2, H]
    history_mask: np.ndarray  # [T, 2, H]
    context: np.ndarray  # [T, 2, C], e.g. current state/goal features
    candidates: np.ndarray  # [T, 2, K, A]
    candidate_ids: np.ndarray  # [T, 2, K]
    true_utility: np.ndarray  # [T, 2, K]
    twin_ids: np.ndarray  # [T]
    episode_ids: np.ndarray  # [T, 2]
    split_group_ids: np.ndarray  # [T], scene/object/mechanism group for leakage audit
    seed_ids: np.ndarray  # [T]
    candidate_effects: np.ndarray | None = None  # [T, 2, K, E]
    schema_version: str = DATASET_SCHEMA

    @property
    def twins(self) -> int:
        return int(self.history_actions.shape[0])

    @property
    def candidates_per_branch(self) -> int:
        return int(self.candidates.shape[2])

    def public_view(self) -> "BindingDataset":
        """Return the exact verifier view with all evaluator-only truth removed."""

        public = replace(
            self,
            true_utility=np.empty((0, 2, 0), dtype=np.float64),
            candidate_effects=None,
            episode_ids=np.full(self.episode_ids.shape, "", dtype=str),
            split_group_ids=np.full(self.split_group_ids.shape, "", dtype=str),
            seed_ids=np.zeros_like(self.seed_ids),
        )
        # The frozen dataclass alone does not protect its NumPy storage.
        arrays = {}
        for field in fields(public):
            value = getattr(public, field.name)
            if isinstance(value, np.ndarray):
                value = value.copy()
                value.flags.writeable = False
                arrays[field.name] = value
        return replace(public, **arrays)


@dataclass(frozen=True)
class PredictionBundle:
    """Scores emitted by one frozen verifier for one intervention case."""

    method_id: str
    dataset_id: str
    split: str
    intervention: str
    scores: np.ndarray  # [T, 2, K], larger is better
    candidate_ids: np.ndarray  # [T, 2, K]
    twin_ids: np.ndarray  # [T]
    dataset_digest: str
    training_data_digests: tuple[str, ...] = ()
    predicted_effects: np.ndarray | None = None  # [T, 2, K, E]
    uncertainty: np.ndarray | None = None  # same shape as predicted_effects
    schema_version: str = PREDICTION_SCHEMA


def dataset_arrays(dataset: BindingDataset) -> dict[str, np.ndarray]:
    arrays = {
        "schema_version": np.asarray(dataset.schema_version),
        "dataset_id": np.asarray(dataset.dataset_id),
        "split": np.asarray(dataset.split),
        "history_actions": np.asarray(dataset.history_actions),
        "history_effects": np.asarray(dataset.history_effects),
        "history_times": np.asarray(dataset.history_times),
        "history_mask": np.asarray(dataset.history_mask, dtype=bool),
        "context": np.asarray(dataset.context),
        "candidates": np.asarray(dataset.candidates),
        "candidate_ids": np.asarray(dataset.candidate_ids),
        "true_utility": np.asarray(dataset.true_utility),
        "twin_ids": np.asarray(dataset.twin_ids),
        "episode_ids": np.asarray(dataset.episode_ids),
        "split_group_ids": np.asarray(dataset.split_group_ids),
        "seed_ids": np.asarray(dataset.seed_ids, dtype=np.int64),
    }
    if dataset.candidate_effects is not None:
        arrays["candidate_effects"] = np.asarray(dataset.candidate_effects)
    return arrays


def prediction_arrays(prediction: PredictionBundle) -> dict[str, np.ndarray]:
    arrays = {
        "schema_version": np.asarray(prediction.schema_version),
        "method_id": np.asarray(prediction.method_id),
        "dataset_id": np.asarray(prediction.dataset_id),
        "split": np.asarray(prediction.split),
        "intervention": np.asarray(prediction.intervention),
        "scores": np.asarray(prediction.scores),
        "candidate_ids": np.asarray(prediction.candidate_ids),
        "twin_ids": np.asarray(prediction.twin_ids),
        "dataset_digest": np.asarray(prediction.dataset_digest),
        "training_data_digests": np.asarray(prediction.training_data_digests, dtype=str),
    }
    if prediction.predicted_effects is not None:
        arrays["predicted_effects"] = np.asarray(prediction.predicted_effects)
    if prediction.uncertainty is not None:
        arrays["uncertainty"] = np.asarray(prediction.uncertainty)
    return arrays


def dataset_digest(dataset: BindingDataset) -> str:
    return _canonical_array_digest(dataset_arrays(dataset))


def validate_dataset(dataset: BindingDataset, *, require_truth: bool = True) -> dict[str, Any]:
    if dataset.schema_version != DATASET_SCHEMA:
        raise ValueError(f"unsupported dataset schema: {dataset.schema_version}")
    if not dataset.dataset_id or not dataset.split:
        raise ValueError("dataset_id and split must be nonempty")
    actions = np.asarray(dataset.history_actions)
    if actions.ndim != 4 or actions.shape[1] != 2:
        raise ValueError("history_actions must have shape [T,2,H,A]")
    twins, branches, history, action_dim = actions.shape
    if twins < 1 or history < 1 or action_dim < 1:
        raise ValueError("dataset dimensions must be positive")
    effects = np.asarray(dataset.history_effects)
    if effects.ndim != 4 or effects.shape[:3] != (twins, branches, history):
        raise ValueError("history_effects must have shape [T,2,H,E]")
    effect_dim = effects.shape[3]
    if effect_dim < 1:
        raise ValueError("effect dimension must be positive")
    context = np.asarray(dataset.context)
    if context.ndim != 3 or context.shape[:2] != (twins, branches) or context.shape[2] < 1:
        raise ValueError("context must have shape [T,2,C] with C > 0")
    if np.asarray(dataset.candidates).ndim != 4:
        raise ValueError("candidates must have shape [T,2,K,A]")
    candidates = np.asarray(dataset.candidates)
    if candidates.shape[:2] != (twins, branches) or candidates.shape[3] != action_dim:
        raise ValueError("candidate dimensions do not match history actions")
    candidate_count = candidates.shape[2]
    if candidate_count < 2:
        raise ValueError("each branch needs at least two candidates")
    _numeric_array(actions, "history_actions", (twins, 2, history, action_dim))
    _numeric_array(effects, "history_effects", (twins, 2, history, effect_dim))
    _numeric_array(np.asarray(dataset.history_times), "history_times", (twins, 2, history))
    _numeric_array(context, "context", context.shape)
    mask = np.asarray(dataset.history_mask)
    if mask.shape != (twins, 2, history) or mask.dtype != np.bool_:
        raise ValueError("history_mask must be boolean with shape [T,2,H]")
    if np.any(mask.sum(axis=2) < 2):
        raise ValueError("every branch needs at least two valid history pairs")
    for name, value in (
        ("history_actions", actions),
        ("history_effects", effects),
        ("history_times", np.asarray(dataset.history_times)),
    ):
        expanded_mask = mask[..., None] if value.ndim == 4 else mask
        if not np.isfinite(value[expanded_mask.repeat(value.shape[-1], axis=-1) if value.ndim == 4 else expanded_mask]).all():
            raise ValueError(f"{name} contains non-finite values in valid positions")
    _numeric_array(candidates, "candidates", (twins, 2, candidate_count, action_dim))
    _string_array(np.asarray(dataset.candidate_ids), "candidate_ids", (twins, 2, candidate_count))
    _string_array(np.asarray(dataset.twin_ids), "twin_ids", (twins,))
    _string_array(np.asarray(dataset.episode_ids), "episode_ids", (twins, 2))
    _string_array(np.asarray(dataset.split_group_ids), "split_group_ids", (twins,))
    seed_ids = np.asarray(dataset.seed_ids)
    if seed_ids.shape != (twins,) or not np.issubdtype(seed_ids.dtype, np.integer):
        raise ValueError("seed_ids must be an integer array with shape [T]")
    if len(set(dataset.twin_ids.tolist())) != twins:
        raise ValueError("twin_ids must be unique")
    if any(not str(value) for value in dataset.twin_ids.tolist()):
        raise ValueError("twin_ids must be nonempty")
    if not np.array_equal(context[:, 0], context[:, 1]):
        raise ValueError("matched branches must have byte-identical public context")
    if not np.array_equal(mask.sum(axis=2)[:, 0], mask.sum(axis=2)[:, 1]):
        raise ValueError("matched branches must have equal history lengths")
    for twin in range(twins):
        valid_by_branch = [np.flatnonzero(mask[twin, branch]) for branch in range(2)]
        action_time = []
        for branch in range(2):
            ids = dataset.candidate_ids[twin, branch].tolist()
            if len(set(ids)) != candidate_count:
                raise ValueError("candidate_ids must be unique within every branch")
            if any(not str(value) for value in ids):
                raise ValueError("candidate_ids must be nonempty")
            valid = valid_by_branch[branch]
            tokens = np.concatenate(
                (actions[twin, branch, valid], dataset.history_times[twin, branch, valid, None]),
                axis=1,
            )
            action_time.append(tokens[np.lexsort(tokens.T[::-1])])
        if not np.array_equal(action_time[0], action_time[1]):
            raise ValueError("matched branches must have the same action-time history marginal")
        if set(dataset.candidate_ids[twin, 0].tolist()) != set(
            dataset.candidate_ids[twin, 1].tolist()
        ):
            raise ValueError("matched branches must expose the same candidate ID set")
        branch_one_index = {
            str(candidate_id): slot
            for slot, candidate_id in enumerate(dataset.candidate_ids[twin, 1])
        }
        branch_one_aligned = dataset.candidates[
            twin,
            1,
            [branch_one_index[str(candidate_id)] for candidate_id in dataset.candidate_ids[twin, 0]],
        ]
        if not np.array_equal(dataset.candidates[twin, 0], branch_one_aligned):
            raise ValueError("matched branches must have identical candidate actions by ID")
    if require_truth:
        utility = np.asarray(dataset.true_utility)
        _numeric_array(utility, "true_utility", (twins, 2, candidate_count))
        if np.any(np.ptp(utility, axis=2) <= 1.0e-12):
            raise ValueError("every branch must have nonzero candidate utility span")
    if dataset.candidate_effects is not None:
        _numeric_array(
            np.asarray(dataset.candidate_effects),
            "candidate_effects",
            (twins, 2, candidate_count, effect_dim),
        )
    return {
        "passed": True,
        "dataset_id": dataset.dataset_id,
        "split": dataset.split,
        "twins": twins,
        "seeds": int(len(np.unique(seed_ids))),
        "history_length_min": int(mask.sum(axis=2).min()),
        "history_length_max": int(mask.sum(axis=2).max()),
        "candidates_per_branch": candidate_count,
        "action_dim": action_dim,
        "effect_dim": effect_dim,
        "context_dim": int(context.shape[2]),
        "dataset_digest": dataset_digest(dataset),
    }


def validate_prediction(prediction: PredictionBundle, dataset: BindingDataset) -> dict[str, Any]:
    if prediction.schema_version != PREDICTION_SCHEMA:
        raise ValueError(f"unsupported prediction schema: {prediction.schema_version}")
    validate_dataset(dataset)
    shape = dataset.true_utility.shape
    _numeric_array(np.asarray(prediction.scores), "scores", shape)
    _string_array(np.asarray(prediction.candidate_ids), "candidate_ids", shape)
    _string_array(np.asarray(prediction.twin_ids), "twin_ids", (dataset.twins,))
    if prediction.dataset_id != dataset.dataset_id or prediction.split != dataset.split:
        raise ValueError("prediction dataset_id/split does not match dataset")
    if not np.array_equal(prediction.twin_ids, dataset.twin_ids):
        raise ValueError("prediction twin_ids do not match dataset order")
    if prediction.dataset_digest != dataset_digest(dataset):
        raise ValueError("prediction was not generated for this exact dataset digest")
    if prediction.dataset_digest in prediction.training_data_digests:
        raise ValueError("evaluation dataset digest appears in training_data_digests")
    if prediction.predicted_effects is not None:
        if dataset.candidate_effects is None:
            raise ValueError("predicted_effects require candidate-effect ground truth")
        _numeric_array(
            np.asarray(prediction.predicted_effects),
            "predicted_effects",
            dataset.candidate_effects.shape,
        )
    if prediction.uncertainty is not None:
        if prediction.predicted_effects is None:
            raise ValueError("uncertainty requires predicted_effects")
        uncertainty = np.asarray(prediction.uncertainty)
        _numeric_array(uncertainty, "uncertainty", prediction.predicted_effects.shape)
        if np.any(uncertainty < 0):
            raise ValueError("uncertainty must be nonnegative")
    for twin in range(dataset.twins):
        for branch in range(2):
            if set(prediction.candidate_ids[twin, branch].tolist()) != set(
                dataset.candidate_ids[twin, branch].tolist()
            ):
                raise ValueError("prediction candidate IDs do not match the dataset")
    return {"passed": True, "method_id": prediction.method_id, "intervention": prediction.intervention}


def save_dataset(path: str | Path, dataset: BindingDataset) -> dict[str, Any]:
    audit = validate_dataset(dataset)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **dataset_arrays(dataset))
    return audit


def load_dataset(path: str | Path) -> BindingDataset:
    with np.load(Path(path), allow_pickle=False) as raw:
        names = set(raw.files)
        required = {
            "schema_version", "dataset_id", "split", "history_actions",
            "history_effects", "history_times", "history_mask", "context", "candidates",
            "candidate_ids", "true_utility", "twin_ids", "episode_ids",
            "split_group_ids", "seed_ids",
        }
        missing = sorted(required - names)
        if missing:
            raise ValueError(f"dataset file misses arrays: {missing}")
        dataset = BindingDataset(
            schema_version=_text_scalar(raw["schema_version"], "schema_version"),
            dataset_id=_text_scalar(raw["dataset_id"], "dataset_id"),
            split=_text_scalar(raw["split"], "split"),
            history_actions=raw["history_actions"].copy(),
            history_effects=raw["history_effects"].copy(),
            history_times=raw["history_times"].copy(),
            history_mask=raw["history_mask"].astype(bool, copy=True),
            context=raw["context"].copy(),
            candidates=raw["candidates"].copy(),
            candidate_ids=raw["candidate_ids"].copy(),
            true_utility=raw["true_utility"].copy(),
            twin_ids=raw["twin_ids"].copy(),
            episode_ids=raw["episode_ids"].copy(),
            split_group_ids=raw["split_group_ids"].copy(),
            seed_ids=raw["seed_ids"].astype(np.int64, copy=True),
            candidate_effects=(raw["candidate_effects"].copy() if "candidate_effects" in names else None),
        )
    validate_dataset(dataset)
    return dataset


def save_prediction(path: str | Path, prediction: PredictionBundle, dataset: BindingDataset) -> None:
    validate_prediction(prediction, dataset)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **prediction_arrays(prediction))


def load_prediction(path: str | Path, dataset: BindingDataset) -> PredictionBundle:
    with np.load(Path(path), allow_pickle=False) as raw:
        names = set(raw.files)
        required = {
            "schema_version", "method_id", "dataset_id", "split", "intervention",
            "scores", "candidate_ids", "twin_ids", "dataset_digest",
            "training_data_digests",
        }
        missing = sorted(required - names)
        if missing:
            raise ValueError(f"prediction file misses arrays: {missing}")
        prediction = PredictionBundle(
            schema_version=_text_scalar(raw["schema_version"], "schema_version"),
            method_id=_text_scalar(raw["method_id"], "method_id"),
            dataset_id=_text_scalar(raw["dataset_id"], "dataset_id"),
            split=_text_scalar(raw["split"], "split"),
            intervention=_text_scalar(raw["intervention"], "intervention"),
            scores=raw["scores"].copy(),
            candidate_ids=raw["candidate_ids"].copy(),
            twin_ids=raw["twin_ids"].copy(),
            dataset_digest=_text_scalar(raw["dataset_digest"], "dataset_digest"),
            training_data_digests=tuple(str(value) for value in raw["training_data_digests"].tolist()),
            predicted_effects=(raw["predicted_effects"].copy() if "predicted_effects" in names else None),
            uncertainty=(raw["uncertainty"].copy() if "uncertainty" in names else None),
        )
    validate_prediction(prediction, dataset)
    return prediction
