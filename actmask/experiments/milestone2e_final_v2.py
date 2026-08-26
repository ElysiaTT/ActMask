"""Versioned corrected final evaluation for Milestone 2E.

This runner is intentionally *not* a continuation of ``milestone2e_full``.
The legacy runner's incremental ranking files predate the corrected standard
NDCG schema, so they are immutable historical evidence rather than inputs to
this evaluation.  Here we load only frozen, validation-selected checkpoints,
regenerate their predictions, calculate corrected ranking metrics, and write
new artifacts exclusively under ``milestone2e_final_v2``.

The full command is CPU-only by construction.  A CUDA-capable host is not a
reason to move this frozen protocol to a different numerical backend: that
would require a separately documented equivalence experiment.  This module
therefore never selects a CUDA device or changes another process to obtain GPU
resources.

``--verify-inputs`` and ``--smoke`` are read-only preflight modes.  The normal
mode performs Stage A2/A3/A5, but deliberately does not run temporal or
completion-extension phases; the final Stage-A orchestrator invokes those
separate modules after this versioned ranking evidence exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from actmask.data.milestone2b_dataset import MILESTONE2B_OOD_AXES
from actmask.data.milestone2c_dataset import HardCandidateActionDataset, RigidTransformDataset
from actmask.data.milestone2e_final_v2_corruptions import GroupSharedRankingCorruptionDataset
from actmask.eval.calibration import reliability_metrics
from actmask.eval.statistics import paired_bootstrap_ci
from actmask.experiments.milestone2c import FAIR_BASELINES_2C, PROJECT_ROOT, predict_baseline
from actmask.experiments.milestone2e import (
    DEFAULT_CONFIG,
    NDCG_DEFINITION,
    RANKING_METRIC_SCHEMA,
    RANKING_METRICS,
    RankingBundle,
    _load_config,
    ranking_metrics_tie_aware,
    validate_checkpoint_metadata,
)
from actmask.experiments.milestone2e_completion_extensions import PointPermutationDataset
from actmask.experiments.milestone2e_final_cache import (
    CORRECTED_NDCG_METRIC_SCHEMA_VERSION,
    FINAL_EVALUATION_SCHEMA_VERSION,
    FairBaselineCache,
    FairBaselineCacheKey,
    FairBaselineWorld,
    canonical_hash,
    canonical_json,
    code_provenance_for_files,
    worlds_from_metadata,
)
from actmask.experiments.milestone2e_full import (
    PRIMARY_HEAD,
    DiagnosticIdentityDataset,
    FactorialPrediction,
    _hybrid_scores,
    _json_digest,
    _model,
    build_datasets,
    predict_factorial,
)


LEGACY_ROOT = Path(PROJECT_ROOT) / "outputs" / "actmask" / "milestone2e"
FINAL_ROOT = Path(PROJECT_ROOT) / "outputs" / "actmask" / "milestone2e_final_v2"
FROZEN_2C_ROOT = Path(PROJECT_ROOT) / "outputs" / "actmask" / "milestone2c"

ARTIFACT_SCHEMA_VERSION = "milestone2e-final-v2-ranking-v1"
RUN_MANIFEST_SCHEMA_VERSION = "milestone2e-final-v2-run-manifest-v1"
INPUT_MANIFEST_SCHEMA_VERSION = "milestone2e-final-v2-frozen-inputs-v1"
FAIR_RANKING_SELECTION_SCHEMA_VERSION = "milestone2e-final-v2-fair-ranking-selection-v1"
EVALUATION_SCHEMA_VERSION = FINAL_EVALUATION_SCHEMA_VERSION
TIE_POLICY = "uniform-within-equal-score-block; candidate_id never breaks a tie"
SCORE_NAMES = ("frozen_hybrid", "learned_only", "fair_only")
A3_PAIRED_METRICS = (
    "top1_success",
    "top3_success",
    "ranking_ap",
    "ranking_roc_auc",
    "mrr",
    "ndcg",
    "normalized_regret",
)
INVARIANCE_VIEWS = (
    "original",
    "random_so3",
    "axis_permutation",
    "sign_flip",
    "group_shared_all_frame_point_permutation",
    "group_shared_per_frame_resampling",
    "group_shared_full_part_occlusion",
)
FIXED_CANDIDATE_COUNTS = (5, 10, 20, 50)
FAIR_RANKING_SELECTION_FILE = "fair_ranking_baseline_selection.json"

# This mixture is predeclared in the Stage-A final protocol.  It is not
# selected after inspecting any ID/OOD result.  It retains the canonical
# successful templates and adds extended temporal/spatial near misses.
CHANGED_CANDIDATE_TEMPLATE_INDICES = (
    0,
    1,
    2,
    3,
    4,
    8,
    10,
    13,
    17,
    18,
    21,
    22,
    24,
    25,
    26,
    27,
    29,
    30,
    31,
    32,
)
CHANGED_CANDIDATE_DISTRIBUTION_ID = "v2_predeclared_template_mixture_20"
CHANGED_CANDIDATE_MASTER_SEED = 20268119
CHANGED_CANDIDATE_GROUPS = 250


@dataclass(frozen=True)
class CheckpointProvenance:
    """Frozen checkpoint identity recorded in every final-v2 artifact."""

    correspondence: str
    backbone: str
    ranking_head: str
    seed: int
    path: str
    sha256: str
    best_epoch: int
    calibration: dict[str, float]
    config_digest: str


@dataclass(frozen=True)
class FairInput:
    """One view's cached fair prediction and the evidence that keys it."""

    prediction: FactorialPrediction
    worlds: tuple[FairBaselineWorld, ...]
    candidate_provenance: dict[str, Any]
    cache_evidence: dict[str, Any]


def _timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable(asdict(value))
    raise TypeError(f"cannot serialize {type(value).__name__} to a final-v2 artifact")


def _read_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _validate_existing_phase_artifact(
    artifact: Mapping[str, Any],
    *,
    stage: str,
    config: Mapping[str, Any],
    selection: Mapping[str, str],
    fair_ranking_selection: Mapping[str, Any],
    code_provenance: Mapping[str, Any],
) -> None:
    """Fail closed before reusing a partially completed final-v2 phase."""

    if artifact.get("artifact_schema") != ARTIFACT_SCHEMA_VERSION or artifact.get("stage") != stage:
        raise ValueError("existing final-v2 phase artifact has an incompatible schema or stage")
    if artifact.get("config_digest") != _json_digest(config):
        raise ValueError("existing final-v2 phase artifact has a stale config digest")
    if canonical_json(artifact.get("source_code_provenance")) != canonical_json(code_provenance):
        raise ValueError("existing final-v2 phase artifact has stale source-code provenance")
    if canonical_json(artifact.get("validation_selected_arm")) != canonical_json(selection):
        raise ValueError("existing final-v2 phase artifact disagrees with validation primary selection")
    baseline = artifact.get("frozen_fair_ranking_baseline") or artifact.get(
        "frozen_strongest_fair_non_oracle_ranking_baseline"
    )
    if not isinstance(baseline, Mapping) or baseline.get("baseline_name") != fair_ranking_selection["selected"]["baseline_name"]:
        raise ValueError("existing final-v2 phase artifact uses a different fair ranking comparator")


def _write_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a final-v2 artifact once, avoiding accidental result mutation."""

    if path.exists():
        raise FileExistsError(f"refusing to overwrite final-v2 artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(_jsonable(dict(payload)), indent=2, sort_keys=True) + "\n")
    try:
        os.link(temporary, path)
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite final-v2 artifact: {path}") from error
    finally:
        if temporary.exists():
            temporary.unlink()


def _ensure_immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Create a deterministic manifest or prove an existing copy matches."""

    expected = _jsonable(dict(payload))
    if path.exists():
        existing = _read_mapping(path)
        if canonical_json(existing) != canonical_json(expected):
            raise ValueError(f"existing final-v2 manifest disagrees with current frozen inputs: {path}")
        return
    _write_json_new(path, expected)


def _sha256_file(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checkpoint_path(
    *, legacy_root: Path, correspondence: str, backbone: str, seed: int
) -> Path:
    return legacy_root / "checkpoints" / f"{correspondence}__{backbone}__{PRIMARY_HEAD}__seed{seed}.pt"


def _selected_primary(config: Mapping[str, Any], *, legacy_root: Path) -> dict[str, str]:
    """Verify validation-only selection without consulting test ranking rows."""

    selection = _read_mapping(legacy_root / "validation_primary_selection.json")
    correspondence = selection.get("correspondence")
    backbone = selection.get("backbone")
    if not isinstance(correspondence, str) or not isinstance(backbone, str):
        raise ValueError("validation_primary_selection.json lacks arm identity")
    if correspondence == "GroundTruthIdentityCorrespondence":
        raise ValueError("the diagnostic oracle cannot be a fair final primary")
    if correspondence not in set(str(item) for item in config["correspondence"]["variants"]):
        raise ValueError("selected correspondence is absent from frozen config")
    if backbone not in set(str(item) for item in config["backbone"]["variants"]):
        raise ValueError("selected backbone is absent from frozen config")

    records_path = legacy_root / "training_records_primary.json"
    records_value = json.loads(records_path.read_text())
    if not isinstance(records_value, list):
        raise ValueError("training_records_primary.json must be a list")
    expected_seeds = tuple(int(item) for item in config["experiment"]["training_seeds"])
    grouped: dict[tuple[str, str], dict[int, Mapping[str, Any]]] = defaultdict(dict)
    for record in records_value:
        if not isinstance(record, Mapping):
            raise ValueError("primary training record is not an object")
        if str(record.get("ranking_head")) != PRIMARY_HEAD:
            continue
        current_correspondence = str(record.get("correspondence"))
        if current_correspondence == "GroundTruthIdentityCorrespondence":
            continue
        current_backbone = str(record.get("backbone"))
        seed = int(record.get("seed"))
        if seed not in expected_seeds:
            raise ValueError("primary training record has an unexpected seed")
        if seed in grouped[(current_correspondence, current_backbone)]:
            raise ValueError("primary training records duplicate a seed")
        try:
            regret = float(record["validation_normalized_regret"])
            top1 = float(record["validation_top1"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("primary training record lacks validation selection metrics") from error
        if not np.isfinite(regret) or not np.isfinite(top1):
            raise ValueError("primary training record has non-finite validation metrics")
        grouped[(current_correspondence, current_backbone)][seed] = record
    complete: list[tuple[tuple[str, str], float, float]] = []
    for arm, rows in grouped.items():
        if set(rows) == set(expected_seeds):
            complete.append(
                (
                    arm,
                    float(np.mean([float(rows[seed]["validation_normalized_regret"]) for seed in expected_seeds])),
                    float(np.mean([float(rows[seed]["validation_top1"]) for seed in expected_seeds])),
                )
            )
    if not complete:
        raise ValueError("no complete five-seed fair validation arms are available")
    best_arm, _, _ = min(complete, key=lambda row: (row[1], -row[2], row[0][0], row[0][1]))
    if best_arm != (correspondence, backbone):
        raise ValueError(
            "validation selection does not agree with complete validation-only primary records: "
            f"selected {(correspondence, backbone)}, derived {best_arm}"
        )
    return {
        "correspondence": correspondence,
        "backbone": backbone,
        "ranking_head": PRIMARY_HEAD,
        "selection_rule": str(selection.get("selection", "validation-only")),
        "source": str(legacy_root / "validation_primary_selection.json"),
        "training_records_sha256": _sha256_file(records_path),
    }


def _legacy_fair_baseline_selection_reference() -> dict[str, Any]:
    """Record, but do not misuse, the old mask-AP baseline selection.

    Milestone 2C chose this artifact by mask AP.  Stage A's comparator is a
    *ranking* comparator, so it is selected afresh on frozen C20 validation
    ranking worlds below.  Keeping this reference makes that scientific
    distinction explicit rather than silently treating an AP winner as a
    ranking winner.
    """

    path = FROZEN_2C_ROOT / "fair_baseline_selection.json"
    selection = _read_mapping(path).get("selected")
    if not isinstance(selection, Mapping):
        raise ValueError("Milestone 2C fair baseline selection is missing")
    return {
        "method": str(selection.get("method")),
        "selection_split": str(_read_mapping(path).get("selection_split", "validation")),
        "selection_metric": "Milestone 2C mask AP (not reused as a Stage-A ranking selection)",
        "source": str(path),
        "source_sha256": _sha256_file(path),
        "selected": _jsonable(selection),
    }


def _fair_prediction_for_baseline(
    baseline_name: str, dataset: Dataset[Any], *, batch_size: int
) -> FactorialPrediction:
    """Evaluate one parameter-free observable baseline for ranking only."""

    constructor = FAIR_BASELINES_2C.get(baseline_name)
    if constructor is None or baseline_name == "GeometryWithLearnedResidual":
        raise ValueError(f"not a parameter-free fair baseline: {baseline_name}")
    baseline = constructor().cpu().eval()
    prediction = predict_baseline(baseline, dataset, batch_size=batch_size)
    utility = np.asarray(prediction.utility, dtype=np.float64)
    bounded = np.clip(utility, 1.0e-5, 1.0 - 1.0e-5)
    return FactorialPrediction(
        utility_probability=utility,
        utility_logits=np.log(bounded / (1.0 - bounded)),
        success=np.asarray(prediction.success, dtype=np.float64),
        candidate_utility=np.asarray(prediction.candidate_utility, dtype=np.float64),
        metadata=[dict(row) for row in prediction.metadata],
        mask_probability=np.empty(0, dtype=np.float64),
        mask_target=np.empty(0, dtype=np.float64),
    )


def _parameter_free_fair_baseline_names() -> tuple[str, ...]:
    names = tuple(sorted(name for name in FAIR_BASELINES_2C if name != "GeometryWithLearnedResidual"))
    if not names:
        raise ValueError("no parameter-free fair baselines are registered")
    return names


def _load_final_fair_ranking_selection(
    path: Path, *, config: Mapping[str, Any], code_provenance: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Validate a final-v2 validation-only ranking-comparator selection."""

    report = _read_mapping(path)
    if report.get("artifact_schema") != FAIR_RANKING_SELECTION_SCHEMA_VERSION:
        raise ValueError(f"incompatible final-v2 fair ranking selection: {path}")
    if report.get("selection_split") != "ranking_val/C20":
        raise ValueError("final-v2 fair comparator was not selected on frozen C20 validation worlds")
    if report.get("config_digest") != _json_digest(config):
        raise ValueError("final-v2 fair comparator selection has a stale config digest")
    if code_provenance is not None and canonical_json(report.get("source_code_provenance")) != canonical_json(code_provenance):
        raise ValueError("final-v2 fair comparator selection has stale source-code provenance")
    selected = report.get("selected")
    if not isinstance(selected, Mapping) or not isinstance(selected.get("baseline_name"), str):
        raise ValueError("final-v2 fair comparator selection lacks selected baseline identity")
    name = str(selected["baseline_name"])
    if name not in _parameter_free_fair_baseline_names():
        raise ValueError("final-v2 selected comparator is not a registered parameter-free baseline")
    return report


def _select_fair_ranking_baseline(
    *,
    config: Mapping[str, Any],
    validation_dataset: Dataset[Any],
    final_root: Path,
    cache: FairBaselineCache,
    code_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Freeze the strongest fair ranking comparator using validation only.

    This must run before any ID/OOD/test condition.  It intentionally excludes
    ``GeometryWithLearnedResidual`` because it is learned, and ranks all
    parameter-free observable baselines by preregistered validation C20
    normalized regret, then Top-1, then method name.
    """

    path = final_root / FAIR_RANKING_SELECTION_FILE
    if path.exists():
        return _load_final_fair_ranking_selection(
            path, config=config, code_provenance=code_provenance
        )
    rows: list[dict[str, Any]] = []
    for index, baseline_name in enumerate(_parameter_free_fair_baseline_names()):
        fair_input = _get_fair_input(
            dataset=validation_dataset,
            candidate_count=20,
            split_id="ranking_val/C20",
            observation_view_id="original",
            corruption_id="none",
            condition_parameters={
                "stage": "A5_pretest_fair_ranking_comparator_selection",
                "selection_metric": "normalized_regret_then_top1",
                "baseline_candidate": baseline_name,
            },
            baseline_name=baseline_name,
            config=config,
            cache=cache,
            code_provenance=code_provenance,
            batch_size=int(config["training"]["evaluation_batch_size"]),
        )
        metrics = _metric_bundle(
            fair_input.prediction,
            fair_input.prediction.utility_probability,
            score_name=f"validation_fair_{baseline_name}",
        )
        rows.append(
            {
                "baseline_name": baseline_name,
                "learned": False,
                "validation_top1_success": float(metrics["top1_success"]),
                "validation_normalized_regret": float(metrics["normalized_regret"]),
                "validation_ranking_ap": float(metrics["ranking_ap"]),
                "validation_ranking_roc_auc": float(metrics["ranking_roc_auc"]),
                "validation_corrected_ndcg": float(metrics["ndcg"]),
                "scene_support_count": len(fair_input.worlds),
                "candidate_set_provenance": fair_input.candidate_provenance,
                "fair_baseline_cache": fair_input.cache_evidence,
                "selection_order": index,
            }
        )
    selected = min(
        rows,
        key=lambda row: (
            float(row["validation_normalized_regret"]),
            -float(row["validation_top1_success"]),
            str(row["baseline_name"]),
        ),
    )
    report = {
        "artifact_schema": FAIR_RANKING_SELECTION_SCHEMA_VERSION,
        "stage": "A5_pretest_fair_ranking_comparator_selection",
        "selection_split": "ranking_val/C20",
        "selection_rule": "minimum validation normalized regret, then maximum validation Top-1, then baseline name",
        "predeclared_before_id_ood_test_evaluation": True,
        "config_digest": _json_digest(config),
        "source_code_provenance": _jsonable(code_provenance),
        "metric_definition_version": RANKING_METRIC_SCHEMA,
        "ndcg_definition": NDCG_DEFINITION,
        "candidate_baselines": rows,
        "selected": dict(selected),
        "legacy_mask_ap_selection_reference": _legacy_fair_baseline_selection_reference(),
        "exclusions": {
            "GeometryWithLearnedResidual": "learned residual baseline; not a parameter-free fair comparator",
        },
    }
    _write_json_new(path, report)
    return report


def _load_primary_models(
    config: Mapping[str, Any], *, legacy_root: Path, selection: Mapping[str, str]
) -> tuple[dict[int, Any], dict[int, CheckpointProvenance]]:
    """Load exactly the five existing chosen CPU checkpoints; never train."""

    config_digest = _json_digest(config)
    models: dict[int, Any] = {}
    specifications: dict[int, CheckpointProvenance] = {}
    for raw_seed in config["experiment"]["training_seeds"]:
        seed = int(raw_seed)
        path = _checkpoint_path(
            legacy_root=legacy_root,
            correspondence=str(selection["correspondence"]),
            backbone=str(selection["backbone"]),
            seed=seed,
        )
        if not path.exists():
            raise FileNotFoundError(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, Mapping):
            raise ValueError(f"checkpoint is not a mapping: {path}")
        validate_checkpoint_metadata(
            checkpoint,
            {
                "architecture": "FactorialTemporalActMask",
                "correspondence": str(selection["correspondence"]),
                "backbone": str(selection["backbone"]),
                "ranking_head": PRIMARY_HEAD,
                "seed": seed,
                "config_digest": config_digest,
            },
        )
        calibration = checkpoint.get("calibration")
        if not isinstance(calibration, Mapping) or not {"coefficient", "intercept"}.issubset(calibration):
            raise ValueError(f"checkpoint lacks frozen validation calibration: {path}")
        model = _model(config, str(selection["correspondence"]), str(selection["backbone"])).cpu()
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        if any(parameter.device.type != "cpu" for parameter in model.parameters()):
            raise RuntimeError(f"model is not CPU-resident after loading: {path}")
        models[seed] = model
        specifications[seed] = CheckpointProvenance(
            correspondence=str(selection["correspondence"]),
            backbone=str(selection["backbone"]),
            ranking_head=PRIMARY_HEAD,
            seed=seed,
            path=str(path),
            sha256=_sha256_file(path),
            best_epoch=int(checkpoint.get("best_epoch", 0)),
            calibration={
                "coefficient": float(calibration["coefficient"]),
                "intercept": float(calibration["intercept"]),
            },
            config_digest=config_digest,
        )
    if len(models) != 5:
        raise ValueError("final v2 requires exactly five frozen primary checkpoints")
    return models, specifications


def _dataset_kwargs(config: Mapping[str, Any]) -> dict[str, Any]:
    data = config["dataset"]
    return {
        "num_points": int(data["num_points"]),
        "observation_frames": int(data["observation_frames"]),
        "future_steps": int(data["future_steps"]),
        "observation_noise_std": float(data["observation_noise_std"]),
        "point_dropout_rate": float(data["point_dropout_rate"]),
        "occlusion_rate": float(data["occlusion_rate"]),
    }


def _changed_candidate_distribution_dataset(config: Mapping[str, Any]) -> Dataset[Any]:
    """Build the predeclared C20 candidate-template-mixture OOD condition."""

    dataset = HardCandidateActionDataset(
        candidate_count=20,
        split="test",
        domain="id",
        master_seed=CHANGED_CANDIDATE_MASTER_SEED,
        split_counts={"train": 1, "val": 1, "test": CHANGED_CANDIDATE_GROUPS},
        ood_groups_per_axis=1,
        template_indices=CHANGED_CANDIDATE_TEMPLATE_INDICES,
        candidate_distribution_id=CHANGED_CANDIDATE_DISTRIBUTION_ID,
        **_dataset_kwargs(config),
    )
    return DiagnosticIdentityDataset(dataset)


def build_final_v2_datasets(config: Mapping[str, Any]) -> dict[str, Dataset[Any]]:
    """Frozen ID/OOD worlds plus the documented fresh mixture-shift worlds."""

    datasets = dict(build_datasets(config))
    shifted = _changed_candidate_distribution_dataset(config)
    occupied: set[int] = set()
    for dataset in datasets.values():
        occupied.update(int(record.group_id) for record in getattr(dataset, "group_records", ()))
    shifted_ids = {int(record.group_id) for record in getattr(shifted, "group_records", ())}
    if not shifted_ids or occupied.intersection(shifted_ids):
        raise ValueError("changed candidate-distribution groups are not fresh/disjoint")
    datasets["ranking_ood/changed_candidate_distribution"] = shifted
    return datasets


def _view_dataset(dataset: Dataset[Any], view: str) -> Dataset[Any]:
    if view == "original":
        return dataset
    if view in {"random_so3", "axis_permutation", "sign_flip"}:
        return DiagnosticIdentityDataset(RigidTransformDataset(dataset, mode=view))
    if view == "group_shared_all_frame_point_permutation":
        return PointPermutationDataset(dataset)
    if view == "group_shared_per_frame_resampling":
        return GroupSharedRankingCorruptionDataset(
            dataset, mode="per_frame_resampling", seed=20261031
        )
    if view == "group_shared_full_part_occlusion":
        return GroupSharedRankingCorruptionDataset(
            dataset, mode="full_part_occlusion", seed=20261032
        )
    raise ValueError(f"unknown final-v2 ranking view: {view}")


def _sample_metadata(dataset: Dataset[Any], *, candidate_count: int) -> tuple[tuple[FairBaselineWorld, ...], list[dict[str, Any]]]:
    """Extract complete ordered candidate-world identity before cache lookup."""

    rows: list[dict[str, Any]] = []
    for index in range(len(dataset)):
        sample = dataset[index]
        metadata = sample.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError("ranking dataset sample lacks metadata")
        row = {str(key): _jsonable(value) for key, value in metadata.items()}
        if int(row.get("candidate_count", candidate_count)) != candidate_count:
            raise ValueError("dataset candidate_count provenance disagrees with condition")
        for field in ("group_id", "candidate_set_id", "candidate_id"):
            if field not in row:
                raise ValueError(f"ranking dataset metadata lacks {field}")
        rows.append(row)
    worlds = worlds_from_metadata(rows, candidate_count=candidate_count)
    if len(rows) != len(worlds) * candidate_count:
        raise ValueError("candidate metadata does not consist of complete candidate worlds")
    return worlds, rows


def _candidate_provenance(
    worlds: Sequence[FairBaselineWorld], rows: Sequence[Mapping[str, Any]], *, split_id: str, candidate_count: int
) -> dict[str, Any]:
    group_ids = [str(world.world_id) for world in worlds]
    layout = [world.as_dict() for world in worlds]
    distribution_values = sorted(
        {
            str(row["candidate_distribution_id"])
            for row in rows
            if row.get("candidate_distribution_id") is not None
        }
    )
    templates = sorted(
        {
            int(row["candidate_template_index"])
            for row in rows
            if row.get("candidate_template_index") is not None
        }
    )
    return {
        "split_id": split_id,
        "grouped_split_identity": f"{split_id}:{canonical_hash(group_ids)}",
        "group_ids": group_ids,
        "group_id_digest": canonical_hash(group_ids),
        "candidate_count": int(candidate_count),
        "candidate_world_count": len(worlds),
        "candidate_id_layout_digest": canonical_hash(layout),
        "candidate_distribution_ids": distribution_values or ["legacy_prefix"],
        "candidate_template_indices": templates if templates else list(range(candidate_count)),
    }


def _prediction_payload(prediction: FactorialPrediction) -> dict[str, Any]:
    """The fair cache stores only fields necessary for ranking recombination."""

    return {
        "payload_schema": "milestone2e-final-v2-fair-ranking-prediction-v1",
        "utility_probability": np.asarray(prediction.utility_probability, dtype=np.float64),
        "utility_logits": np.asarray(prediction.utility_logits, dtype=np.float64),
        "success": np.asarray(prediction.success, dtype=np.float64),
        "candidate_utility": np.asarray(prediction.candidate_utility, dtype=np.float64),
        "metadata": [{str(key): _jsonable(value) for key, value in row.items()} for row in prediction.metadata],
    }


def _prediction_from_payload(payload: Any, *, expected_rows: Sequence[Mapping[str, Any]]) -> FactorialPrediction:
    if not isinstance(payload, Mapping) or payload.get("payload_schema") != "milestone2e-final-v2-fair-ranking-prediction-v1":
        raise ValueError("fair cache payload uses an unknown ranking prediction schema")
    try:
        probability = np.asarray(payload["utility_probability"], dtype=np.float64).reshape(-1)
        logits = np.asarray(payload["utility_logits"], dtype=np.float64).reshape(-1)
        success = np.asarray(payload["success"], dtype=np.float64).reshape(-1)
        utility = np.asarray(payload["candidate_utility"], dtype=np.float64).reshape(-1)
        metadata = [dict(row) for row in payload["metadata"]]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("fair cache payload is malformed") from error
    if not (probability.size == logits.size == success.size == utility.size == len(metadata) == len(expected_rows)):
        raise ValueError("fair cache payload length does not match the requested candidate worlds")
    if not all(np.isfinite(value).all() for value in (probability, logits, success, utility)):
        raise ValueError("fair cache payload has non-finite ranking values")
    expected_index = [
        (str(row["candidate_set_id"]), str(row["candidate_id"])) for row in expected_rows
    ]
    actual_index = [
        (str(row.get("candidate_set_id")), str(row.get("candidate_id"))) for row in metadata
    ]
    if actual_index != expected_index:
        raise ValueError("fair cache payload candidate ordering does not match the requested dataset")
    return FactorialPrediction(
        utility_probability=probability,
        utility_logits=logits,
        success=success,
        candidate_utility=utility,
        metadata=metadata,
        mask_probability=np.empty(0, dtype=np.float64),
        mask_target=np.empty(0, dtype=np.float64),
    )


def _source_files() -> list[Path]:
    paths = [
        Path(__file__),
        Path(__file__).with_name("milestone2e_final_cache.py"),
        Path(__file__).with_name("milestone2e_full.py"),
        Path(__file__).with_name("milestone2e.py"),
        Path(__file__).with_name("milestone2e_completion_extensions.py"),
        Path(__file__).parent.parent / "models" / "milestone2e.py",
        Path(__file__).parent.parent / "models" / "milestone2c_baselines.py",
        Path(__file__).parent.parent / "data" / "milestone2b_dataset.py",
        Path(__file__).parent.parent / "data" / "milestone2c_dataset.py",
        Path(__file__).parent.parent / "data" / "milestone2e_final_v2_corruptions.py",
    ]
    return paths


def _get_fair_input(
    *,
    dataset: Dataset[Any],
    candidate_count: int,
    split_id: str,
    observation_view_id: str,
    corruption_id: str,
    condition_parameters: Mapping[str, Any],
    baseline_name: str,
    config: Mapping[str, Any],
    cache: FairBaselineCache,
    code_provenance: Mapping[str, Any],
    batch_size: int,
) -> FairInput:
    worlds, rows = _sample_metadata(dataset, candidate_count=candidate_count)
    candidate_provenance = _candidate_provenance(
        worlds, rows, split_id=split_id, candidate_count=candidate_count
    )
    key = FairBaselineCacheKey.from_context(
        baseline_name=baseline_name,
        baseline_parameters={
            "selection": "Milestone 2C validation-selected strongest fair non-oracle baseline",
            "condition_parameters": _jsonable(condition_parameters),
        },
        config=config,
        code_provenance=code_provenance,
        worlds=worlds,
        split_id=split_id,
        observation_view_id=observation_view_id,
        corruption_id=corruption_id,
        candidate_count=candidate_count,
        evaluation_schema_version=EVALUATION_SCHEMA_VERSION,
        metric_schema_version=CORRECTED_NDCG_METRIC_SCHEMA_VERSION,
    )
    result = cache.get_or_compute(
        key,
        lambda: _prediction_payload(
            _fair_prediction_for_baseline(baseline_name, dataset, batch_size=batch_size)
        ),
    )
    prediction = _prediction_from_payload(result.payload, expected_rows=rows)
    return FairInput(
        prediction=prediction,
        worlds=worlds,
        candidate_provenance=candidate_provenance,
        cache_evidence={
            "cache_schema": key.schema_version,
            "cache_key_digest": result.key_digest,
            "cache_path": str(result.path),
            "cache_hit": bool(result.cache_hit),
            "cache_created_at_utc": result.created_at_utc,
            "config_hash": key.config_hash,
            "metric_schema_version": key.metric_schema_version,
            "view_id": observation_view_id,
            "corruption_id": corruption_id,
        },
    )


def _metric_bundle(prediction: FactorialPrediction, scores: np.ndarray, *, score_name: str) -> dict[str, Any]:
    return ranking_metrics_tie_aware(
        RankingBundle(
            scores=np.asarray(scores, dtype=np.float64),
            success=np.asarray(prediction.success, dtype=np.float64),
            utility=np.asarray(prediction.candidate_utility, dtype=np.float64),
            metadata=prediction.metadata,
        ),
        score_name=score_name,
    )


def _evaluate_one_seed(
    model: Any, *, specification: CheckpointProvenance, dataset: Dataset[Any], fair: FactorialPrediction, batch_size: int
) -> dict[str, Any]:
    learned = predict_factorial(model, dataset, batch_size=batch_size)
    hybrid_scores, calibrated = _hybrid_scores(learned, fair, specification.calibration)
    hybrid = _metric_bundle(learned, hybrid_scores, score_name="frozen_hybrid_alpha_0.5")
    learned_only = _metric_bundle(learned, calibrated, score_name="validation_calibrated_learned_only")
    fair_only = _metric_bundle(fair, fair.utility_probability, score_name="validation_selected_fair_baseline")
    hybrid["utility_calibration"] = reliability_metrics(calibrated, learned.success, bins=15)
    learned_only["utility_calibration"] = reliability_metrics(calibrated, learned.success, bins=15)
    fair_only["utility_calibration"] = reliability_metrics(fair.utility_probability, fair.success, bins=15)
    return {
        "seed": int(specification.seed),
        "checkpoint": _jsonable(specification),
        "frozen_hybrid": hybrid,
        "learned_only": learned_only,
        "fair_only": fair_only,
    }


def _metric_names_with_ties() -> tuple[str, ...]:
    return (*RANKING_METRICS, "tie_fraction")


def _aggregate_seed_metrics(rows: Sequence[Mapping[str, Any]], *, score_name: str) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot aggregate an empty seed result")
    result: dict[str, Any] = {"seeds": [int(row["seed"]) for row in rows]}
    for metric in _metric_names_with_ties():
        values = np.asarray([float(row[score_name][metric]) for row in rows], dtype=np.float64)
        result[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1 if values.size > 1 else 0)),
            "values": values.tolist(),
        }
    calibration_keys = ("ece", "brier", "average_precision", "pr_auc")
    result["utility_calibration"] = {}
    for metric in calibration_keys:
        values = np.asarray(
            [float(row[score_name]["utility_calibration"][metric]) for row in rows], dtype=np.float64
        )
        result["utility_calibration"][metric] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1 if values.size > 1 else 0)),
            "values": values.tolist(),
        }
    return result


def _group_value_map(metrics: Mapping[str, Any], metric: str) -> dict[int, float]:
    rows = metrics.get("per_group")
    if not isinstance(rows, list):
        raise ValueError("ranking metrics lack per-group values needed for paired inference")
    result: dict[int, float] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("ranking per-group row is malformed")
        group_id = int(row["group_id"])
        if group_id in result:
            raise ValueError("ranking metrics duplicate a group ID")
        result[group_id] = float(row[metric])
    return result


def _aggregate_group_metric_across_seeds(
    rows: Sequence[Mapping[str, Any]], *, score_name: str, metric: str
) -> tuple[list[int], np.ndarray]:
    per_seed = [_group_value_map(row[score_name], metric) for row in rows]
    group_ids = sorted(per_seed[0])
    if not group_ids or any(sorted(item) != group_ids for item in per_seed[1:]):
        raise ValueError("fixed-seed results do not align on the same candidate worlds")
    matrix = np.asarray([[item[group_id] for group_id in group_ids] for item in per_seed], dtype=np.float64)
    return group_ids, matrix.mean(axis=0)


def _paired_group_mean_report(
    first: Sequence[Mapping[str, Any]],
    second: Sequence[Mapping[str, Any]],
    *,
    first_score_name: str,
    second_score_name: str,
    config: Mapping[str, Any],
    seed_offset: int,
    metrics: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Bootstrap paired scene means after averaging the five trained seeds."""

    if [int(row["seed"]) for row in first] != [int(row["seed"]) for row in second]:
        raise ValueError("paired conditions do not retain the same fixed seed order")
    report: dict[str, Any] = {
        "unit": "candidate_set; each value is the mean over the five fixed trained seeds",
        "first_minus_second": True,
        "first_score_name": first_score_name,
        "second_score_name": second_score_name,
        "metrics": {},
    }
    requested_metrics = tuple(_metric_names_with_ties() if metrics is None else metrics)
    if not requested_metrics or any(metric not in _metric_names_with_ties() for metric in requested_metrics):
        raise ValueError("paired report requests an unknown ranking metric")
    for index, metric in enumerate(requested_metrics):
        first_ids, first_values = _aggregate_group_metric_across_seeds(
            first, score_name=first_score_name, metric=metric
        )
        second_ids, second_values = _aggregate_group_metric_across_seeds(
            second, score_name=second_score_name, metric=metric
        )
        if first_ids != second_ids:
            raise ValueError("paired conditions do not retain the same candidate-world IDs")
        summary = paired_bootstrap_ci(
            first_values,
            second_values,
            seed=seed_offset + index * 37,
            resamples=int(config["statistics"]["bootstrap_resamples"]),
        )
        difference = first_values - second_values
        summary.update(
            {
                "wins": int(np.count_nonzero(difference > 1.0e-12)),
                "ties": int(np.count_nonzero(np.abs(difference) <= 1.0e-12)),
                "losses": int(np.count_nonzero(difference < -1.0e-12)),
            }
        )
        summary["direction_note"] = (
            "negative hybrid-minus-baseline normalized regret is favorable to the hybrid"
            if metric in {"regret", "normalized_regret", "failure_when_success_exists"}
            else "positive delta is favorable to the first condition"
        )
        report["metrics"][metric] = summary
    return report


def _seed_deltas(
    first: Sequence[Mapping[str, Any]], second: Sequence[Mapping[str, Any]], *, first_score_name: str, second_score_name: str
) -> list[dict[str, Any]]:
    right = {int(row["seed"]): row for row in second}
    result: list[dict[str, Any]] = []
    for row in first:
        seed = int(row["seed"])
        if seed not in right:
            raise ValueError("paired seed values are incomplete")
        result.append(
            {
                "seed": seed,
                "delta": {
                    metric: float(row[first_score_name][metric]) - float(right[seed][second_score_name][metric])
                    for metric in _metric_names_with_ties()
                },
            }
        )
    return result


def _condition_report(
    *,
    condition_id: str,
    description: str,
    split_id: str,
    candidate_count: int,
    fair_input: FairInput,
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    pairing_seed: int,
    condition_parameters: Mapping[str, Any],
    fair_ranking_selection: Mapping[str, Any],
    include_fair_pairing: bool,
) -> dict[str, Any]:
    report = {
        "condition_id": condition_id,
        "description": description,
        "split_id": split_id,
        "candidate_count": int(candidate_count),
        "condition_parameters": _jsonable(condition_parameters),
        "frozen_fair_ranking_baseline": {
            "baseline_name": str(fair_ranking_selection["selected"]["baseline_name"]),
            "selection_split": str(fair_ranking_selection["selection_split"]),
            "selection_rule": str(fair_ranking_selection["selection_rule"]),
            "selection_artifact": FAIR_RANKING_SELECTION_FILE,
        },
        "candidate_set_provenance": fair_input.candidate_provenance,
        "fair_baseline_cache": fair_input.cache_evidence,
        "scene_support_count": len(fair_input.worlds),
        "per_seed": list(rows),
        "absolute_metrics": {
            "frozen_hybrid": _aggregate_seed_metrics(rows, score_name="frozen_hybrid"),
            "learned_only": _aggregate_seed_metrics(rows, score_name="learned_only"),
            # The fair method is deterministic, but duplicating it in every
            # seed row makes candidate ordering auditable.  Its five values
            # must remain exactly equal.
            "fair_only": _aggregate_seed_metrics(rows, score_name="fair_only"),
        },
    }
    if include_fair_pairing:
        report.update(_fair_pairing_fields(rows, config=config, pairing_seed=pairing_seed))
    return report


def _fair_pairing_fields(
    rows: Sequence[Mapping[str, Any]], *, config: Mapping[str, Any], pairing_seed: int
) -> dict[str, Any]:
    """Add Stage-A5 fair-comparator CIs without treating seed replicas as scenes."""

    fair_rows = [
        {**row, "fair_only": row["fair_only"]}
        for row in rows
    ]
    return {
        "seed_level_hybrid_minus_fair": _seed_deltas(
            rows, fair_rows, first_score_name="frozen_hybrid", second_score_name="fair_only"
        ),
        "seed_level_learned_only_minus_fair": _seed_deltas(
            rows, fair_rows, first_score_name="learned_only", second_score_name="fair_only"
        ),
        "paired_hybrid_minus_fair_group_mean": _paired_group_mean_report(
            rows,
            fair_rows,
            first_score_name="frozen_hybrid",
            second_score_name="fair_only",
            config=config,
            seed_offset=pairing_seed,
        ),
        "paired_learned_only_minus_fair_group_mean": _paired_group_mean_report(
            rows,
            fair_rows,
            first_score_name="learned_only",
            second_score_name="fair_only",
            config=config,
            seed_offset=pairing_seed + 503,
        ),
    }


def _evaluate_condition(
    *,
    condition_id: str,
    description: str,
    split_id: str,
    dataset: Dataset[Any],
    candidate_count: int,
    observation_view_id: str,
    corruption_id: str,
    condition_parameters: Mapping[str, Any],
    models: Mapping[int, Any],
    specifications: Mapping[int, CheckpointProvenance],
    config: Mapping[str, Any],
    cache: FairBaselineCache,
    code_provenance: Mapping[str, Any],
    pairing_seed: int,
    fair_ranking_selection: Mapping[str, Any],
    include_fair_pairing: bool = True,
) -> dict[str, Any]:
    baseline_name = str(fair_ranking_selection["selected"]["baseline_name"])
    fair_input = _get_fair_input(
        dataset=dataset,
        candidate_count=candidate_count,
        split_id=split_id,
        observation_view_id=observation_view_id,
        corruption_id=corruption_id,
        condition_parameters=condition_parameters,
        baseline_name=baseline_name,
        config=config,
        cache=cache,
        code_provenance=code_provenance,
        batch_size=int(config["training"]["evaluation_batch_size"]),
    )
    rows = [
        _evaluate_one_seed(
            models[seed],
            specification=specifications[seed],
            dataset=dataset,
            fair=fair_input.prediction,
            batch_size=int(config["training"]["evaluation_batch_size"]),
        )
        for seed in sorted(models)
    ]
    return _condition_report(
        condition_id=condition_id,
        description=description,
        split_id=split_id,
        candidate_count=candidate_count,
        fair_input=fair_input,
        rows=rows,
        config=config,
        pairing_seed=pairing_seed,
        condition_parameters=condition_parameters,
        fair_ranking_selection=fair_ranking_selection,
        include_fair_pairing=include_fair_pairing,
    )


def _invariance_report(
    *,
    config: Mapping[str, Any],
    datasets: Mapping[str, Dataset[Any]],
    models: Mapping[int, Any],
    specifications: Mapping[int, CheckpointProvenance],
    selection: Mapping[str, str],
    cache: FairBaselineCache,
    code_provenance: Mapping[str, Any],
    fair_ranking_selection: Mapping[str, Any],
) -> dict[str, Any]:
    counts: dict[str, Any] = {}
    for count_index, count in enumerate(FIXED_CANDIDATE_COUNTS):
        base = datasets[f"ranking_test/{count}"]
        views: dict[str, Any] = {}
        original_rows: Sequence[Mapping[str, Any]] | None = None
        for view_index, view in enumerate(INVARIANCE_VIEWS):
            dataset = _view_dataset(base, view)
            parameters = {
                "stage": "A3_final_ranking_invariance",
                "view": view,
                "view_seed": (
                    20260911
                    if view in {"random_so3", "axis_permutation", "sign_flip"}
                    else 20261331
                    if view == "group_shared_all_frame_point_permutation"
                    else 20261031
                    if view == "group_shared_per_frame_resampling"
                    else 20261032
                    if view == "group_shared_full_part_occlusion"
                    else None
                ),
            }
            condition = _evaluate_condition(
                condition_id=view,
                description=(
                    "Frozen unmodified observations"
                    if view == "original"
                    else "One deterministic group-shared observation transformation; labels and candidate actions are unchanged."
                ),
                split_id=f"id/ranking_test/C{count}",
                dataset=dataset,
                candidate_count=count,
                observation_view_id=view,
                corruption_id="none" if view == "original" else view,
                condition_parameters=parameters,
                models=models,
                specifications=specifications,
                config=config,
                cache=cache,
                code_provenance=code_provenance,
                pairing_seed=20262000 + count_index * 1000 + view_index * 100,
                fair_ranking_selection=fair_ranking_selection,
                include_fair_pairing=False,
            )
            if view == "original":
                original_rows = condition["per_seed"]
            else:
                if original_rows is None:
                    raise AssertionError("original view must precede transformed views")
                condition["paired_view_minus_original_group_mean"] = _paired_group_mean_report(
                    condition["per_seed"],
                    original_rows,
                    first_score_name="frozen_hybrid",
                    second_score_name="frozen_hybrid",
                    config=config,
                    seed_offset=20265000 + count_index * 1000 + view_index * 100,
                    metrics=A3_PAIRED_METRICS,
                )
                condition["seed_level_view_minus_original"] = _seed_deltas(
                    condition["per_seed"],
                    original_rows,
                    first_score_name="frozen_hybrid",
                    second_score_name="frozen_hybrid",
                )
            views[view] = condition
        if tuple(views) != INVARIANCE_VIEWS:
            raise AssertionError("A3 invariance views do not match the frozen protocol")
        counts[f"C{count}"] = {"candidate_count": count, "views": views}
    return {
        "artifact_schema": ARTIFACT_SCHEMA_VERSION,
        "stage": "A3_final_ranking_invariance",
        "config_digest": _json_digest(config),
        "source_code_provenance": _jsonable(code_provenance),
        "metric_definition_version": RANKING_METRIC_SCHEMA,
        "ndcg_definition": NDCG_DEFINITION,
        "tie_policy": TIE_POLICY,
        "validation_selected_arm": dict(selection),
        "frozen_fair_ranking_baseline": {
            "baseline_name": str(fair_ranking_selection["selected"]["baseline_name"]),
            "selection_split": str(fair_ranking_selection["selection_split"]),
            "selection_rule": str(fair_ranking_selection["selection_rule"]),
        },
        "checkpoint_provenance": [asdict(specifications[seed]) for seed in sorted(specifications)],
        "fixed_views": list(INVARIANCE_VIEWS),
        "candidate_counts": counts,
        "protocol_note": (
            "Every metric was regenerated from selected frozen checkpoints and the versioned fair cache. "
            "No legacy factorial_incremental JSON was read as a corrected-NDCG input."
        ),
    }


def _candidate_count_conditions_from_invariance(
    invariance: Mapping[str, Any], *, config: Mapping[str, Any]
) -> dict[str, Any]:
    conditions: dict[str, Any] = {}
    for index, count in enumerate(FIXED_CANDIDATE_COUNTS):
        raw = invariance["candidate_counts"][f"C{count}"]["views"]["original"]
        # Copy the C-count condition and give it an A5-specific condition id;
        # the raw predictions were already regenerated by A3 under exactly the
        # same frozen worlds, so this avoids a second learned/fair evaluation.
        copied = dict(raw)
        copied["condition_id"] = f"id_candidate_count_C{count}"
        copied["description"] = (
            f"ID frozen candidate worlds at fixed C{count}; reused only from the freshly regenerated A3 original view."
        )
        copied["condition_parameters"] = {
            "stage": "A5_fixed_candidate_count_comparison",
            "candidate_count": count,
            "source": "A3/original regenerated corrected evaluation",
        }
        copied.update(
            _fair_pairing_fields(
                copied["per_seed"], config=config, pairing_seed=20269000 + index * 1000
            )
        )
        conditions[f"id_candidate_count_C{count}"] = copied
    return conditions


def _ood_conditions_report(
    *,
    config: Mapping[str, Any],
    datasets: Mapping[str, Dataset[Any]],
    models: Mapping[int, Any],
    specifications: Mapping[int, CheckpointProvenance],
    cache: FairBaselineCache,
    code_provenance: Mapping[str, Any],
    fair_ranking_selection: Mapping[str, Any],
) -> dict[str, Any]:
    conditions: dict[str, Any] = {}
    descriptions = {
        "unseen_velocity_magnitude": "Unseen velocity-magnitude domain.",
        "unseen_acceleration": "Unseen acceleration domain.",
        "unseen_curvature": "Unseen curved nonlinear-motion domain.",
        "unseen_observation_delay": "Unseen observation-delay domain.",
        "unseen_action_delay": "Unseen action-execution-delay domain.",
        "unseen_noise_level": "Unseen point-noise domain.",
        "unseen_occlusion_rate": "Unseen occlusion-rate domain.",
        "unseen_object_count": "Unseen increased-distractor object-count domain (3 to 7 objects; fixed point budget).",
        "unseen_hard_case_combination": "Unseen fixed hard-combination domain.",
    }
    for index, axis in enumerate(MILESTONE2B_OOD_AXES):
        dataset = datasets[f"ranking_ood/{axis}"]
        conditions[axis] = _evaluate_condition(
            condition_id=axis,
            description=descriptions[axis],
            split_id=f"ood/{axis}",
            dataset=dataset,
            candidate_count=20,
            observation_view_id="original",
            corruption_id=axis,
            condition_parameters={"stage": "A5_ood", "axis": axis},
            models=models,
            specifications=specifications,
            config=config,
            cache=cache,
            code_provenance=code_provenance,
            pairing_seed=20267000 + index * 1000,
            fair_ranking_selection=fair_ranking_selection,
        )
    shifted = datasets["ranking_ood/changed_candidate_distribution"]
    conditions["changed_candidate_distribution"] = _evaluate_condition(
        condition_id="changed_candidate_distribution",
        description="Predeclared candidate-template-mixture shift on fresh fixed C20 candidate worlds.",
        split_id="ood/changed_candidate_distribution",
        dataset=shifted,
        candidate_count=20,
        observation_view_id="original",
        corruption_id="predeclared_candidate_template_mixture_shift",
        condition_parameters={
            "stage": "A5_ood",
            "candidate_distribution_id": CHANGED_CANDIDATE_DISTRIBUTION_ID,
            "template_indices": list(CHANGED_CANDIDATE_TEMPLATE_INDICES),
            "template_indices_digest": canonical_hash(CHANGED_CANDIDATE_TEMPLATE_INDICES),
            "fresh_master_seed": CHANGED_CANDIDATE_MASTER_SEED,
            "fresh_group_count": CHANGED_CANDIDATE_GROUPS,
            "predeclared_before_test_evaluation": True,
        },
        models=models,
        specifications=specifications,
        config=config,
        cache=cache,
        code_provenance=code_provenance,
        pairing_seed=20268000,
        fair_ranking_selection=fair_ranking_selection,
    )
    return conditions


def _id_ood_report(
    *,
    config: Mapping[str, Any],
    datasets: Mapping[str, Dataset[Any]],
    invariance: Mapping[str, Any],
    models: Mapping[int, Any],
    specifications: Mapping[int, CheckpointProvenance],
    selection: Mapping[str, str],
    cache: FairBaselineCache,
    code_provenance: Mapping[str, Any],
    fair_ranking_selection: Mapping[str, Any],
) -> dict[str, Any]:
    candidate_count_conditions = _candidate_count_conditions_from_invariance(invariance, config=config)
    ood_conditions = _ood_conditions_report(
        config=config,
        datasets=datasets,
        models=models,
        specifications=specifications,
        cache=cache,
        code_provenance=code_provenance,
        fair_ranking_selection=fair_ranking_selection,
    )
    # A5 explicitly includes a ranking-resampling condition.  It is the A3 C20
    # group-shared view, not candidate-specific legacy corruption.
    resampling = dict(
        invariance["candidate_counts"]["C20"]["views"]["group_shared_per_frame_resampling"]
    )
    resampling["condition_id"] = "group_shared_per_frame_resampling"
    resampling["description"] = "Group-shared C20 ranking resampling condition regenerated in A3."
    resampling["condition_parameters"] = {
        "stage": "A5_ood_observation_corruption",
        "view": "group_shared_per_frame_resampling",
        "source": "A3 regenerated corrected evaluation",
    }
    resampling.update(
        _fair_pairing_fields(
            resampling["per_seed"], config=config, pairing_seed=20269500
        )
    )
    ood_conditions["group_shared_per_frame_resampling"] = resampling
    return {
        "artifact_schema": ARTIFACT_SCHEMA_VERSION,
        "stage": "A5_complete_id_ood_ranking",
        "config_digest": _json_digest(config),
        "source_code_provenance": _jsonable(code_provenance),
        "metric_definition_version": RANKING_METRIC_SCHEMA,
        "ndcg_definition": NDCG_DEFINITION,
        "tie_policy": TIE_POLICY,
        "validation_selected_arm": dict(selection),
        "frozen_strongest_fair_non_oracle_ranking_baseline": {
            "baseline_name": str(fair_ranking_selection["selected"]["baseline_name"]),
            "selection_split": str(fair_ranking_selection["selection_split"]),
            "selection_rule": str(fair_ranking_selection["selection_rule"]),
            "selection_artifact": FAIR_RANKING_SELECTION_FILE,
        },
        "checkpoint_provenance": [asdict(specifications[seed]) for seed in sorted(specifications)],
        "id_fixed_candidate_count_conditions": candidate_count_conditions,
        "ood_conditions": ood_conditions,
        "paired_delta_semantics": {
            "reported_delta": "hybrid minus frozen fair baseline, paired by candidate world after five-seed hybrid mean",
            "regret_direction": "negative normalized-regret delta favors the hybrid; positive Top-1 delta favors the hybrid",
            "positive_stability_is_insufficient": True,
        },
        "protocol_note": (
            "The fair comparator was selected once using only frozen C20 ranking validation worlds. "
            "No OOD/test result selected a baseline, template mixture, calibration, or model."
        ),
    }


def _evaluation_schema(
    config: Mapping[str, Any], *, selection: Mapping[str, str], fair_ranking_selection: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "artifact_schema": ARTIFACT_SCHEMA_VERSION,
        "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
        "metric_definition_version": RANKING_METRIC_SCHEMA,
        "ndcg_definition": NDCG_DEFINITION,
        "ndcg_compatibility": {
            "historical_legacy_incremental": "unversioned / old non-standard NDCG; immutable and excluded",
            "final_v2": RANKING_METRIC_SCHEMA,
            "difference": "Top-1, Top-3, regret, AP, AUC, MRR, and calibration definitions are retained; only NDCG is regenerated using standard utility-gain DCG/IDCG.",
        },
        "metrics": list(RANKING_METRICS),
        "additional_ranking_metrics": ["tie_fraction", "utility_calibration"],
        "tie_policy": TIE_POLICY,
        "score_definitions": {
            "frozen_hybrid": "0.5 group-normalized validation-Platt learned score + 0.5 group-normalized fixed fair baseline score",
            "learned_only": "existing validation-only Platt calibration, no recalibration on transformed/test/OOD data",
            "fair_only": "parameter-free fair baseline selected once on frozen C20 ranking validation by normalized regret then Top-1",
        },
        "validation_selected_arm": dict(selection),
        "frozen_fair_ranking_comparator": {
            "baseline_name": str(fair_ranking_selection["selected"]["baseline_name"]),
            "selection_split": str(fair_ranking_selection["selection_split"]),
            "selection_rule": str(fair_ranking_selection["selection_rule"]),
            "selection_artifact": FAIR_RANKING_SELECTION_FILE,
        },
        "frozen_config_digest": _json_digest(config),
        "candidate_counts": list(FIXED_CANDIDATE_COUNTS),
        "grouped_split_unit": "candidate_set_id / base scene; paired CIs resample candidate worlds, not candidates or seed replicas",
    }


def _frozen_input_manifest(
    config: Mapping[str, Any],
    *,
    selection: Mapping[str, str],
    specifications: Mapping[int, CheckpointProvenance],
    fair_ranking_selection: Mapping[str, Any],
    legacy_root: Path,
    final_root: Path,
) -> dict[str, Any]:
    legacy_incremental = sorted((legacy_root / "factorial_incremental").glob("*.json"))
    source_files = [
        Path(DEFAULT_CONFIG),
        legacy_root / "validation_primary_selection.json",
        legacy_root / "training_records_primary.json",
        FROZEN_2C_ROOT / "fair_baseline_selection.json",
    ]
    return {
        "artifact_schema": INPUT_MANIFEST_SCHEMA_VERSION,
        "legacy_source_root": str(legacy_root),
        "final_write_root": str(final_root),
        "frozen_config": str(DEFAULT_CONFIG),
        "frozen_config_sha256": _sha256_file(Path(DEFAULT_CONFIG)),
        "frozen_config_digest": _json_digest(config),
        "validation_selected_arm": dict(selection),
        "legacy_mask_ap_fair_baseline_reference": _legacy_fair_baseline_selection_reference(),
        "frozen_fair_ranking_baseline": {
            "baseline_name": str(fair_ranking_selection["selected"]["baseline_name"]),
            "selection_split": str(fair_ranking_selection["selection_split"]),
            "selection_rule": str(fair_ranking_selection["selection_rule"]),
            "selection_artifact": FAIR_RANKING_SELECTION_FILE,
            "selection_artifact_sha256": _sha256_file(final_root / FAIR_RANKING_SELECTION_FILE),
        },
        "primary_checkpoints": [asdict(specifications[seed]) for seed in sorted(specifications)],
        "source_artifact_sha256": {str(path): _sha256_file(path) for path in source_files},
        "legacy_factorial_incremental_count": len(legacy_incremental),
        "legacy_incremental_policy": "Never read as corrected-NDCG input; kept immutable as historical context only.",
        "final_runner_code_provenance": code_provenance_for_files(
            _source_files(), repository_root=Path(PROJECT_ROOT)
        ),
    }


def verify_inputs(
    config_path: Path = DEFAULT_CONFIG,
    *,
    legacy_root: Path = LEGACY_ROOT,
    final_root: Path = FINAL_ROOT,
) -> dict[str, Any]:
    """Read-only proof that frozen checkpoints and selected arm are usable."""

    config = _load_config(Path(config_path))
    if str(config["experiment"]["device"]).lower() != "cpu":
        raise ValueError("frozen Milestone 2E config is no longer CPU-only")
    if tuple(int(value) for value in config["ranking"]["candidate_counts"]) != FIXED_CANDIDATE_COUNTS:
        raise ValueError("final v2 requires frozen C5/C10/C20/C50 counts")
    if len(config["experiment"]["training_seeds"]) != 5:
        raise ValueError("final v2 requires five frozen training seeds")
    selection = _selected_primary(config, legacy_root=Path(legacy_root))
    # Loading validates every checkpoint's exact metadata but does not run a
    # forward pass or create any output/cache artifact.
    _, specifications = _load_primary_models(config, legacy_root=Path(legacy_root), selection=selection)
    selection_path = Path(final_root) / FAIR_RANKING_SELECTION_FILE
    fair_selection: dict[str, Any] = {
        "status": "not_yet_selected",
        "policy": "full run selects the strongest parameter-free fair ranking baseline on frozen ranking_val/C20 only",
        "legacy_mask_ap_reference": _legacy_fair_baseline_selection_reference(),
    }
    if selection_path.exists():
        fair_selection = _load_final_fair_ranking_selection(
            selection_path,
            config=config,
            code_provenance=code_provenance_for_files(_source_files(), repository_root=Path(PROJECT_ROOT)),
        )
    return {
        "status": "verified",
        "read_only": True,
        "cpu_protocol": True,
        "cuda_visible_to_current_python": bool(torch.cuda.is_available()),
        "cuda_device_count_to_current_python": int(torch.cuda.device_count()),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "legacy_source_root": str(legacy_root),
        "final_write_root": str(final_root),
        "validation_selected_arm": selection,
        "primary_checkpoints": [asdict(specifications[seed]) for seed in sorted(specifications)],
        "fair_ranking_baseline": fair_selection,
        "corrected_metric_schema": RANKING_METRIC_SCHEMA,
        "legacy_incremental_not_used": True,
    }


def run(
    config_path: Path = DEFAULT_CONFIG,
    *,
    legacy_root: Path = LEGACY_ROOT,
    final_root: Path = FINAL_ROOT,
) -> dict[str, Any]:
    """Perform Stage A2/A3/A5 evaluation without mutating legacy outputs."""

    started = time.perf_counter()
    legacy_root, final_root = Path(legacy_root), Path(final_root)
    verification = verify_inputs(config_path, legacy_root=legacy_root, final_root=final_root)
    config = _load_config(Path(config_path))
    torch.set_num_threads(int(config["training"]["num_threads"]))
    selection = dict(verification["validation_selected_arm"])
    models, specifications = _load_primary_models(config, legacy_root=legacy_root, selection=selection)
    source_provenance = code_provenance_for_files(_source_files(), repository_root=Path(PROJECT_ROOT))
    datasets = build_final_v2_datasets(config)
    cache = FairBaselineCache(final_root / "fair_baseline_cache")
    fair_ranking_selection = _select_fair_ranking_baseline(
        config=config,
        validation_dataset=datasets["ranking_val/20"],
        final_root=final_root,
        cache=cache,
        code_provenance=source_provenance,
    )

    # Deterministic manifests are safe to resume: a changed config/source or
    # checkpoint digest is a hard failure, never a silent continuation.
    _ensure_immutable_json(
        final_root / "evaluation_schema.json",
        _evaluation_schema(
            config, selection=selection, fair_ranking_selection=fair_ranking_selection
        ),
    )
    _ensure_immutable_json(
        final_root / "frozen_input_manifest.json",
        _frozen_input_manifest(
            config,
            selection=selection,
            specifications=specifications,
            fair_ranking_selection=fair_ranking_selection,
            legacy_root=legacy_root,
            final_root=final_root,
        ),
    )

    invariance_path = final_root / "invariance" / "primary_ranking_invariance.json"
    if invariance_path.exists():
        invariance = _read_mapping(invariance_path)
        _validate_existing_phase_artifact(
            invariance,
            stage="A3_final_ranking_invariance",
            config=config,
            selection=selection,
            fair_ranking_selection=fair_ranking_selection,
            code_provenance=source_provenance,
        )
    else:
        invariance = _invariance_report(
            config=config,
            datasets=datasets,
            models=models,
            specifications=specifications,
            selection=selection,
            cache=cache,
            code_provenance=source_provenance,
            fair_ranking_selection=fair_ranking_selection,
        )
        _write_json_new(invariance_path, invariance)

    id_ood_path = final_root / "id_ood" / "primary_id_ood_ranking.json"
    if id_ood_path.exists():
        id_ood = _read_mapping(id_ood_path)
        _validate_existing_phase_artifact(
            id_ood,
            stage="A5_complete_id_ood_ranking",
            config=config,
            selection=selection,
            fair_ranking_selection=fair_ranking_selection,
            code_provenance=source_provenance,
        )
    else:
        id_ood = _id_ood_report(
            config=config,
            datasets=datasets,
            invariance=invariance,
            models=models,
            specifications=specifications,
            selection=selection,
            cache=cache,
            code_provenance=source_provenance,
            fair_ranking_selection=fair_ranking_selection,
        )
        _write_json_new(id_ood_path, id_ood)

    run_manifest_path = final_root / "run_manifest_a2_a3_a5.json"
    if not run_manifest_path.exists():
        _write_json_new(
            run_manifest_path,
            {
                "artifact_schema": RUN_MANIFEST_SCHEMA_VERSION,
                "completed_at_utc": _timestamp(),
                "runtime_seconds": time.perf_counter() - started,
                "cpu_only": True,
                "gpu_used": False,
                "cuda_visible_to_current_python": bool(torch.cuda.is_available()),
                "legacy_source_root": str(legacy_root),
                "final_write_root": str(final_root),
                "config_digest": _json_digest(config),
                "source_code_provenance": source_provenance,
                "artifacts": {
                    "evaluation_schema": str(final_root / "evaluation_schema.json"),
                    "frozen_input_manifest": str(final_root / "frozen_input_manifest.json"),
                    "a3_invariance": str(invariance_path),
                    "a5_id_ood": str(id_ood_path),
                },
                "legacy_incremental_not_used": True,
            },
        )
    return {
        "status": "complete",
        "stage": "A2/A3/A5",
        "cpu_only": True,
        "gpu_used": False,
        "invariance_path": str(invariance_path),
        "id_ood_path": str(id_ood_path),
        "run_manifest_path": str(run_manifest_path),
    }


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-inputs",
        action="store_true",
        help="read and provenance-check frozen inputs only; do not evaluate or write an artifact",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="alias for --verify-inputs; deliberately non-decisional and read-only",
    )
    parser.add_argument("--legacy-root", type=Path, default=LEGACY_ROOT)
    parser.add_argument("--output-root", type=Path, default=FINAL_ROOT)
    arguments = parser.parse_args(argv)
    if arguments.verify_inputs or arguments.smoke:
        report = verify_inputs(legacy_root=arguments.legacy_root, final_root=arguments.output_root)
        print(report["status"])
        return report
    report = run(legacy_root=arguments.legacy_root, final_root=arguments.output_root)
    print(report["status"])
    return report


if __name__ == "__main__":
    main()


__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "CHANGED_CANDIDATE_DISTRIBUTION_ID",
    "CHANGED_CANDIDATE_MASTER_SEED",
    "CHANGED_CANDIDATE_TEMPLATE_INDICES",
    "FINAL_ROOT",
    "INVARIANCE_VIEWS",
    "LEGACY_ROOT",
    "build_final_v2_datasets",
    "main",
    "run",
    "verify_inputs",
]
