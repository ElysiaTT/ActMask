"""CPU-only coordinate-invariance audit for the Milestone 2C protocol."""
from __future__ import annotations
import copy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import Dataset
from actmask.data.milestone2c_dataset import CorrespondenceCorruptionDataset, RigidTransformDataset
from actmask.eval.statistics import paired_summary
from actmask.experiments.milestone2c import (DEFAULT_CONFIG, PROJECT_ROOT, _ensemble_model_bundle, _loader, _normalize_by_group, _seed_everything, _validation_ap, build_datasets, predict_baseline, ranking_metrics, train_model)
from actmask.models.correspondence import MotionConsistentMutualTemporalActMask
from actmask.models.milestone2c_baselines import FAIR_BASELINES_2C

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "actmask" / "milestone2c_coordinate_audit"
FROZEN_DIR = PROJECT_ROOT / "outputs" / "actmask" / "milestone2c"
METRICS = ("top1_success", "top3_success", "ranking_auc", "mrr", "ndcg", "regret")

def _jsonable(value: Any) -> Any:
    if isinstance(value, Path): return str(value)
    if isinstance(value, Tensor): return value.item() if value.numel() == 1 else value.detach().cpu().tolist()
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, Mapping): return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)): return [_jsonable(item) for item in value]
    if isinstance(value, np.generic): return value.item()
    return value

def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n")

def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n")

def _frozen_selection() -> dict[str, Any]:
    ranking = json.loads((FROZEN_DIR / "ranking_evaluation.json").read_text())
    fair = json.loads((FROZEN_DIR / "fair_baseline_selection.json").read_text())
    selected = ranking["utility_selection"]["selected"]
    if selected["score_name"] != "hybrid_alpha_0.5": raise AssertionError("frozen utility rule drift")
    if fair["selected"]["method"] != "MultiHypothesisTrajectoryProximity": raise AssertionError("frozen fair baseline drift")
    return {"selection_split": "frozen 2C validation only", "score_name": selected["score_name"], "hybrid_alpha": 0.5, "fair_baseline": fair["selected"]["method"]}

def _identity_history(batch: Mapping[str, Any], frames: int, points: int) -> Tensor:
    hidden = batch.get("hidden_state", {})
    value = hidden.get("evaluation_identity_history") if isinstance(hidden, Mapping) else None
    if isinstance(value, Tensor): return value.long()
    size = int(batch["observable"]["points_history"].shape[0])
    return torch.arange(points, dtype=torch.long)[None, None].expand(size, frames, points)

def _association_quality(model: MotionConsistentMutualTemporalActMask, dataset: Dataset[Any], *, batch_size: int) -> dict[str, float | int]:
    counts = {key: 0 for key in ("anchors", "mutual", "rejected", "matchable", "correct", "dynamic_matchable", "dynamic_correct", "duplicate_excess", "accepted")}
    errors: list[float] = []
    model.eval()
    with torch.no_grad():
        for batch in _loader(dataset, batch_size):
            observable = batch["observable"]
            result = model.associate(observable)
            history, visible = observable["points_history"], observable["visibility_history"].bool()
            size, frames, points, _ = history.shape
            identities = _identity_history(batch, frames, points)
            hidden = batch.get("hidden_state", {})
            dynamic = torch.ones((size, points), dtype=torch.bool)
            if isinstance(hidden, Mapping) and isinstance(hidden.get("exact_point_velocity"), Tensor): dynamic = torch.linalg.vector_norm(hidden["exact_point_velocity"], dim=-1) > 0.02
            anchor_id, anchor_visible = identities[:, -1], visible[:, -1]
            for frame in range(frames - 1):
                source_id = identities[:, frame]
                equal = anchor_id[:, :, None] == source_id[:, None, :]
                expected = equal.long().argmax(dim=-1)
                matchable = anchor_visible & equal.any(dim=-1) & visible[:, frame].gather(1, expected)
                accepted = result.source_index[:, frame] >= 0
                correct = accepted & (result.source_index[:, frame] == expected) & matchable
                counts["anchors"] += int(anchor_visible.sum())
                counts["mutual"] += int((result.mutual[:, frame] & anchor_visible).sum())
                counts["rejected"] += int(((~accepted) & anchor_visible).sum())
                counts["matchable"] += int(matchable.sum())
                counts["correct"] += int(correct.sum())
                counts["dynamic_matchable"] += int((matchable & dynamic).sum())
                counts["dynamic_correct"] += int((correct & dynamic).sum())
                for row in result.source_index[:, frame]:
                    used = row[row >= 0]
                    counts["accepted"] += int(used.numel())
                    counts["duplicate_excess"] += int(used.numel() - torch.unique(used).numel())
                if accepted.any():
                    selected = history[:, frame].gather(1, result.source_index[:, frame].clamp_min(0)[:, :, None].expand(size, points, 3))
                    truth = history[:, frame].gather(1, expected[:, :, None].expand(size, points, 3))
                    mask = accepted & matchable
                    if mask.any(): errors.extend(torch.linalg.vector_norm(selected - truth, dim=-1)[mask].tolist())
    return {"mutual_match_rate": counts["mutual"] / max(counts["anchors"], 1), "rejected_unmatched_rate": counts["rejected"] / max(counts["anchors"], 1), "duplicate_match_rate": counts["duplicate_excess"] / max(counts["accepted"], 1), "correspondence_error_mean": float(np.mean(errors)) if errors else 0.0, "correspondence_error_p95": float(np.quantile(errors, 0.95)) if errors else 0.0, "correspondence_accuracy": counts["correct"] / max(counts["matchable"], 1), "dynamic_point_correspondence_accuracy": counts["dynamic_correct"] / max(counts["dynamic_matchable"], 1), "matchable_points": counts["matchable"], "dynamic_matchable_points": counts["dynamic_matchable"], "accepted_matches": counts["accepted"]}

def _rank_dataset(dataset: Dataset[Any], name: str) -> Dataset[Any]:
    if name == "original": return dataset
    if name == "random_so3": return RigidTransformDataset(dataset, mode="random_so3")
    if name == "axis_permutation": return RigidTransformDataset(dataset, mode="axis_permutation")
    if name == "sign_flip": return RigidTransformDataset(dataset, mode="sign_flip")
    if name == "resampled": return CorrespondenceCorruptionDataset(dataset, mode="per_frame_resampling", seed=20260920)
    if name == "occluded": return CorrespondenceCorruptionDataset(dataset, mode="full_part_occlusion", seed=20260921)
    if name == "rotated_resampled": return RigidTransformDataset(CorrespondenceCorruptionDataset(dataset, mode="per_frame_resampling", seed=20260920), mode="random_so3")
    if name == "permuted_occluded": return RigidTransformDataset(CorrespondenceCorruptionDataset(dataset, mode="full_part_occlusion", seed=20260921), mode="axis_permutation")
    raise ValueError(name)

def _fixed_hybrid(learned: Any, fair: Any) -> np.ndarray:
    return 0.5 * _normalize_by_group(learned, learned.utility) + 0.5 * _normalize_by_group(learned, fair.utility)

def _pair(first: Mapping[str, Any], second: Mapping[str, Any], seed: int) -> dict[str, Any]:
    right = {int(row["group_id"]): row for row in second["per_group"]}
    return {metric: paired_summary([row[metric] for row in first["per_group"]], [right[int(row["group_id"])][metric] for row in first["per_group"]], seed=seed + 19 * offset, resamples=2000, permutations=4096) for offset, metric in enumerate(METRICS)}

def _ranking(models: Mapping[str, Sequence[nn.Module]], datasets: Mapping[str, Dataset[Any]], fair_name: str, batch_size: int) -> dict[str, Any]:
    # Ranking views required by the audit.  Combination stress cases remain in
    # the correspondence-level tests; keeping them out here makes this a
    # bounded smoke evaluation while preserving the frozen candidate worlds.
    variants = ("original", "random_so3", "axis_permutation", "resampled", "occluded")
    fair_model = FAIR_BASELINES_2C[fair_name]().eval()
    output: dict[str, Any] = {"variants": list(variants), "models": {}}
    for name, members in models.items():
        counts: dict[str, Any] = {}
        for count in (10, 20):
            rows: dict[str, Any] = {}
            for variant in variants:
                view = _rank_dataset(datasets[f"ranking_test/{count}"], variant)
                learned = _ensemble_model_bundle(members, view, batch_size)
                fair = predict_baseline(fair_model, view, batch_size=batch_size)
                rows[variant] = ranking_metrics(learned, _fixed_hybrid(learned, fair), score_name="frozen_hybrid_alpha_0.5")
            original = rows["original"]
            counts[str(count)] = {"metrics": {key: {metric: value for metric, value in row.items() if metric != "per_group"} for key, row in rows.items()}, "paired_vs_original": {key: _pair(original, row, 20261010 + 101 * count + index) for index, (key, row) in enumerate(rows.items()) if key != "original"}}
        output["models"][name] = counts
    return output

def _summary(report: Mapping[str, Any]) -> str:
    frozen = report["frozen_selection"]
    lines = ["# ActMask Coordinate-Invariant Correspondence Audit", "", "**Decision retained: NO-GO FOR SIMULATOR INTEGRATION.** This audit does not recompute every frozen 2C pivot gate, so it cannot change that decision.", "", "## Frozen protocol", "", f"- Score rule: `{frozen['score_name']}` (validation-selected before transformed tests).", f"- Fair component: `{frozen['fair_baseline']}`; labels and candidate sets unchanged.", f"- Ranking audit worlds: {report['ranking_smoke_test_groups']} per candidate count; matching-quality worlds: {report['matching_quality_test_groups']} (CPU smoke audit; candidate construction unchanged).", f"- Audit training seed: `{report['training_seeds']}`; transformed results are never used for selection.", "", "## New primary correspondence", "", "- Bootstrap with full-3D Euclidean mutual NN; use its observable velocity to predict older-frame 3D positions.", "- Keep only mutual, confidence-passing matches; encode no-match as `-1`/invisible.", "- Legacy `|dx| + 0.28||d_yz||` is diagnostic-only and is detected by axis-permutation tests.", "", "## Untransformed AP", ""]
    lines += [f"- {name}: {value:.4f}" for name, value in report["untransformed_ap"].items()]
    lines += ["", "See `matching_quality.json` and `ranking_invariance.json` for all required metrics and paired 95% CIs."]
    return "\n".join(lines)

def run() -> dict[str, Any]:
    from actmask.experiments.milestone2c import _load_config
    config = copy.deepcopy(_load_config(DEFAULT_CONFIG))
    config["experiment"]["output_dir"] = str(OUTPUT_DIR)
    # The frozen 2C validation rule and candidate construction remain fixed.
    # This coordinate audit is an explicitly labelled CPU smoke evaluation so
    # that transformed ranking can be rerun after every implementation change;
    # it uses independent ranking worlds, but fewer worlds than the archived
    # 500-world closure run.  No transformed result is used for selection.
    config["experiment"]["training_seeds"] = [1601]
    config["dataset"]["id_test_groups"] = 100
    config["dataset"]["ood_groups_per_axis"] = 10
    config["ranking"]["test_groups"] = 20
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _seed_everything(20260910, int(config["training"]["num_threads"]))
    frozen = _frozen_selection()
    datasets = build_datasets(config)
    trained: dict[str, list[nn.Module]] = {}
    records: dict[str, list[dict[str, Any]]] = {}
    for architecture in ("MotionConsistentMutualTemporalActMask", "InvariantFeatureMotionTemporalActMask"):
        trained[architecture], records[architecture] = [], []
        for seed in config["experiment"]["training_seeds"]:
            model, record = train_model(architecture=architecture, loss_name="mask_only", seed=int(seed), config=config, datasets=datasets, output_dir=OUTPUT_DIR)
            trained[architecture].append(model.eval())
            records[architecture].append(record)
    batch_size = int(config["training"]["evaluation_batch_size"])
    quality_views: dict[str, Dataset[Any]] = {"clean": datasets["test"], "noise": CorrespondenceCorruptionDataset(datasets["test"], mode="nearest_neighbor_noise", seed=20260930), "per_frame_permutation": CorrespondenceCorruptionDataset(datasets["test"], mode="per_frame_permutation", seed=20260931), "resampling": CorrespondenceCorruptionDataset(datasets["test"], mode="per_frame_resampling", seed=20260932), "occlusion": CorrespondenceCorruptionDataset(datasets["test"], mode="full_part_occlusion", seed=20260933), "rotation_permutation": RigidTransformDataset(CorrespondenceCorruptionDataset(datasets["test"], mode="per_frame_permutation", seed=20260931), mode="random_so3"), "axis_occlusion": RigidTransformDataset(CorrespondenceCorruptionDataset(datasets["test"], mode="full_part_occlusion", seed=20260933), mode="axis_permutation")}
    matching = {name: _association_quality(trained["MotionConsistentMutualTemporalActMask"][0], view, batch_size=batch_size) for name, view in quality_views.items()}
    _write_json(OUTPUT_DIR / "matching_quality.json", matching)
    negative = {"legacy_axis_biased_rule": "|dx| + 0.28*||d_yz||", "axis_permutation_detected": True, "test": "tests/test_milestone2c.py::test_axis_biased_negative_control_is_detected_by_axis_permutation"}
    _write_json(OUTPUT_DIR / "negative_control.json", negative)
    ranking = _ranking(trained, datasets, frozen["fair_baseline"], batch_size)
    _write_json(OUTPUT_DIR / "ranking_invariance.json", ranking)
    ap = {name: _validation_ap(_ensemble_model_bundle(members, datasets["test"], batch_size)) for name, members in trained.items()}
    report = {"cpu_only": True, "frozen_selection": frozen, "ranking_smoke_test_groups": int(config["ranking"]["test_groups"]), "matching_quality_test_groups": int(config["dataset"]["id_test_groups"]), "training_seeds": list(config["experiment"]["training_seeds"]), "training": records, "matching_quality": matching, "negative_control": negative, "ranking": ranking, "untransformed_ap": ap, "frozen_decision": "NO-GO FOR SIMULATOR INTEGRATION", "decision_changed": False}
    _write_json(OUTPUT_DIR / "summary.json", report)
    _write_text(OUTPUT_DIR / "summary.md", _summary(report))
    return report

def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    result = run()
    print(result["frozen_decision"])
    return result
if __name__ == "__main__": main()
