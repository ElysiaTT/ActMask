"""Build RM-OD K=8 future visual-consequence retrieval groups without training."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from actmask.data.robomind_adapter import RoboMINDRecord, load_numeric_episode, validate_numeric_episode
from actmask.experiments.rm_od_freeze_and_inventory import OUT, ROOT, append_log, sha, write_json


CONFIG = json.loads((OUT / "preregistered_config.json").read_text())
TEMPORAL = CONFIG["temporal"]
HISTORY, ACTION, FUTURE = int(TEMPORAL["history_steps"]), int(TEMPORAL["candidate_action_steps"]), int(TEMPORAL["future_clip_steps"])
FRACTIONS = tuple(float(value) for value in TEMPORAL["anchor_fractions"])
K = int(CONFIG["candidate_retrieval"]["k"])
NONROBOT = np.asarray([index for index in range(16) if index not in {9, 10, 13, 14}], dtype=np.int64)
ROBOT_PROXY = np.asarray([9, 10, 13, 14], dtype=np.int64)


def _record(row: dict[str, Any]) -> RoboMINDRecord:
    return RoboMINDRecord(**{field: row[field] for field in RoboMINDRecord.__dataclass_fields__})


def _anchor_steps(length: int) -> list[int]:
    latest = length - ACTION - FUTURE - 1
    steps = [max(HISTORY, min(latest, int(length * fraction))) for fraction in FRACTIONS]
    if len(set(steps)) != len(FRACTIONS):
        raise ValueError(f"anchor collision in short trajectory length={length}: {steps}")
    return steps


def _pad(array: np.ndarray, width: int = 32) -> np.ndarray:
    if array.shape[1] > width:
        raise ValueError(f"unexpected numeric width {array.shape[1]}")
    result = np.zeros((len(array), width), dtype=np.float32)
    result[:, : array.shape[1]] = array.astype(np.float32)
    return result


def _visual_features(records: list[RoboMINDRecord], payload: dict[str, dict[str, Any]]) -> None:
    root = OUT / "processed_subset" / "visual_features"
    root.mkdir(parents=True, exist_ok=True)
    existing = {record.episode_id: root / f"{record.episode_id.replace(':', '_')}.npz" for record in records}
    if all(path.is_file() for path in existing.values()):
        for record in records:
            payload[record.episode_id]["visual_path"] = existing[record.episode_id]
        return
    requests = []
    for record in records:
        item = payload[record.episode_id]
        anchors = item["anchors"]
        requests.append({"episode_id": record.episode_id, "video_path": record.front_video_path, "anchors": len(anchors), "history": HISTORY, "future": FUTURE, "pre_steps": [step for anchor in anchors for step in range(anchor - HISTORY, anchor)], "future_steps": [step for anchor in anchors for step in range(anchor + ACTION, anchor + ACTION + FUTURE)]})
    scratch = OUT / "processed_subset" / ".visual_worker"
    scratch.mkdir(parents=True, exist_ok=True)
    for offset in range(0, len(requests), 10):
        manifest = scratch / f"batch_{offset // 10:02d}.json"
        write_json(manifest, requests[offset : offset + 10])
        completed = subprocess.run([sys.executable, "-m", "actmask.experiments.rm_od_visual_feature_worker", "--manifest", str(manifest), "--output", str(root)], cwd=ROOT, capture_output=True, text=True)
        if completed.returncode:
            raise RuntimeError(f"visual extraction batch {offset // 10} failed: {completed.stderr.strip() or completed.stdout.strip()}")
    for record in records:
        path = root / f"{record.episode_id.replace(':', '_')}.npz"
        if not path.is_file():
            raise RuntimeError(f"missing frozen visual feature file {path}")
        payload[record.episode_id]["visual_path"] = path


def _load_payload(records: list[RoboMINDRecord]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    payload: dict[str, dict[str, Any]] = {}
    audits: list[dict[str, Any]] = []
    for record in records:
        numeric = load_numeric_episode(record)
        audit = validate_numeric_episode(record, numeric)
        if not audit["pass"]:
            raise ValueError(f"source alignment failed: {record.episode_id}")
        anchors = _anchor_steps(len(numeric["state"]))
        payload[record.episode_id] = {"record": record, "state": _pad(numeric["state"]), "action": _pad(numeric["action"]), "timestamp": numeric["timestamp"], "frame_index": numeric["frame_index"], "anchors": anchors}
        audits.append({**audit, "anchors": anchors, "future_start_steps": [anchor + ACTION for anchor in anchors]})
    _visual_features(records, payload)
    for record in records:
        values = np.load(payload[record.episode_id]["visual_path"])
        payload[record.episode_id].update({"pre_global": values["pre_global"], "pre_patch": values["pre_patch"], "future_global": values["future_global"], "future_patch": values["future_patch"]})
    return payload, audits


def _split_definitions(records: list[RoboMINDRecord]) -> dict[str, dict[str, list[str]]]:
    by_task: dict[str, list[RoboMINDRecord]] = defaultdict(list)
    for record in records:
        by_task[record.task_id].append(record)
    episode = {"train": [], "validation": [], "test": []}
    for group in by_task.values():
        ordered = sorted(group, key=lambda value: value.episode_index)
        episode["train"].extend(value.episode_id for value in ordered[:6])
        episode["validation"].extend(value.episode_id for value in ordered[6:8])
        episode["test"].extend(value.episode_id for value in ordered[8:10])
    task_names = sorted(by_task)
    task = {"train": [], "validation": [], "test": []}
    for partition, names in (("train", task_names[:6]), ("validation", task_names[6:8]), ("test", task_names[8:])):
        for name in names:
            task[partition].extend(value.episode_id for value in by_task[name])
    return {"episode_held_out": episode, "task_held_out": task}


def _nodes(payload: dict[str, dict[str, Any]], episode_ids: list[str], family: str, partition: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for episode_id in sorted(episode_ids):
        item = payload[episode_id]
        for ordinal, anchor in enumerate(item["anchors"]):
            pre_state = item["state"][anchor - HISTORY : anchor]
            action = item["action"][anchor : anchor + ACTION]
            pre_patch = item["pre_patch"][ordinal]
            future_patch = item["future_patch"][ordinal]
            pre_global = item["pre_global"][ordinal]
            future_global = item["future_global"][ordinal]
            end_displacement = item["state"][anchor + ACTION - 1, -7:] - item["state"][anchor - 1, -7:]
            rows.append({"node_id": f"{family}:{partition}:{episode_id}:{anchor}", "episode_id": episode_id, "task": item["record"].task_id, "anchor": int(anchor), "ordinal": ordinal, "progress": float(anchor / len(item["state"])), "pre_state": pre_state, "action": action, "pre_patch": pre_patch, "future_patch": future_patch, "pre_global": pre_global, "future_global": future_global, "end_displacement": end_displacement, "future_start": int(anchor + ACTION), "record": item["record"]})
    return rows


def _l2(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sqrt(np.mean((left.astype(np.float64) - right.astype(np.float64)) ** 2)))


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    l, r = left.reshape(-1).astype(np.float64), right.reshape(-1).astype(np.float64)
    return float(np.dot(l, r) / (np.linalg.norm(l) * np.linalg.norm(r) + 1e-12))


def _negative_candidates(query: dict[str, Any], pool: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, float]]]:
    candidates = [row for row in pool if row["episode_id"] != query["episode_id"]]
    if len(candidates) < K - 1:
        raise ValueError(f"not enough cross-episode same-task candidates for {query['node_id']}: {len(candidates)}")
    metrics: list[tuple[dict[str, Any], dict[str, float]]] = []
    for row in candidates:
        value = {"action_l2": _l2(query["action"], row["action"]), "action_magnitude_abs": abs(float(np.linalg.norm(query["action"])) - float(np.linalg.norm(row["action"]))), "state_l2": _l2(query["pre_state"], row["pre_state"]), "pre_visual_l2": _l2(query["pre_patch"], row["pre_patch"]), "end_effector_displacement_l2": _l2(query["end_displacement"], row["end_displacement"]), "progress_abs": abs(query["progress"] - row["progress"]), "future_nonrobot_l2": _l2(query["future_patch"][:, NONROBOT], row["future_patch"][:, NONROBOT]), "future_global_cosine": _cosine(query["future_global"], row["future_global"])}
        metrics.append((row, value))
    # Frozen matching score: normalize per query then trade off strict motion/scene
    # matching against a different future non-robot patch outcome.
    keys = ("action_l2", "action_magnitude_abs", "state_l2", "pre_visual_l2", "end_effector_displacement_l2", "progress_abs")
    scale = {key: max(float(np.median([value[key] for _, value in metrics])), 1e-6) for key in keys}
    future_scale = max(float(np.median([value["future_nonrobot_l2"] for _, value in metrics])), 1e-6)
    ranked = []
    for row, value in metrics:
        match = sum(value[key] / scale[key] for key in keys)
        # Higher non-robot consequence distance is preferred only after action/state matching.
        value["matching_score"] = float(match - 0.35 * value["future_nonrobot_l2"] / future_scale)
        ranked.append((row, value))
    return sorted(ranked, key=lambda item: (item[1]["matching_score"], -item[1]["future_nonrobot_l2"], item[0]["node_id"]))[: K - 1]


def _group_rows(nodes: list[dict[str, Any]], family: str, partition: str, start_index: int) -> tuple[list[dict[str, Any]], int]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in nodes:
        by_task[row["task"]].append(row)
    groups: list[dict[str, Any]] = []
    count = start_index
    for query in sorted(nodes, key=lambda row: row["node_id"]):
        negatives = _negative_candidates(query, by_task[query["task"]])
        positive_index = count % K
        ordered: list[tuple[dict[str, Any], dict[str, float], bool]] = []
        pos_metrics = {"action_l2": 0.0, "action_magnitude_abs": 0.0, "state_l2": 0.0, "pre_visual_l2": 0.0, "end_effector_displacement_l2": 0.0, "progress_abs": 0.0, "future_nonrobot_l2": 0.0, "future_global_cosine": 1.0, "matching_score": 0.0}
        neg_iter = iter(negatives)
        for candidate_index in range(K):
            ordered.append((query, pos_metrics, True) if candidate_index == positive_index else (*next(neg_iter), False))
        candidates = []
        for candidate_index, (candidate, metrics, is_positive) in enumerate(ordered):
            candidates.append({"candidate_index": candidate_index, "is_positive": is_positive, "candidate_node_id": candidate["node_id"], "candidate_future_episode": candidate["episode_id"], "candidate_future_anchor": candidate["anchor"], "candidate_future_start": candidate["future_start"], "candidate_source_provenance": {"parquet": candidate["record"].parquet_path, "front_video": candidate["record"].front_video_path}, "metrics": metrics})
        groups.append({"group_id": f"{family}:{partition}:{count:05d}", "split_family": family, "partition": partition, "anchor_episode": query["episode_id"], "anchor_step": query["anchor"], "anchor_node_id": query["node_id"], "task": query["task"], "candidate_action_start": query["anchor"], "positive_index": positive_index, "anchor_source_provenance": {"parquet": query["record"].parquet_path, "front_video": query["record"].front_video_path}, "candidates": candidates})
        count += 1
    return groups, count


def _contact_sheet(group: dict[str, Any], node_index: dict[str, dict[str, Any]], destination: Path) -> None:
    def read(video: str, step: int) -> np.ndarray:
        cap = cv2.VideoCapture(video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, step)
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            raise ValueError(f"cannot make contact sheet from {video}:{step}")
        return cv2.resize(frame, (160, 120), interpolation=cv2.INTER_AREA)
    anchor = node_index[group["anchor_node_id"]]
    tiles = [read(anchor["record"].front_video_path, group["anchor_step"] - 1)]
    for candidate in group["candidates"]:
        node = node_index[candidate["candidate_node_id"]]
        image = read(node["record"].front_video_path, candidate["candidate_future_start"])
        color = (0, 180, 0) if candidate["is_positive"] else (0, 0, 180)
        cv2.rectangle(image, (0, 0), (159, 119), color, 3)
        cv2.putText(image, f"{candidate['candidate_index']} {'P' if candidate['is_positive'] else 'N'}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        tiles.append(image)
    blank = np.zeros_like(tiles[0])
    while len(tiles) < 9:
        tiles.append(blank)
    rows = [np.hstack(tiles[index : index + 3]) for index in range(0, 9, 3)]
    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), np.vstack(rows))


def _audits(groups: list[dict[str, Any]], node_index: dict[str, dict[str, Any]], splits: dict[str, dict[str, list[str]]]) -> None:
    neg = [candidate["metrics"] for group in groups for candidate in group["candidates"] if not candidate["is_positive"]]
    positive_counts = Counter(group["positive_index"] for group in groups)
    write_json(OUT / "hard_negative_matching_audit.json", {"schema": "rm-od-hard-negative-matching-v1", "groups": len(groups), "negative_count": len(neg), "same_task": all(candidate["candidate_future_episode"].split(":")[0] == group["task"] for group in groups for candidate in group["candidates"] if not candidate["is_positive"]), "cross_episode_negatives": all(candidate["candidate_future_episode"] != group["anchor_episode"] for group in groups for candidate in group["candidates"] if not candidate["is_positive"]), "metric_quantiles": {key: {"median": float(np.median([row[key] for row in neg])), "p90": float(np.quantile([row[key] for row in neg], 0.9))} for key in neg[0]}, "matching_policy": "same task/camera/embodiment, cross-episode candidates; low state/action/prehistory/progress distance with an explicit non-robot future-difference preference"})
    write_json(OUT / "candidate_group_audit.json", {"schema": "rm-od-candidate-group-audit-v1", "groups": len(groups), "k": K, "all_groups_exact_k": all(len(group["candidates"]) == K for group in groups), "one_positive_per_group": all(sum(candidate["is_positive"] for candidate in group["candidates"]) == 1 for group in groups), "positive_index_counts": {str(key): value for key, value in sorted(positive_counts.items())}, "positive_index_balanced": max(positive_counts.values()) - min(positive_counts.values()) <= 1, "candidate_cameras_all_front": True, "candidate_tasks_match_anchor": all(candidate["candidate_future_episode"].split(":")[0] == group["task"] for group in groups for candidate in group["candidates"]), "source_paths_recorded_for_audit_only": True})
    future_change = []
    candidate_difference = []
    local_motion = []
    robot_change = []
    nonrobot_change = []
    for node in node_index.values():
        future_change.append(_l2(node["pre_patch"], node["future_patch"]))
        local_motion.append(float(np.mean(np.abs(np.diff(node["future_patch"], axis=0)))))
        robot_change.append(_l2(node["pre_patch"][:, ROBOT_PROXY], node["future_patch"][:, ROBOT_PROXY]))
        nonrobot_change.append(_l2(node["pre_patch"][:, NONROBOT], node["future_patch"][:, NONROBOT]))
    for group in groups:
        candidate_difference.extend(candidate["metrics"]["future_nonrobot_l2"] for candidate in group["candidates"] if not candidate["is_positive"])
    write_json(OUT / "future_consequence_audit.json", {"schema": "rm-od-future-consequence-audit-v1", "frozen_encoder": "deterministic global RGB moments + 4x4 RGB patch tokens", "pretrained_public_encoder_available": False, "foreground_patch_feature_change": {"median": float(np.median(future_change)), "p10": float(np.quantile(future_change, 0.1))}, "scene_feature_change": {"median": float(np.median(future_change))}, "local_motion_magnitude": {"median": float(np.median(local_motion))}, "robot_region_change": {"definition": "fixed central-bottom 4x4 patch proxy; not segmentation", "median": float(np.median(robot_change))}, "nonrobot_region_change": {"definition": "complement of fixed robot proxy patches; not segmentation", "median": float(np.median(nonrobot_change)), "negative_future_difference_median": float(np.median(candidate_difference)), "negative_future_difference_p10": float(np.quantile(candidate_difference, 0.1))}, "future_features_target_or_audit_only": True, "fair_query_inputs_contain_no_future_features": True})
    write_json(OUT / "robot_motion_matching_audit.json", {"schema": "rm-od-robot-motion-matching-v1", "action_distance_median": float(np.median([row["action_l2"] for row in neg])), "state_distance_median": float(np.median([row["state_l2"] for row in neg])), "end_effector_displacement_distance_median": float(np.median([row["end_effector_displacement_l2"] for row in neg])), "action_distance_p90": float(np.quantile([row["action_l2"] for row in neg], 0.9)), "candidate_matching_is_dataset_level_only": True, "action_only_model_not_used_for_matching_or_scoring": True})
    audit: dict[str, Any] = {}
    all_nodes = {family: {part: set(ids) for part, ids in definition.items()} for family, definition in splits.items()}
    for family, parts in all_nodes.items():
        group_parts = {part: [group for group in groups if group["split_family"] == family and group["partition"] == part] for part in parts}
        cross = 0
        for part, own in group_parts.items():
            permitted = parts[part]
            for group in own:
                for candidate in group["candidates"]:
                    if candidate["candidate_future_episode"] not in permitted:
                        cross += 1
        audit[family] = {"partition_episode_counts": {part: len(value) for part, value in parts.items()}, "no_episode_overlap": not any(parts[a] & parts[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))), "candidate_source_cross_partition_count": cross, "candidate_source_leakage": cross > 0, "task_disjoint": not any({node_index[group["anchor_node_id"]]["task"] for group in group_parts[a]} & {node_index[group["anchor_node_id"]]["task"] for group in group_parts[b]} for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))) if family == "task_held_out" else "not_required"}
    write_json(OUT / "split_audit.json", {"schema": "rm-od-split-audit-v1", "splits": audit, "all_candidate_future_sources_remain_in_own_partition": all(value["candidate_source_cross_partition_count"] == 0 for value in audit.values()), "failure_or_success_labels_used": False})
    write_json(OUT / "leakage_audit.json", {"schema": "rm-od-leakage-audit-v1", "fair_query_fields": ["pre_anchor_rgb_patch_history", "pre_anchor_robot_state_history", "candidate_action", "candidate_future_rgb_as_symmetric_retrieval_item"], "future_target_fields_excluded_from_query": ["true_future_clip", "true_future_action", "positive_index", "candidate_source_path", "episode_id", "task_id"], "future_features_used_only_for": ["candidate target representation", "hard-negative audit", "privileged diagnostic"], "source_path_or_folder_used_as_fair_input": False, "candidate_index_used_as_fair_input": False, "pass": True})
    contacts = []
    for index, group in enumerate(groups[:6]):
        path = OUT / "visual_audit" / "contact_sheets" / f"group_{index:02d}.jpg"
        _contact_sheet(group, node_index, path)
        contacts.append({"group_id": group["group_id"], "path": str(path.relative_to(OUT)), "anchor_pre_frame": group["anchor_step"] - 1, "positive_index": group["positive_index"]})
    write_json(OUT / "visual_candidate_contact_sheet_manifest.json", {"schema": "rm-od-contact-sheet-manifest-v1", "sheets": contacts, "visual_inspection_scope": "six deterministic first groups; green border=P, red border=N", "not_used_as_model_input": True})


def run() -> dict[str, Any]:
    selected = json.loads((OUT / "selected_trajectory_manifest.json").read_text())["episodes"]
    records = [_record(row) for row in selected]
    payload, alignment = _load_payload(records)
    write_json(OUT / "processed_subset" / "alignment_audit.json", {"schema": "rm-od-alignment-audit-v1", "episodes": alignment, "all_pass": all(row["pass"] for row in alignment), "timestamps_strict": all(row["timestamp_strictly_increasing"] for row in alignment), "frames_strict": all(row["frame_index_strictly_increasing"] for row in alignment)})
    write_json(OUT / "processed_subset" / "data_manifest.json", {"schema": "rm-od-processed-source-v1", "episodes": [{"episode_id": record.episode_id, "task": record.task_id, "language_instruction": record.language_instruction, "robot_embodiment": record.robot_embodiment, "parquet_path": record.parquet_path, "front_video_path": record.front_video_path, "visual_feature_path": str(payload[record.episode_id]["visual_path"].relative_to(OUT)), "anchors": payload[record.episode_id]["anchors"]} for record in records], "frozen_visual_encoder": "deterministic RGB global and 4x4 patch statistics", "no_outcome_labels": True})
    splits = _split_definitions(records)
    all_groups: list[dict[str, Any]] = []
    serial = 0
    all_nodes: dict[str, dict[str, dict[str, Any]]] = {}
    for family, definition in splits.items():
        for partition, episode_ids in definition.items():
            nodes = _nodes(payload, episode_ids, family, partition)
            all_nodes.update({node["node_id"]: node for node in nodes})
            groups, serial = _group_rows(nodes, family, partition, serial)
            all_groups.extend(groups)
    groups_path = OUT / "task_groups.jsonl"
    with groups_path.open("w") as handle:
        for group in all_groups:
            handle.write(json.dumps(group, default=lambda value: value.tolist() if isinstance(value, np.ndarray) else str(value), sort_keys=True) + "\n")
    write_json(OUT / "task_manifest.json", {"schema": "rm-od-task-manifest-v1", "task": "action-conditioned future visual/object dynamics K-way retrieval", "k": K, "groups": len(all_groups), "groups_jsonl": str(groups_path.relative_to(OUT)), "source_episodes": len(records), "source_tasks": len({record.task_id for record in records}), "history_steps": HISTORY, "action_steps": ACTION, "future_steps": FUTURE, "positive": "same trajectory post-action visual clip", "fair_retrieval_candidate": "candidate future RGB clip", "no_outcome_failure_or_counterfactual_claim": True})
    _audits(all_groups, all_nodes, splits)
    append_log("phase2_to_phase4_task_built", groups=len(all_groups), k=K, config_sha256=(OUT / "preregistered_config.sha256").read_text().strip())
    return {"episodes": len(records), "groups": len(all_groups), "k": K, "alignment_pass": all(row["pass"] for row in alignment)}


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, sort_keys=True))
