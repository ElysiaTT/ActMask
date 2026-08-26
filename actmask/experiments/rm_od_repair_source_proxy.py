"""One permitted RM-OD repair: balance candidate source trajectory provenance."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from actmask.experiments.rm_od_build_task import K, NONROBOT, _audits, _l2, _load_payload, _nodes, _record, _split_definitions
from actmask.experiments.rm_od_freeze_and_inventory import OUT, append_log, write_json
from actmask.experiments.rm_od_metrics import group_metrics, score_rows


def _metrics(query: dict, candidate: dict) -> dict[str, float]:
    return {"action_l2": _l2(query["action"], candidate["action"]), "action_magnitude_abs": abs(float(np.linalg.norm(query["action"])) - float(np.linalg.norm(candidate["action"]))), "state_l2": _l2(query["pre_state"], candidate["pre_state"]), "pre_visual_l2": _l2(query["pre_patch"], candidate["pre_patch"]), "end_effector_displacement_l2": _l2(query["end_displacement"], candidate["end_displacement"]), "progress_abs": abs(query["progress"] - candidate["progress"]), "future_nonrobot_l2": _l2(query["future_patch"][:, NONROBOT], candidate["future_patch"][:, NONROBOT]), "future_global_cosine": 0.0, "matching_score": 0.0}


def _groups(nodes: list[dict], family: str, partition: str, serial: int) -> tuple[list[dict], int]:
    by_episode = {node["episode_id"]: [] for node in nodes}
    for node in nodes:
        by_episode[node["episode_id"]].append(node)
    rows = []
    for query in sorted(nodes, key=lambda node: node["node_id"]):
        alternatives = sorted([node for node in by_episode[query["episode_id"]] if node["node_id"] != query["node_id"]], key=lambda node: node["node_id"])
        if len(alternatives) != K - 1:
            raise ValueError(f"same-episode source-balancing requires exactly {K - 1} alternatives")
        positive = serial % K
        iterator = iter(alternatives)
        candidates = []
        for index in range(K):
            node, is_positive = (query, True) if index == positive else (next(iterator), False)
            metrics = {"action_l2": 0.0, "action_magnitude_abs": 0.0, "state_l2": 0.0, "pre_visual_l2": 0.0, "end_effector_displacement_l2": 0.0, "progress_abs": 0.0, "future_nonrobot_l2": 0.0, "future_global_cosine": 1.0, "matching_score": 0.0} if is_positive else _metrics(query, node)
            candidates.append({"candidate_index": index, "is_positive": is_positive, "candidate_node_id": node["node_id"], "candidate_future_episode": node["episode_id"], "candidate_future_anchor": node["anchor"], "candidate_future_start": node["future_start"], "candidate_source_provenance": {"parquet": node["record"].parquet_path, "front_video": node["record"].front_video_path}, "metrics": metrics})
        rows.append({"group_id": f"{family}:{partition}:repair1:{serial:05d}", "split_family": family, "partition": partition, "anchor_episode": query["episode_id"], "anchor_step": query["anchor"], "anchor_node_id": query["node_id"], "task": query["task"], "candidate_action_start": query["anchor"], "positive_index": positive, "anchor_source_provenance": {"parquet": query["record"].parquet_path, "front_video": query["record"].front_video_path}, "candidates": candidates})
        serial += 1
    return rows, serial


def run() -> dict:
    precheck = json.loads((OUT / "initial_shortcut_precheck.json").read_text())
    if precheck["decision"] != "RM_OD_ACTION_PATH_SHORTCUT":
        raise RuntimeError("source-proxy repair is allowed only after the documented initial shortcut failure")
    selected = json.loads((OUT / "selected_trajectory_manifest.json").read_text())["episodes"]
    records = [_record(row) for row in selected]
    payload, _ = _load_payload(records)
    splits = _split_definitions(records)
    all_nodes: dict[str, dict] = {}
    all_groups: list[dict] = []
    serial = 0
    for family, definition in splits.items():
        for partition, episode_ids in definition.items():
            nodes = _nodes(payload, episode_ids, family, partition)
            all_nodes.update({node["node_id"]: node for node in nodes})
            groups, serial = _groups(nodes, family, partition, serial)
            all_groups.extend(groups)
    path = OUT / "task_groups.jsonl"
    path.write_text("".join(json.dumps(group, sort_keys=True) + "\n" for group in all_groups))
    manifest = json.loads((OUT / "task_manifest.json").read_text())
    manifest.update({"groups": len(all_groups), "hard_negative_repair": {"repair_number": 1, "reason": "full candidate source path achieved Recall@1=1.0", "new_policy": "all K candidate futures are different anchors from the anchor episode; source trajectory/path is constant within every group", "no_config_gate_changed": True}})
    write_json(OUT / "task_manifest.json", manifest)
    _audits(all_groups, all_nodes, splits)
    matching = json.loads((OUT / "hard_negative_matching_audit.json").read_text())
    matching.update({"negative_source_variant": "same_episode_different_anchor", "source_path_balanced_within_group": True, "repair_number": 1, "cross_episode_negatives": False})
    write_json(OUT / "hard_negative_matching_audit.json", matching)
    candidate = json.loads((OUT / "candidate_group_audit.json").read_text())
    candidate["candidate_source_path_constant_within_group"] = all(all(value["candidate_source_provenance"]["parquet"] == group["anchor_source_provenance"]["parquet"] for value in group["candidates"]) for group in all_groups)
    write_json(OUT / "candidate_group_audit.json", candidate)
    repair = {}
    for family in sorted({group["split_family"] for group in all_groups}):
        repair[family] = {}
        for partition in ("train", "validation", "test"):
            groups = [group for group in all_groups if group["split_family"] == family and group["partition"] == partition]
            rows = score_rows(groups, lambda group: [1.0 if value["candidate_source_provenance"]["parquet"] == group["anchor_source_provenance"]["parquet"] else 0.0 for value in group["candidates"]])
            repair[family][partition] = group_metrics(rows)
    write_json(OUT / "repair1_source_path_precheck.json", {"schema": "rm-od-repair1-source-path-precheck-v1", "source_path_proxy": repair, "max_test_recall_at_1": max(value["test"]["recall_at_1"] for value in repair.values()), "threshold": 0.225, "pass": max(value["test"]["recall_at_1"] for value in repair.values()) <= 0.225})
    append_log("phase2_repair1_source_proxy", groups=len(all_groups), source_path_test_recall_at_1=max(value["test"]["recall_at_1"] for value in repair.values()))
    return {"groups": len(all_groups), "source_path_test_recall_at_1": max(value["test"]["recall_at_1"] for value in repair.values())}


if __name__ == "__main__":
    print(json.dumps(run(), sort_keys=True))
