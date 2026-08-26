"""Derived preregistered robustness/headroom measures from v3 baseline output."""
from __future__ import annotations
import json
from pathlib import Path

def _score(table, regime):
    return float(table[regime]["pair_order_accuracy"])

def run(root: str | Path) -> dict:
    root=Path(root); report=json.loads((root/"standard_baseline_report.json").read_text()); output={"schema":"milestone4r-v3-derived-headroom-v1","metric":"pair_order_accuracy","tasks":{}}
    for task,value in report["tasks"].items():
        fair=value["fair"]; oracle=value["oracle"]; association=value["association"]; geometry=fair["world_pointcloud_gru"]; ordered=fair["ordered_rgbd_gru"]; unordered=fair["unordered_rgbd"]
        best_name,best=max(((name,_score(item,"combined_held_dropout")) for name,item in fair.items()),key=lambda item:item[1])
        output["tasks"][task]={
          "cross_view_retention_world_pointcloud_gru": _score(geometry,"held_camera_ood")/_score(geometry,"multi_camera_id"),
          "partial_observation_retention_world_pointcloud_gru": _score(geometry,"combined_held_dropout")/max(_score(geometry,"complete_history"),1e-8),
          "ordered_temporal_advantage_rgbd": _score(ordered,"combined_held_dropout")-_score(unordered,"combined_held_dropout"),
          "geometric_normalization_advantage": _score(geometry,"combined_held_dropout")-_score(ordered,"combined_held_dropout"),
          "association_gap_oracle_segmentation_minus_soft_geometric": _score(oracle["oracle_segmentation_pointcloud_gru"],"combined_held_dropout")-_score(association["association_soft_geometric"],"combined_held_dropout"),
          "model_headroom_state_minus_best_standard_fair": _score(oracle["clean_state_history_gru"],"combined_held_dropout")-best,
          "best_standard_fair": best_name,
          "best_standard_fair_score": best,
          "state_oracle_score": _score(oracle["clean_state_history_gru"],"combined_held_dropout")
        }
    (root/"derived_headroom_metrics.json").write_text(json.dumps(output,indent=2,sort_keys=True)+"\n");return output
