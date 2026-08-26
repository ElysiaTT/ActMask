from __future__ import annotations

import json

import numpy as np

from actmask.data.milestone4v_visual_pilot import load_fair_visual_inputs


def test_shared_visual_history_loader_excludes_metadata_and_segmentation(tmp_path) -> None:
    np.savez_compressed(
        tmp_path / "MovingCubeIntercept_histories.npz",
        rgb=np.zeros((2, 6, 4, 4, 3), dtype=np.uint8),
        depth_mm=np.full((2, 6, 4, 4), 1000, dtype=np.uint16),
    )
    rows = [
        {"history_ref": 1, "world_id": 7, "candidate_slot": 3, "branch": 1,
         "success": True, "segmentation_id": 9, "candidate_actions": [[0, 0, 0]] * 12, "tcp_state": [0, 1, 2]}
    ]
    (tmp_path / "MovingCubeIntercept_candidates.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    fair = load_fair_visual_inputs(tmp_path, "MovingCubeIntercept")
    assert set(fair) == {"rgbd_history", "timestamps", "visibility", "observation_confidence", "candidate_actions", "nominal_action_timing", "tcp_state"}
    assert fair["rgbd_history"].shape == (1, 6, 4, 4, 4)
    assert np.allclose(fair["rgbd_history"][..., -1], 1.0)
    assert np.array_equal(fair["tcp_state"], [[0, 1, 2]])
