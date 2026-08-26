"""Inference-only R2 must stop rather than replace missing 4R-v3 checkpoints."""
from __future__ import annotations
import hashlib,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs"/"actmask"/"milestone5a_r2_inference_metric_repair"
HASH="d9b15dd7fea42871252cb50118a3aa5ee9b3b073397171c3deff6fc63d54d565"

def test_r2_config_and_checkpoint_stop_are_frozen():
    assert hashlib.sha256((OUT/"preregistered_config.json").read_bytes()).hexdigest()==HASH
    inventory=json.loads((OUT/"artifact_inventory.json").read_text()); decision=json.loads((OUT/"milestone5a_r2_final_decision.json").read_text())
    assert inventory["decision"]=="B. CHECKPOINTS_MISSING"
    assert all(not row["exists"] for row in inventory["learned_model_artifacts"])
    assert decision["decision"]=="C. STOP_MISSING_ARTIFACTS"
    assert decision["raw_scores_generated"] is False
