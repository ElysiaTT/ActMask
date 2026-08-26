"""The 4V renderer blocker must not mutate frozen state-only evidence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "outputs/actmask/milestone3r_nl_v2/full_run_config.json"
EXPECTED = "12c692d032ba37bdb14761a1c63ff8ceccff72af5f037d936ebaed8748016a47"
VISUAL_PROTOCOL = ROOT / "outputs/actmask/milestone4v_visual_pilot/visual_protocol.json"
VISUAL_PROTOCOL_HASH = "125f74eeb7ebae9a7074bf43614d982ef395c5ff1bb25b514d9d98c8642f7077"


def test_4v_preflight_qualifies_the_explicit_egl_icd_without_visual_fallback():
    audit = json.loads((ROOT / "outputs/actmask/milestone4v_visual_pilot/environment_audit.json").read_text())
    decision = json.loads((ROOT / "outputs/actmask/milestone4v_visual_pilot/milestone4v_decision.json").read_text())
    assert audit["status"] == "QUALIFIED"
    assert audit["renderer"]["qualified"] is True
    assert audit["renderer"]["required_runtime_environment"]["VK_ICD_FILENAMES"].endswith("test_nvidia_icd.json")
    assert decision["decision"] == "STAGE_A_QUALIFIED_PROCEED_TO_VISUAL_PROTOCOL"
    assert decision["visual_data_generated"] is False
    assert decision["milestone4m_authorized"] is False


def test_4v_preflight_keeps_frozen_v2_config_unchanged():
    assert hashlib.sha256(FROZEN.read_bytes()).hexdigest() == EXPECTED


def test_4v_visual_protocol_is_frozen_before_generation():
    protocol = json.loads(VISUAL_PROTOCOL.read_text())
    assert hashlib.sha256(VISUAL_PROTOCOL.read_bytes()).hexdigest() == VISUAL_PROTOCOL_HASH
    assert protocol["pilot_scale"]["base_worlds_per_task"] == 128
    assert protocol["pilot_scale"]["candidates_per_world"] == 10
    assert protocol["pilot_scale"]["history_frames"] == 6
    assert "segmentation_id" in protocol["forbidden_fair_fields"]
    assert protocol["runtime_environment"]["VK_ICD_FILENAMES"].endswith("test_nvidia_icd.json")
