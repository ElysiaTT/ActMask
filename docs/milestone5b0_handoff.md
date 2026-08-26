# Milestone 5B-0 handoff

All new data and results are isolated under
`outputs/actmask/milestone5b0_relational_probe/`.

Primary handoff files are `state_probe/data_manifest.json`,
`moment_matching_audit.json`, `candidate_diversity_audit.json`,
`state_probe_baseline_report.json`, `raw_score_manifest.json`,
`ci_consistency_audit.json`, `headroom_metrics.json`, and `final_decision.json`.
The four standard learned controls each have three seed directories under
`learned/`; each contains `checkpoint.pt`, configs, preprocessing snapshot,
data hash, exact command, raw scores, and metrics.

To check the completed non-heavy audit:

```bash
/home/tzh/conda_envs/actmask/bin/python -m pytest -q tests/test_milestone5b0_relational_probe.py
```

To regenerate the no-render GPU-PhysX data on this host, retain the working
Vulkan ICD environment needed by SAPIEN scene construction:

```bash
VK_ICD_FILENAMES=/etc/vulkan/icd.d/test_nvidia_icd.json XDG_RUNTIME_DIR=/tmp \
  /home/tzh/conda_envs/actmask/bin/python -m actmask.data.milestone5b0_state_probe
```
