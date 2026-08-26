# Milestone 5B-Data-F handoff

All saturation-forensic outputs are isolated under
`outputs/actmask/milestone5b_data_f_saturation_forensics/`; the 5B-Data bundle
remains read only.

Key reports are `source_inventory.json`, `saturation_breakdown.json`,
`cue_ablation_diagnostics.json`, `split_stress_diagnostics.json`,
`redesign_recommendation.json`, and `final_decision.json`. The two explicitly
diagnostic (not final-method) GRU split runs and their raw-score artifacts are
under `diagnostic_retrains/`.

Verify with:

```bash
/home/tzh/conda_envs/actmask/bin/python -m pytest -q \
  tests/test_milestone5b_data_f_saturation_forensics.py
```
