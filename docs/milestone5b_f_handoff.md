# Milestone 5B-F handoff

All new forensic artifacts are isolated under
`outputs/actmask/milestone5b_f_plateau_forensics/`; 5B-0 and 5B remain
read-only sources.

Primary artifacts are `source_inventory.json`, `per_task_breakdown.json`,
`error_overlap_report.json`, `fair_observation_identifiability.json`,
`action_conditioning_sensitivity.json`, `ablation_equivalence_audit.json`,
`oracle_field_minimality.json`, `metric_split_sanity.json`, and
`final_decision.json`.

Run the non-heavy verification with:

```bash
/home/tzh/conda_envs/actmask/bin/python -m pytest -q \
  tests/test_milestone5b_f_plateau_forensics.py
```

The next milestone must redesign an observable identity/contact cue rather
than adding a new fair model to the currently unidentifiable probe.
