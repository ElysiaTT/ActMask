# Milestone 5B-Data handoff

All repaired-probe outputs are isolated under
`outputs/actmask/milestone5b_data_observable_cues/`; no 5B-0, 5B, or 5B-F
artifact is modified.

Key files are `moment_cue_matching_audit.json`,
`fair_identifiability_audit.json`, `candidate_diversity_audit.json`,
`baseline_report.json`, `raw_score_manifest.json`, `ci_consistency_audit.json`,
`headroom_metrics.json`, and `final_decision.json`. Learned baseline artifacts
are under `learned/` and include checkpoints, configurations, data hash,
command, logs, raw scores, and metrics for seeds 17/29/43.

Run the non-heavy audit with:

```bash
/home/tzh/conda_envs/actmask/bin/python -m pytest -q \
  tests/test_milestone5b_data_observable_cues.py
```

The next redesign must not reuse this probe as a method-evaluation benchmark:
the standard temporal baselines already saturate the oracle.
