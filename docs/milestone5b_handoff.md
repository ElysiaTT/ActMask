# Milestone 5B handoff

All method outputs are isolated under
`outputs/actmask/milestone5b_relational_verifier/`. The 5B-0 source is read
only and verified by the frozen composite data SHA-256 in this milestone's
preregistered configuration.

Key artifacts are `method_report.json`, `comparison_to_5b0.json`,
`headroom_closure.json`, `ablation_report.json`, `input_field_audit.json`,
`ci_consistency_audit.json`, `raw_score_manifest.json`, and
`final_decision.json`. Full-model seed artifacts are under
`learned/RelDynVerifier/`; each includes its checkpoint, model/preprocessing/
training configuration, command, data hash, log, raw scores and metrics.

The non-heavy completion audit is:

```bash
/home/tzh/conda_envs/actmask/bin/python -m pytest -q \
  tests/test_milestone5b_relational_verifier.py
```

The negative conclusion is reproducible from saved test raw scores; do not
overwrite the source 5B-0 outputs or change the 5B acceptance gates.
