# Milestone 3R-NL-v2 Handoff

## Completed v2 probe artifacts

- `outputs/actmask/milestone3r_nl_v2/v1_forensic_audit.json`
- `outputs/actmask/milestone3r_nl_v2/preregistered_config.json`
- `outputs/actmask/milestone3r_nl_v2/probe_final/` (ID plus three OOD generators)
- `outputs/actmask/milestone3r_nl_v2/evaluation/candidate_diversity_audit.json`
- `outputs/actmask/milestone3r_nl_v2/evaluation/probe_evaluation.json`
- `outputs/actmask/milestone3r_nl_v2/evaluation/ood_overlap_audit.json`
- `outputs/actmask/milestone3r_nl_v2/probe_decision.json`
- `outputs/actmask/milestone3r_nl_v2/full_decision.json`

The generator is `actmask/experiments/milestone3r_nl_v2.py`; the evaluator is
`actmask/experiments/milestone3r_nl_v2_probe.py`.  Fair model input consists
only of observable history, timestamps, visibility/confidence, candidate
actions, nominal timing and TCP state.  Hidden parameters and identifiers are
metadata/audit-only.

## Full-run completion

The frozen full configuration is `outputs/actmask/milestone3r_nl_v2/full_run_config.json`
(SHA-256 `12c692d032ba37bdb14761a1c63ff8ceccff72af5f037d936ebaed8748016a47`).
It completed 92,000 additional GPU-PhysX branch executions and five-seed GRU
confirmation. The authoritative result is
`outputs/actmask/milestone3r_nl_v2/full_evaluation/full_evaluation.json`; all
final nonlinear-OOD gates pass. C5/C10 prefixes are in `full/id_c5/` and
`full/id_c10/`, derived from C20 without additional execution.

Do not change v1, the v2 probe configuration, or the full-run configuration.
Any visual work requires separate human authorization.

## Migrated environment

Use `/home/tzh/conda_envs/actmask/bin/python`. Its stale package links were
repaired to the matching Torch/CUDA payload in `/home/tzh/conda_envs/papers`;
do not revert those links to the removed `/data/env/...` prefix.

## Legacy visual status

`legacy_visual_test_status = not_present_in_collection`: no separately named
high-DPI/legacy-visual test node exists in this checkout's pytest collection.
It was not treated as a state-only scientific gate, and no renderer artifact
was modified.
