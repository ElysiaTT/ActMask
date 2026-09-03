# Experiment status and ownership boundary

Date: 2026-09-03

The repository contains several generations of ActMask experiments. Their
presence does not mean they are all active or positive results. Paths remain in
place to preserve old reports and tests; this document is the logical archive
boundary used for remote work.

## Active track

The active research track is the counterfactual action-effect binding audit:

- reusable evaluator: `binding_bench/`;
- entry scripts: `scripts/run_binding_bench_example.py`,
  `evaluate_binding_bench.py`, and `evaluate_binding_benchmark.py`;
- data/source probe: `scripts/probe_binding_datasets.py`;
- readiness checks: `scripts/audit_binding_readiness.py`,
  `scripts/check_binding_gpu_readiness.py`, and
  `scripts/check_remote_migration.py`;
- frozen design: `docs/counterfactual_binding_benchmark_v1.md` and
  `docs/binding_gpu_preregister.json`.

The active CPU harness is ready. The existing learned method is not: v2 and v3
remain terminal NO-GO results, while v1 is a saturation diagnostic.

## Historical evidence, retained but not active

The following files preserve how the current direction was reached:

- `scripts/cpu_validate_timearrow_direction.py` and
  `scripts/cpu_timearrow_v3.py`;
- `scripts/cpu_binding_pre_admission.py`;
- `scripts/cpu_binding_learned_v1.py`;
- `scripts/cpu_binding_v2_antisaturation.py`;
- `scripts/cpu_binding_v3_sysid_residual.py`;
- their preregistrations under `docs/`, reports under `audit/`, and JSON
  evidence under `audit/results/`.

Do not tune against or rerun these by default on the remote GPU host. They are
baselines and negative-result provenance, not the production entrypoint.

## Legacy repository material

Older `milestone*`, `rm_*`, `actioncheck`, paper, submission, and release
directories are retained to avoid breaking historical manuscripts and tests.
They are out of scope for the first binding GPU run unless a preregistered
comparison explicitly names them.

## Planned, not implemented

The G1-G4 modules under `actmask/experiments/binding_gpu/` do not yet exist.
Their commands in the GPU preregistration are a frozen interface, not evidence
of execution. Implementation begins only after a named Linux/NVIDIA host passes
G0 and the cost boundary is approved.
