# ActMask

This repository contains the original ActMask action-conditioned future-mask
prototype and its subsequent audit experiments. The active research direction
is now the model-agnostic Counterfactual Action-Effect Binding Audit. Existing
v1/v2/v3 learned runs are diagnostic or NO-GO results, not a validated method
improvement.

The original mask prototype remains documented in
[README_actmask.md](README_actmask.md). The current repository boundary and
next-owner actions are in [HANDOFF.md](HANDOFF.md).

## Counterfactual binding audit

CPU G1 snapshot replay and G2 bounded physical twins are now implemented,
with an Ubuntu CPU CI workflow. See [the CPU smoke contract and commands](docs/binding_cpu_g1_g2.md).
This is transport validation, not a new model result or GPU authorization.
The [bounded hardening task](docs/cpu_transport_improvement_task.md) records the
follow-up prompt and acceptance checks for corrupted artifacts, joint effect
matching, snapshot isolation and deterministic regeneration.

The current research priority is the model-agnostic action-effect binding
audit, not a claim that the existing learned verifier is better. Start with:

- [benchmark contract](docs/counterfactual_binding_benchmark_v1.md);
- [public data/simulator decision](docs/binding_dataset_selection_2026-09.md);
- [GPU preregistration](docs/binding_gpu_preregister.md);
- [evidence-based readiness report](audit/binding_benchmark_readiness_report.md).
- [active/legacy experiment boundary](docs/experiment_status.md);
- [remote GPU handoff](docs/remote_gpu_handoff.md).

The independent benchmark code is NumPy-only. Its focused validation does not
import the repository's current broken Windows PyTorch installation:

```powershell
python scripts/run_binding_bench_example.py `
  --output-dir audit/results/binding_bench_example
python -m pytest tests/test_binding_bench.py -q
python scripts/audit_binding_readiness.py `
  --output audit/results/binding_project_readiness.json `
  --markdown-output audit/binding_benchmark_readiness_report.md
```

`METHOD_GO` from the synthetic example means the harness detects its planted
signal. It is not an ActMask method result. The current overall decision is
`GPU_START_NO_GO`; use the preregistered remote checklist before any compute or
spending.

The root `requirements.txt` belongs to the original CPU-only prototype. For
the active benchmark use `requirements/binding-cpu.txt`; never copy a local
Windows environment or private SSH material to the remote host.
