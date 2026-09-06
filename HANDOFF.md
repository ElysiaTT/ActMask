# ActMask handoff

## Current focus

The active direction is the model-agnostic Counterfactual Action-Effect Binding
Audit. The reusable CPU benchmark is ready, but the existing learned verifier
is not a positive result: v1 is a saturation diagnostic and v2/v3 are NO-GO.

The public-source decision is ManiSkill 3 as the primary counterfactual
simulator and RoboMimic/robosuite as the backup. Bounded format probes passed;
CPU-only physical replay now has implemented G1/G2 checks and Ubuntu CI;
simulator replay on the target GPU host has not yet been run. Start with
`docs/binding_cpu_g1_g2.md` for the new smoke scope and reproduction commands.

## Start here

1. Read `docs/experiment_status.md` for active versus historical paths.
2. Read `docs/counterfactual_binding_benchmark_v1.md` for the evaluator.
3. Read `docs/remote_gpu_handoff.md` before moving to another machine.
4. Inspect `audit/binding_benchmark_readiness_report.md` for the current
   evidence-backed GO/NO-GO breakdown.

## Reproduce the active CPU layer

Use Python 3.11 from the repository root:

```bash
python -m pip install -r requirements/binding-cpu.txt
python scripts/run_binding_bench_example.py \
  --output-dir audit/results/binding_bench_example
python -m pytest tests/test_binding_bench.py -q
python scripts/audit_binding_readiness.py \
  --output audit/results/binding_project_readiness.json \
  --markdown-output audit/binding_benchmark_readiness_report.md
```

The example's `METHOD_GO` is a benchmark self-test against a deliberately weak
baseline. It is not an ActMask method claim.

## Repository boundaries

- `binding_bench/`: active, dependency-light benchmark package.
- `scripts/cpu_binding_*` and `scripts/cpu_timearrow_*`: retained historical
  experiments and negative-result provenance.
- `actmask/experiments/binding_gpu/`: active CPU G1/G2 physical transport smoke.
- Other `actmask/` paths: original model code and older experimental generations.
- `docs/`, `audit/`: frozen protocols, reports, and curated evidence.
- `paper*`, `submission/`, `release_candidate/`: historical manuscript and
  packaging material; drafts are not accepted claims.

## GPU boundary

Current status is `GPU_START_NO_GO`. Do not use the root CPU-only
`requirements.txt` on the GPU host. Follow `docs/remote_gpu_handoff.md` and the
frozen preregistration. A named native-Linux/NVIDIA host, explicit cost ceiling,
passing G0 report, and implemented G1/G2 modules are required before data
generation. Visual training remains disabled.

No private SSH configuration, credentials, local environments, HDF5 datasets,
or caches belong in Git.
