# Milestone 3R-NL Handoff

## Completed artifacts

- Frozen configuration: `outputs/actmask/milestone3r_nl_v1/preregistered_config.json`
- Strict GPU pair generator: `actmask/experiments/milestone3r_nl.py`
- GPU task families: `actmask/data/maniskill_3r_nl_tasks.py`
- Probe and full evaluation: `probe_evaluation.json`, `evaluation/full_evaluation.json`
- C20 simulator data and aligned C5/C10 prefixes: `full_c20/`, `full_c5/`, `full_c10/`
- Correction/fairness logs: `physical_corrections.json`, `fairness_fixes.json`

## Reproduction

Use `/data/env/tzh/conda_envs/actmask/bin/python`, with `CUDA_VISIBLE_DEVICES=0`
and `MS_ASSET_DIR=/data/projects/tzh/papers/ActMask/.maniskill_data`.

```bash
python -c "from actmask.experiments.milestone3r_nl_full import run; run('outputs/actmask/milestone3r_nl_v1/full_c20','outputs/actmask/milestone3r_nl_v1/evaluation',steps=120)"
```

This recomputes learned evaluation; it does not regenerate PhysX data.  Do not
overwrite frozen 3Q/3R artifacts or alter the v1 config hash.

## Next authorized decision needed

Do not treat v1 as a nonlinear-OOD pass.  A follow-on must be newly versioned
and pre-registered before any data generation.  It needs an explicitly
bounded additional GPU execution budget for independent damping/frequency,
threshold/delay, and held-mechanism OOD data, plus candidate policies that
produce action-dependent outcomes while retaining exact matched-final pairs.
It should keep the existing tests and add a non-degenerate within-world
candidate-label test and a reversal-sensitive causal criterion before scale-up.
