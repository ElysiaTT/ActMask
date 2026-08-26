# Milestone 3P GPU Handoff

## Completed state

The minimal ManiSkill GPU pilot is complete in the existing
`/data/env/tzh/conda_envs/actmask` environment, as explicitly requested by the
user after the initial isolated-environment instruction. RTX 4090 D works with
PyTorch 2.6.0+cu124, ManiSkill 3.0.1, and SAPIEN 3.0.3. The full pilot decision
is **A — GO TO FULL GPU BENCHMARK**.

## Key artifacts

- `outputs/actmask/milestone3p_gpu_benchmark/maniskill_state_gpu_smoke.json`:
  CUDA/state/vector/determinism qualification.
- `outputs/actmask/milestone3p_gpu_benchmark/maniskill_pilot/`: strict model
  inputs, labels, metadata, matching audit, B5 records, corruptions, five-seed
  confirmation, and logs.
- `actmask/data/maniskill_state_tasks.py`: three collision-only GPU-PhysX task
  families and simulator success definitions.
- `actmask/data/maniskill_pilot.py`: data generation and input isolation.
- `actmask/experiments/milestone3p_maniskill_baselines.py`: B5 baselines,
  metrics, corruptions, and B6 evidence.

## Constraints to preserve

`.maniskill_data` is project-local (301 MiB). Do not download a large asset
pack, pretrained model, VLA, demonstration dataset, Isaac Sim, or RoboTwin
without new authorization. Do not alter frozen CPU conclusions or kill
unrelated GPU processes for memory.

The host lacks `vulkaninfo` and a usable Vulkan ICD. State simulation works,
but RGB-D/depth/point-cloud rendering is blocked. Do not install system Vulkan
components without explicit permission.

## Completion audit (2026-07-23)

Current-state verification, rather than only the original run logs, confirmed
that the `actmask` environment imports PyTorch 2.6.0+cu124, ManiSkill 3.0.1,
and SAPIEN 3.0.3 with CUDA available; `pip check` reports no broken
requirements. The three generated input archives contain only the seven
observable fields, and the regenerated matching audit passes all 108
static-balanced and 108 matched-counterfactual pairs per task at tolerance
`1e-6`. The B5 report covers every required static/dynamic method, regime,
metric, corruption view, three-seed pilot, and five-seed key confirmation.

The full project suite was rerun from this environment: **136 passed** in
58.52 seconds. The only warnings are existing matplotlib/PyParsing deprecations
and the documented missing Vulkan runtime/ICD. This proves completion of the
state-based pilot; visual RGB-D/depth/point-cloud qualification remains blocked
by the host condition described above.

## Recommended next bounded work

Scale only the approved state-based benchmark: more grouped worlds, more seeds,
and mechanisms requiring signed velocity/phase rather than merely motion
presence. Retain exact static/action matching, grouped splits, simulator labels,
and the strict input whitelist. Re-run B5 and B6 before claiming transfer beyond
this procedural pilot.
