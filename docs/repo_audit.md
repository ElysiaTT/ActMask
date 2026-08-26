# ActMask Repository Audit

## Audit scope and timing

This audit was performed on **2026-07-13 (Asia/Shanghai), before any ActMask implementation was added**. The complete contents of `/root/blockdata/tzh/papers/ActMask`, including hidden entries, were inspected. At that point the directory was empty and `git -C /root/blockdata/tzh/papers/ActMask rev-parse --show-toplevel` confirmed that it was not a Git repository.

The immediate workspace was also inspected read-only to identify possible coupling risks. Nearby projects include `FastSAM-main`, `gsplat-main`, and `poke_and_splat`; they are separate projects and are not part of this repository.

## Existing structure and capabilities

The pre-implementation repository structure was:

```text
ActMask/
└── (empty)
```

| Area searched | Audit result |
| --- | --- |
| Entrypoints and demo scripts | Absent |
| RGB-D, point-cloud, scene-flow, and other 3D processing | Absent |
| Simulator interfaces | Absent |
| Robot or camera interfaces | Absent |
| Policy, agent, or VLA code | Absent |
| Model definitions | Absent |
| Dataset and data-loading code | Absent |
| Training utilities | Absent |
| Evaluation and metric utilities | Absent |
| Configuration files or conventions | Absent |
| Tests and test configuration | Absent |
| Packaging and dependency files | Absent |
| Documentation | Absent |

There are therefore no internal components, APIs, configuration conventions, dependency declarations, or checkpoints that can be reused. Nearby repositories may contain segmentation, Gaussian-splatting, or manipulation code, but importing them by relative path would create undocumented coupling and is outside this milestone's scope.

## Missing components

The first milestone must introduce the whole minimal stack: a deterministic synthetic dynamic point-cloud dataset, shared baseline-mask interface, lightweight action-conditioned PyTorch model, CPU training and evaluation entrypoints, metrics, headless 3D visualization, configuration, tests, dependency specification, and usage/design documentation. It must also define stable tensor shapes, action semantics, mask-generation geometry, paired-scene identifiers, checkpoint format, output locations, and deterministic seed handling because no prior contracts exist.

Simulator, RGB-D, scene-flow, policy/VLA, and robot integrations should remain explicitly out of scope. The synthetic data contract should be the boundary that a later real-data adapter implements.

## Safest integration path

1. Treat `ActMask` as a small, self-contained greenfield Python project and use the requested `actmask/`, `configs/`, `docs/`, and `tests/` layout without moving or importing sibling source trees.
2. Create a **new isolated environment** at `/root/blockdata/tzh/conda_envs/actmask`; do not reuse or modify `/root/blockdata/tzh/conda_envs/papers` or any other existing environment. Keep the environment outside the repository and record reproducible dependencies inside the repository.
3. Keep the runtime dependency set small: CPU-compatible PyTorch, NumPy, PyYAML, Matplotlib with a non-interactive backend, and pytest. Use only public, stable APIs and do not add CUDA extensions, simulators, pretrained weights, external datasets, or network-dependent tests.
4. Make package modules the canonical entrypoints (`python -m ...`), use one small YAML configuration, and resolve paths relative to the project/config rather than sibling repositories.
5. Add functionality incrementally in dependency order: data contract and geometry, baselines/model, metrics, training/evaluation, visualization, then end-to-end documentation. Verify each layer with deterministic CPU tests and a short smoke run.
6. Write generated checkpoints, metrics, and figures only beneath `outputs/actmask/`; keep generated artifacts out of imports and tests.

## Files and locations not to modify

No pre-existing file inside `ActMask` required preservation because the directory was empty at audit time. Implementation must nevertheless leave all neighboring projects and shared workspace state untouched, especially:

- `/root/blockdata/tzh/papers/FastSAM-main/`
- `/root/blockdata/tzh/papers/gsplat-main/`
- `/root/blockdata/tzh/papers/poke_and_splat/`
- `/root/blockdata/tzh/embodied_agent/`, `/root/blockdata/tzh/pi05/`, and `/root/blockdata/tzh/issac_sim/`
- Existing environments under `/root/blockdata/tzh/conda_envs/`, including `papers`
- Shared caches, datasets, secrets, and editor settings under `/root/blockdata/tzh/`

The only authorized workspace-level addition is the new isolated ActMask environment under `/root/blockdata/tzh/conda_envs/actmask`. ActMask source, configuration, documentation, tests, and outputs should otherwise stay within this project directory.

## Dependency and compatibility risks

- A default PyTorch install may select CUDA packages or require network access. Pin a known CPU-compatible build in the environment documentation and verify tensor/device behavior on CPU.
- Installing into an existing environment could alter unrelated projects or accidentally make tests pass through undeclared packages. The dedicated environment and an explicit dependency file prevent this.
- Matplotlib can select an interactive display backend on a headless host. Force `Agg` before importing plotting APIs and save figures to disk.
- NumPy/PyTorch dtype promotion, random-number-generator differences, and data-loader workers can weaken reproducibility. Use explicit `float32` tensors, seed all generators, and keep smoke-test loading single-process.
- Sparse masks can produce misleading accuracy or undefined precision/recall. Define zero-denominator behavior and test precision, recall, F1, and IoU on edge cases.
- Action time/duration units and tensor layouts can drift between data, baselines, model, and visualization. Centralize and document the action schema and validate shapes at boundaries.
- Checkpoints are sensitive to model/config changes and unsafe arbitrary deserialization. Store a small state dictionary plus configuration metadata, load on CPU, and avoid checkpoints from untrusted sources.
- Imports that depend on the current working directory or sibling `PYTHONPATH` entries are fragile. Use package module entrypoints and test from the repository root in the clean ActMask environment.

This audit is the immutable pre-implementation baseline; later files in the directory are additions made after the audit, not previously existing reusable code.
