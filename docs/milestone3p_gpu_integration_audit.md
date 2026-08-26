# Milestone 3P GPU Integration Audit

## Scope

This is the authorized Stage-B local audit only. It did not run a simulator,
benchmark, container, install, download, or modify an external project.

## B0: GPU qualification

The initial audit snapshot found no visible GPU. That external condition later
changed and was re-qualified without altering the CPU environment. The
`papers` environment now exposes an NVIDIA GeForce RTX 4090 D (24,564 MiB;
driver 580.76.05) through PyTorch `2.6.0+cu124`.

The fixed observable-only mini-batch B0 test passed for both
`CurrentPositionProximity` and `EstimatedTimeAlignedTrajectoryProximity`:
CPU--CUDA FP32 maximum absolute error is `2.98e-8`, rankings are identical,
two CUDA evaluations are bitwise equal, and FP16 outputs are finite with the
same rankings. Deterministic algorithms were enabled for this check. See
`outputs/actmask/milestone3p_gpu_benchmark/gpu_qualification_runtime.json`.

This qualification covers only the small tested operators. Simulator kernels
and any new CUDA operator must be qualified again in their actual benchmark.

## B1: local simulator/resource audit

ActMask itself has a CPU-only requirements file and no simulator adapter. The
only locally discovered potentially related source tree is
`/data/projects/tzh/pi05/isaacsim-teleop` (60 MB), which is an external Isaac
Sim teleoperation project, not an ActMask dependency. Its README requires an
NVIDIA RTX GPU plus either Docker/NVIDIA Container Toolkit or a preinstalled
Isaac Sim 4.5.0 runtime. It includes small URDF, XML, USD, and USDA assets.

No local Isaac Sim installation was found. Docker client software is present,
but the Docker daemon remains unavailable after the GPU recheck. No ActMask
simulator trajectories, RGB-D recordings, point clouds, robot logs, camera
calibrations, action/contact/success labels, or scene-flow outputs were found.
The external candidate is consequently not usable in the current session and
must not be automatically adopted because it has a different teleoperation
scope.

## Decision

**B0 PASSED; STOP BEFORE B2 GPU BENCHMARK — HUMAN RESOURCE DECISION REQUIRED.**
The Stage-A benchmark result remains valid and is not affected by this local
data/runtime block.

Choose one of the following before resuming:

1. Provide an existing approved local simulator/runtime and identify which
   local assets and task family ActMask may use.
2. Provide approved realistic recordings/trajectories plus the success-label
   definition, if simulator execution is not intended.
3. Explicitly authorize installation/download of a named simulator and assets;
   this is a scope expansion and will not be assumed.
