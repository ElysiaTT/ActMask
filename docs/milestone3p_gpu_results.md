# Milestone 3P GPU Results

## B0 qualification result

**PASS.** The current session has one NVIDIA GeForce RTX 4090 D (24,564 MiB;
driver 580.76.05) and 24,080 MiB free at qualification. The `papers`
environment uses PyTorch `2.6.0+cu124` and reports one visible CUDA device.

For a fixed ten-row observable-only counterfactual mini-batch, both tested
baselines passed CPU--CUDA FP32 agreement at `atol=1e-5, rtol=1e-4` with a
maximum absolute error of `2.98e-8`. CUDA repeats were bitwise equal; FP16
outputs were finite and retained the same candidate rankings. The full runtime
record is `outputs/actmask/milestone3p_gpu_benchmark/gpu_qualification_runtime.json`.

## B2 benchmark status

**Not started.** GPU availability alone does not provide the required
realistic simulator or recorded-data benchmark. No local Isaac Sim runtime,
Docker daemon, ActMask trajectories, RGB-D recordings, point clouds, robot
logs, camera calibration, action/contact labels, or success labels is
available. No simulator, dependency, asset, or external source project was
installed, started, downloaded, or imported.

## Next authority required

Provide an approved local runtime and task/asset scope, approved realistic
recordings with a success-label definition, or explicit installation/download
authorization. Until then, no Stage-B benchmark claim is made.
