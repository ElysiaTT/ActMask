# Dependency scopes

The root `requirements.txt` reproduces the original CPU ActMask prototype and
pins a CPU-only PyTorch wheel. Do not use it to prepare the binding GPU host.

- `binding-cpu.txt`: NumPy-only benchmark, tests, and bounded HDF5 probes.
- `binding-sim-cpu.txt`: pinned ManiSkill/SAPIEN CPU G1/G2 runtime; install
  `torch==2.6.0+cpu` separately from the official CPU index. See
  `docs/binding_cpu_g1_g2.md`. No robot assets are required.
- `binding-gpu.txt`: CPU benchmark dependencies plus pinned ManiSkill 3.0.1.

CUDA PyTorch is intentionally absent from `binding-gpu.txt`: the correct wheel
depends on the observed remote NVIDIA driver. On the approved Linux host, set
an exact `BINDING_TORCH_SPEC` and official PyTorch wheel index before running
`bash scripts/remote/bootstrap_binding_env.sh`. Archive the selected values,
`nvidia-smi`, and `pip freeze` with the run.

Neither requirements file downloads robot datasets. Demonstrations and
generated rollouts belong on the named remote work volume, not in Git.
