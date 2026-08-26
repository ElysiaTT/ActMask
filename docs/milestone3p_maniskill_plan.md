# Milestone 3P ManiSkill Plan and Runtime Record

## Scope and environment decision

Stage B uses stable PyPI `mani_skill==3.0.1` and `sapien==3.0.3` on the RTX
4090 D. It uses only project-local procedural collision geometry, a controlled
TCP proxy, and project-local runtime assets. Isaac Sim, RoboTwin, pretrained
models, large datasets, robot hardware, and system package changes remain out
of scope.

The original authorization requested an isolated environment. The user later
explicitly changed this to “环境直接改actmask的就行了”; therefore the work uses
the existing `/data/env/tzh/conda_envs/actmask` environment. No new environment
was created and no unrelated existing process was terminated.

`torch==2.6.0+cu124` is exposed in `actmask` through read-only symbolic links
to the version-identical package tree already present in `papers`. A standalone
CUDA-wheel attempt failed during the 665 MB cuDNN transfer; the precise 24 KB
partial residue was removed and the validated links used instead. `papers` was
not modified. The graphical OpenCV runtime was replaced only in `actmask` by a
read-only link to existing headless OpenCV, avoiding a system `libGL` change.

## B1 pre-install and runtime record

- OS/kernel: Ubuntu 24.04.2 LTS; Linux 5.15.0-94-generic.
- GPU: NVIDIA GeForce RTX 4090 D; driver 580.76.05; 24,564 MiB total and
  24,080 MiB free at final check.
- Python/runtime: 3.11.15; PyTorch 2.6.0+cu124 / CUDA 12.4; ManiSkill 3.0.1;
  SAPIEN 3.0.3; OpenCV runtime 4.10.0.
- Environment size: 1.2 GiB. `/data/env` had 3.8 GiB free and 1,650,249 free
  inodes at final audit.
- Project-local assets: `.maniskill_data` is 301 MiB: the official GPU-PhysX
  archive is 77,938,378 bytes and extracted `libPhysXGpu_64.so` is 236,705,440
  bytes.
- Transfer logs: `outputs/actmask/milestone3p_gpu_benchmark/maniskill_install.log`
  and `pytorch_cuda_install.log`. Retained ManiSkill/SAPIEN wheels are
  101,729,066 / 51,304,469 bytes.

`MS_ASSET_DIR` is `.maniskill_data` and `MS_SKIP_ASSET_DOWNLOAD_PROMPT=1` is
supplied for each run. No unrelated asset pack was downloaded.

`vulkaninfo` is absent. SAPIEN reported missing system `libvulkan` and Vulkan
ICD, and its renderer probe failed to find a rendering device. No sudo, system
package, CUDA-driver replacement, or Vulkan installation was attempted. The
approved fallback is therefore state observation only: RGB-D, depth, and point
cloud rendering remain an explicit host-level blocker.

The reproducible state/GPU smoke is
`actmask.experiments.milestone3p_maniskill_smoke`. It passes imports, one
state rigid-body episode with a simulator `success` field, fixed-seed reset,
and an 8-way GPU vector smoke. The latter achieved 2,079.3 aggregate
control-steps/s over 40 steps with state shape `[8, 20]`. Exact record:
`outputs/actmask/milestone3p_gpu_benchmark/maniskill_state_gpu_smoke.json`.

## B2 task construction and labels

All environments are collision-only GPU PhysX tasks in
`actmask.data.maniskill_state_tasks`; no camera, material, robot asset, or
analytic baseline creates labels.

1. `MovingCubeIntercept`: success is post-execution intercept distance within
   the proxy success radius.
2. `MovingContainerPlacement`: success requires actual acquire/carry/release,
   horizontal containment, height tolerance, and low final payload speed.
3. `RotatingTargetInteraction`: success is post-execution TCP-to-rotating-goal
   interaction distance within the success radius.

For each family a hidden post-decision dynamic branch changes a cube kick,
container translation, or target angular velocity. Positive/negative members
have the same frozen candidate action and restored decision state; their label
is read from `info["success"]` after GPU-PhysX execution. Three mechanisms are
included per family; no policy training is performed.

## B3/B4 contract

The only model-input NPZ fields are `history`, `timestamps`, `visibility`,
`observation_confidence`, `candidate_actions`, `nominal_action_timing`, and
`tcp_state`. Future trajectory/contact/dynamics, task phase, seed, candidate
ID, regime, and success are forbidden. `load_model_inputs` rejects all other
fields and a regression test verifies this failure condition.

The pilot has 108 base worlds per task (36 per regime), six candidates per
world, and 1,944 examples total. Base worlds are assigned as a group to a
deterministic 60/20/20 train/validation/test split. Static-balanced and
matched-counterfactual regimes each have 108 label-flipping pairs per task.
The audit verifies zero current-state, action, and static-feature mismatch at
tolerance `1e-6`, with preceding-history difference at least 0.4000 (container)
or 0.4905 (cube/rotating target).

All generated files and audits are under
`outputs/actmask/milestone3p_gpu_benchmark/maniskill_pilot/`.
