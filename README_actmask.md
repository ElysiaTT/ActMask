# ActMask: CPU-only milestone 1

ActMask is a small research prototype for action-conditioned future masking in
dynamic 3D scenes. Given current points, per-point velocities, and a candidate
gripper motion, it predicts which points the action will contact or affect. This
milestone is deliberately synthetic and CPU-only: it downloads no data or model
weights and requires no simulator, camera, robot, or CUDA installation.

## Environment

The requested environment has already been created and verified at
`/root/blockdata/tzh/conda_envs/actmask`. Activate it, then work from the
repository root `/root/blockdata/tzh/papers/ActMask`:

```bash
conda activate /root/blockdata/tzh/conda_envs/actmask
cd /root/blockdata/tzh/papers/ActMask
```

To recreate the environment at a clean prefix, use:

```bash
conda create -p /root/blockdata/tzh/conda_envs/actmask python=3.11 pip -y
conda activate /root/blockdata/tzh/conda_envs/actmask
python -m pip install -r requirements.txt
```

Do not rerun `conda create` while that prefix already exists.

`requirements.txt` pins a CPU PyTorch wheel and uses PyTorch's official CPU
wheel index as an additional package source. It does not install a CUDA wheel.
The project itself is imported directly from the repository root; no editable
package installation is needed.

## Run the milestone

Train first. Evaluation of the learned model and visualization of learned
predictions use `outputs/actmask/best_model.pt`, which training creates. The
three analytic baselines need no checkpoint.

```bash
python -m actmask.training.train --config configs/actmask/toy_cpu.yaml
python -m actmask.eval.evaluate --config configs/actmask/toy_cpu.yaml
python -m actmask.visualization.visualize_masks --config configs/actmask/toy_cpu.yaml
pytest tests/test_actmask_shapes.py
```

The commands are intended to be run in that order. Training prints real train
and validation losses and writes the best CPU checkpoint plus
`training_metrics.json`. Evaluation computes metrics for `NoMask`,
`MotionMagnitudeMask`, `ActionProximityMask`, and learned `ActMask`, then writes
`evaluation_metrics.json`. Visualization uses Matplotlib's headless `Agg`
backend and writes one or more PNG comparisons under `outputs/actmask/`.

All paths, dataset sizes, random seeds, optimizer settings, metric threshold,
and model widths are configured in `configs/actmask/toy_cpu.yaml`. Outputs are
prototype diagnostics, not claimed research results.

## Data and tensor contract

Each synthetic item contains `points [N,3]`, `velocities [N,3]`, `action [8]`,
`mask [N]`, scalar `success`, scalar `pair_id`, scalar `variant_id`, and a
scenario name. The action is
`[start_x,start_y,start_z,end_x,end_y,end_z,duration_s,radius]`. Models and all
baselines accept batched points, velocities, and actions and return raw mask
logits of shape `[B,N]`.

See [docs/actmask_design.md](docs/actmask_design.md) for the precise synthetic
geometry, mask and metric semantics, limitations, and proposed next steps.
