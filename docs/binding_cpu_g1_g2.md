# CPU G1/G2 transport freeze (2026-09-06)

This is a bounded implementation addendum, not a change to
`binding_gpu_preregister.json` and not scientific dataset or method admission.
No GPU, remote host, paid compute, robot assets, or training is used.

## What is actually simulated

ManiSkill 3.0.1 owns a single SAPIEN/PhysX CPU scene without rendering.
A 0.5 kg box-shaped rod has identical geometry, visible initial pose and
inertia in both branches, but mirrored hidden centers of mass (COM).
An action chooses where on the rod a fixed upward 6 N force is applied.
PhysX integrates the resulting motion; there is no outcome teleportation.
The outcome is height change and unsigned tilt; utility is height minus
0.1 times tilt. Eight control steps at 20 Hz, physics at 100 Hz.

This is an external-force fixture in free space, **not robot lifting**, grasp
success, contact-rich manipulation, or a replacement for the preregistered
three-family benchmark. Symmetric actions and unsigned tilt deliberately
make this smoke simple. High accuracy here is not a positive research result.

## G1 snapshot boundary

Snapshots contain body pose/quaternion, linear/angular velocity, COM, mass,
inertia and elapsed steps, with a versioned numeric NPZ schema (no pickle).
They are valid at completed control-step boundaries in this contact-free,
deterministic fixture. It has no robot controller or random step-time state.
They do not claim to serialize arbitrary PhysX contact caches, controllers,
assets, RNGs, or cross-version/platform simulation state.

Checks cover repeated replay after deliberately disturbing motion and COM,
disk round-trip, recreation of the environment, moving-state replay, elapsed
time, nonzero motion, and the same action producing a different result under
the other COM. Maximum full-trajectory error must be <= 1e-5.

## G2 data boundary

Default: two seeds, eight twins per seed (16 total); eight independent anchor
probes and 12 candidates per branch. Each execution restores the anchor.
History times are zero because these are independent reset probes, **not**
a continuous temporal trajectory. Branch signs and action slots are shuffled;
opaque IDs do not encode COM or seed. There is only a transport smoke split,
not train/validation/test or OOD admission.

All attempts, including rejected ones, retain snapshots, hidden mechanism,
effects and full trajectories in `evaluator/`. No adaptive retry is allowed.
Admission into this tiny artifact requires identical public state, effect
marginals within 1e-4, utility span > 0.05 and different best actions.
`public/inputs.npz` uses an exact allowlist: history actions/effects/times/mask,
visible context, candidate actions/IDs, twin IDs and schema identifiers.
It contains no hidden COM, seed, episode/group metadata, candidate effects,
utility or replay snapshots. The adapter strips audit metadata and copies
arrays before invoking a method. This is an API/data-export boundary, not
a security sandbox: do not give an untrusted method filesystem access to
`evaluator/`. Twin pairing itself is public in the existing v1 contract.

The independent audit checks SHA-256, public/private consistency, schema,
utility derivation and **all** accepted trajectories by replaying the actions
in reverse execution order in a fresh environment. It then runs the existing
five-case intervention suite using nearest-history-probe and uniform fixtures.
No learned model is proposed. Original method gates are unchanged: this sample
is too small for admission and its `METHOD_NO_GO` is expected.

The integration also fixes cross-branch TPA alignment after independent slot
permutations. Empty difficulty bins now serialize as JSON null, not NaN.
Historical result files remain historical; their old slot TPA was not recomputed.

## Reproduce on Windows or Linux, CPU only

Use a fresh Python 3.11 environment, not the root legacy requirements:

```bash
python -m pip install torch==2.6.0+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements/binding-sim-cpu.txt
python -m pytest tests/test_binding_bench.py tests/test_binding_sim_cpu.py -q
python -m actmask.experiments.binding_gpu.snapshot_smoke --output-dir artifacts/binding_cpu/g1
python -m actmask.experiments.binding_gpu.generate --output-dir artifacts/binding_cpu/g2 --twins-per-seed 8
python -m actmask.experiments.binding_gpu.audit_dataset --dataset-dir artifacts/binding_cpu/g2 --output-dir artifacts/binding_cpu/g2_audit
```

G2 refuses a nonempty output directory. Choose a new directory for reruns;
do not overwrite evidence. CI runs these commands on Ubuntu without CUDA and
uploads generated evidence for 14 days. Missing simulator dependencies fail
the job; integration tests are never skipped. SAPIEN's missing Pinocchio warning
is expected: this fixture has no robot. Core dependency versions are pinned;
transitive dependencies are not a fully hermetic lock.

## Handoff boundary

Push source, tests, CI and small curated reports to GitHub; recreate artifacts
from the reviewed commit on the Linux machine. Do not copy the Windows venv.
The CPU modules deliberately force `physx_cpu`; merely installing a CUDA wheel
does not turn them into a GPU generator. G3 scientific dataset admission and
G4 model comparison remain unimplemented. On the Linux GPU host, first rerun
this CPU smoke and G0; then separately implement/validate GPU backend state
restore, robot/task state, splits and the full preregistered admission gates.
No GPU training starts automatically from this handoff.

## Primary references

- [ManiSkill installation and CPU support](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html)
- [ManiSkill custom state and replay](https://maniskill.readthedocs.io/en/latest/user_guide/tutorials/custom_tasks/advanced.html)
- [Pinned ManiSkill 3.0.1 BaseEnv implementation](https://github.com/haosulab/ManiSkill/blob/v3.0.1/mani_skill/envs/sapien_env.py)
