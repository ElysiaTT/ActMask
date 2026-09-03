# Public data and simulator selection for Binding Audit

Date: 2026-09-03
Decision: **ManiSkill primary; RoboMimic/robosuite backup**

## Selection criterion

This ranking is intentionally not a popularity ranking. The decisive question
is whether we can restore the *same pre-action world state* and execute several
candidate actions from it. A sequential log provides factual
`observation-action-next_observation` transitions, but only a restorable
simulator snapshot can provide labels for actions that were not taken.

The benchmark therefore distinguishes:

- **counterfactual source:** state/snapshot + compatible simulator + deterministic
  reset/restore + action stepping;
- **transition source:** synchronized observations/actions with the next step
  implicit or explicit, but no way to ask what another action would have done.

Only the first class can be the primary evaluation source. The second class can
support representation pretraining or a real-data consistency appendix, never
the main counterfactual claim.

## Dataset matrix

| Rank | Candidate | Synchronized obs/action/next | Raw actions | Restorable world state | Replay / unseen-action rollout | License | Scale and smallest defensible entry | Binding-audit decision |
|---:|---|---|---|---|---|---|---|---|
| 1 | **ManiSkill 3** | Yes after replay; raw demos store actions and `T+1` environment states. | Yes; control mode recorded in metadata. | **Yes**, including actor/articulation state and custom task state through `get_state_dict`/`set_state_dict`. | **Yes**. Official tools replay from state; restored snapshots can branch to multiple actions. | Framework and demo card Apache-2.0; verify any third-party task assets separately. | Download per task. Probed PickCube teleop HDF5 is 0.384 MB; motion-planning HDF5 is 29.3 MB. Visual replay and training can be generated later. | **Primary.** Best fit for true counterfactual rollout, heterogeneous hidden mechanisms, and GPU-vectorized generation. |
| 2 | **RoboMimic + robosuite** | Yes; processed files may contain `obs`/`next_obs`; raw files contain ordered state/action pairs from which observations are regenerated. | Yes. | **Yes**, flattened MuJoCo states plus per-demo model XML and dataset environment args. | **Yes** through robosuite state restoration and stepping, subject to version pinning. | Code and official HF dataset card MIT. | Task-selective raw HDF5. Probed Lift PH is 36.7 MB with 200 demos; real image releases range from 1.9 GB to 58 GB. | **Backup.** Strong independent physics engine and easy low-dimensional pilot; slower counterfactual collection than ManiSkill GPU vectorization. |
| 3 | **LIBERO** | Yes; per-demo actions, MuJoCo states, generated observations/rewards/dones. | Yes, 7-D actions. | Yes, with BDDL/task setup and flattened MuJoCo state. | Technically yes, but official conversion warns when action replay diverges from recorded next state by more than 0.01. Exactness must be requalified. | Code MIT; datasets CC BY 4.0. | Downloadable by suite/task; official page does not publish a reliable compact byte budget, so HEAD/manifest probing is mandatory before acquisition. | Phase-2 external validity only. It adds tasks/language, but replay drift and an older dependency stack make it a weaker first platform. |
| 4 | **DROID** | Yes in RLDS; each step contains observation, 7-D action, reward/terminal markers, so next observation is the next step. | Yes; raw HDF5 has low-dimensional trajectories. | **No simulator world snapshot.** Robot proprioception is not the full physical scene state. | Logged action replay on the real robot is possible operationally, but offline counterfactual rollout is not. | Dataset CC BY 4.0; policy code MIT. | Official `droid_100`: 100 episodes / 2 GB. Full RLDS: 1.7 TB; raw stereo-HD release: 8.7 TB. | Auxiliary real-data transition/pretraining source only. Do not infer unseen-action outcomes from it. |
| 5 | **BridgeData V2** | Yes in RLDS: observations/states/actions by step; next observation is the next step. | Yes, Cartesian/gripper action. | No complete simulator snapshot. | No offline counterfactual replayer; real WidowX execution requires hardware. | Data CC BY 4.0; code MIT. | 60,096 trajectories. Official TFDS page reports 387.49 GiB; no compact official subset is guaranteed. | Auxiliary only; too large and lacks counterfactual labels for first GPU stage. |
| 6 | **Open X-Embodiment** | Unified RLDS episode/step format, but fields and action semantics vary by contributing dataset. | Yes, heterogeneous across embodiments. | Generally no; converted RLDS does not standardize simulator-restorable state. | Generally no. A few component datasets may have their own simulator, which should be evaluated at the source rather than through OXE. | OXE software Apache-2.0 and other released materials CC BY 4.0; component citations/terms still require audit. | 1M+ trajectories across 22 embodiments in the paper; no single small homogeneous action space or download budget. | Poor primary benchmark source; useful later for cross-dataset pretraining/shortcut analysis only. |

## Official evidence

- ManiSkill documents that raw demos include initial states, actions, seeds and
  can be replayed; state replay is specifically required for identical
  observations/rewards. It also documents `get_state_dict`/`set_state_dict` for
  complete custom state and reports that Linux + NVIDIA supports GPU simulation,
  while Windows and WSL do not: [trajectory replay](https://maniskill.readthedocs.io/en/latest/user_guide/datasets/replay.html),
  [custom state](https://github.com/haosulab/ManiSkill/blob/main/docs/source/user_guide/tutorials/custom_tasks/advanced.md),
  [system support and license](https://github.com/haosulab/ManiSkill), and
  [demonstration format](https://maniskill.readthedocs.io/en/v3.0.0b20/user_guide/datasets/demos.html).
- RoboMimic defines raw robosuite datasets with environment args, per-demo
  MuJoCo model XML, flattened states, and actions, and provides a state-to-observation
  converter that can emit `next_obs`: [dataset overview](https://robomimic.github.io/docs/datasets/overview.html),
  [robosuite datasets](https://robomimic.github.io/docs/datasets/robosuite.html),
  and [official repository](https://github.com/ARISE-Initiative/robomimic).
- LIBERO publishes four human-demonstration suites, fixed initial states, and
  sparse task success; its own collection/conversion code persists MuJoCo state,
  actions, model XML and detects replay divergence: [official README and license](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/README.md),
  [dataset conversion](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/scripts/create_dataset.py),
  and [dataset page](https://libero-project.github.io/datasets).
- DROID documents its RLDS/raw schemas and size; the paper reports 76k
  demonstrations, 350 hours, 564 scenes, and CC BY 4.0:
  [dataset documentation](https://droid-dataset.github.io/droid/the-droid-dataset),
  [policy/data release](https://github.com/droid-dataset/droid_policy_learning),
  and [paper](https://arxiv.org/abs/2403.12945).
- BridgeData V2 reports 60,096 trajectories and CC BY 4.0, while TFDS reports
  the 387.49 GiB RLDS schema: [project page](https://rail-berkeley.github.io/bridgedata/),
  [official code](https://github.com/rail-berkeley/bridge_data_v2), and
  [TFDS catalog](https://www.tensorflow.org/datasets/catalog/bridge).
- OXE defines an RLDS episode collection and reports licensing, while its paper
  reports 1M+ trajectories and 22 embodiments:
  [official repository](https://github.com/google-deepmind/open_x_embodiment)
  and [paper](https://robotics-transformer-x.github.io/paper.pdf).

## Bounded CPU data probe

`scripts/probe_binding_datasets.py` queried pinned official Hugging Face
revisions, downloaded only two files into an automatically deleted temporary
directory, verified the official LFS SHA-256, and inspected HDF5 structure.

| Probe | Revision / file | Observed structure | Result |
|---|---|---|---|
| ManiSkill | `d674485...`; PickCube teleop `trajectory.h5`, 383,621 bytes | 10 trajectories; first has 105 8-D actions and four environment-state leaves, each with 106 rows. | SHA and `T+1` transition alignment pass. |
| RoboMimic | `74fa018...`; Lift PH `demo_v15.hdf5`, 36,699,586 bytes | 200 demos; first has 59 7-D actions, 59 32-D raw states, model XML, controller info and root environment args. | SHA and structural replay prerequisites pass. |

Machine-readable evidence is `audit/results/binding_dataset_probe.json`. Re-run:

```powershell
python scripts/probe_binding_datasets.py `
  --output audit/results/binding_dataset_probe.json
```

The probe proves file authenticity and structural feasibility, not simulator
runtime reproducibility. The current Windows environment has none of
`mani_skill`, `robomimic`, `robosuite`, or `libero` installed; its PyTorch import
also fails at `c10.dll`. ManiSkill officially does not support GPU simulation
on Windows or WSL. Therefore GPU generation must run in a separate pinned
Linux/NVIDIA environment.

## How the primary dataset becomes a binding benchmark

The downloaded demonstrations are not themselves counterfactual labels. They
provide task definitions, seed/state snapshots, and plausible action scales.
The primary construction is a **restored-state rollout table**:

1. Select a preregistered anchor state and keep public geometry/current
   observation fixed.
2. Create two visually matched branches whose hidden physical mechanism differs
   (for example mirrored center-of-mass, hinge response, or contact/slip law).
3. Restore the anchor before every probe. Execute a balanced, symmetric probe
   action set and record the synchronized effect after a fixed horizon.
4. Restore the same anchor before every candidate. Execute all candidates and
   compute utility from simulator state, contact, and success—not from a model.
5. Retain a twin only when public-current and candidate equality, mechanism
   concealment, replay tolerance, utility span, crossed preference, and branch
   balance all pass. Attempted and rejected worlds remain in the manifest.
6. Split before model training by task family, scene/object identity, mechanism
   family, and generation seed. No snapshot, episode, or object variant may
   cross partitions.

This uses public ManiSkill data and infrastructure but produces a new,
execution-grounded audit dataset. It avoids the invalid shortcut of treating a
single logged continuation as the ground truth for arbitrary candidate actions.
