# GPU preregistration: Counterfactual Binding Audit

Frozen: 2026-09-03, before remote GPU access
Machine-readable protocol: `docs/binding_gpu_preregister.json`
Current execution status: **GPU_START_NO_GO**

## Scope

The first GPU job is a state-only data-generation and simulator-integrity
pilot. It does not train a vision backbone and cannot establish a learned
ActMask improvement. Its purpose is to generate execution-grounded outcomes
for several candidate actions restored from the same state.

Primary platform is ManiSkill 3.0.1 on Linux/NVIDIA. The public PickCube teleop
trajectory at pinned revision `d674485...` supplies a verified state/action
format and realistic control-scale reference. New benchmark worlds use only
procedural geometry so third-party asset ambiguity does not enter the initial
license boundary. RoboMimic/robosuite is the independent backup simulator.

## Physical twin construction

Three procedural task families are frozen:

1. visually identical rods with mirrored hidden center of mass;
2. visually identical articulated panels with directional hidden latch or
   restoring response;
3. visually identical blocks/supports with mirrored anisotropic slip.

The hidden mechanism is evaluator state, included in restorable task state but
excluded from every public observation and method input. Each twin shares
public current state, target, action coordinate frame, candidate identities and
candidate actions. Symmetric probe sets are used so branches match action and
effect marginals while differing in correspondence.

Every history probe and candidate starts by restoring the same anchor snapshot.
After an eight-control-step action chunk, the generator records public effect
deltas and evaluator-only contact, state, success and utility. This repeated-
trial design is deliberate: a logged single continuation cannot label actions
that were never executed.

## Frozen scale and split rule

Five seeds are `17001..17005`. Each has 384 train twins, 128 validation twins,
and 128 evaluation twins for each of ID, parameter OOD, sparse history,
observation-noise OOD and held mechanism. Histories normally contain 8–20
probes; sparse history contains 4–7. There are 12 candidate chunks per branch.

Splits are made before model training by task family, procedural scene/object
identity, mechanism family, parameter interval and generation seed. Twin IDs,
both episode IDs, and `split_group_ids` must have zero overlap. All generation
attempts—including replay failures, low utility-span worlds, non-crossed twins
and rejected marginal matches—remain in a manifest. A retained-only manifest
is invalid.

## Data admission before any model training

Repeated snapshot/action rollouts must agree within `1e-5`; twin public states
must match within `1e-5`; candidate action arrays must be byte-identical. Every
branch needs utility span at least 0.05, at least 70% informative candidates,
at least 70% crossed-preference twins, balanced branch assignment, and at least
20% pre-filter retention.

Candidate-ID/slot/current/action/effect-only and other nonbinding controls must
not solve the construction. A generic binding witness must be informative but
not saturated: TPA `[0.70, 0.95)`, normalized regret at least 0.02, binding
break TPA drop at least 0.15 and regret increase at least 0.10. Complete-pair
permutation may change metrics by at most 0.01 and candidate-slot permutation
by at most 0.005.

Failure terminates the learned-model stage. Results may motivate a newly
versioned exploratory construction, but thresholds or splits cannot be edited
after observing outcomes.

## Learned-method stage

Mandatory admission baselines are current-only, candidate ID/slot, action-only,
effect-only, equal-capacity unbound history, last effect, nearest, kNN5,
local-linear, Poly2 ridge, RBF ridge, and sparse inverse reachability. Candidate
truth, hidden mechanism, source IDs and future frames are forbidden inputs.

The independent evaluator in `binding_bench/` chooses the strongest baseline.
On every split a proposed method must reduce normalized regret by at least 10%,
win at least four of five seeds, have a hierarchical-bootstrap lower 95% bound
above zero, and improve Top-1 by at least 0.03. It must also pass the binding,
twin-swap, pair/slot invariance and anti-saturation gates frozen in the JSON.
One split passing is insufficient.

## Resource envelope

The state pilot requires Linux/NVIDIA, at least 12 GiB VRAM, 50 GiB free disk,
and is capped at 8 GPU-hours and 10 GiB of retained output. This is an upper
bound, not an expected bill. The later visual stage requires a separate
authorization, at least 24 GiB VRAM, 200 GiB free disk, and is capped at 48
GPU-hours/120 GiB output.

No provider, hourly price, or maximum payable amount has been approved. The
machine-readable `approved_cost_amount` is therefore `null` and
`cost_authorized=false`; paid GPU execution is forbidden until those fields and
the exact remote host are reviewed.

## Staged commands

These commands are a runbook, not authorization to execute them now.

```bash
# G0: fresh Linux environment; keep data/cache inside the named work volume.
python3.11 -m venv .venv-binding-gpu
source .venv-binding-gpu/bin/activate
python -m pip install --upgrade pip
python -m pip install "mani_skill==3.0.1" pytest
# Install a CUDA-enabled PyTorch wheel selected for the observed driver only
# after nvidia-smi has been archived. Do not copy the current Windows install.

# G0 evidence and CPU harness.
nvidia-smi
python -m pip freeze
python -m pytest tests/test_binding_bench.py -q
python scripts/check_binding_gpu_readiness.py \
  --config configs/binding_gpu_readiness.json \
  --workspace "$PWD" \
  --output audit/results/binding_gpu_readiness_remote.json

# G1: simulator smoke only (commands become available with the generator).
python -m actmask.experiments.binding_gpu.snapshot_smoke \
  --preregister docs/binding_gpu_preregister.json \
  --output audit/results/binding_gpu/snapshot_smoke.json

# G2: bounded data-construction smoke; no learned model.
python -m actmask.experiments.binding_gpu.generate \
  --preregister docs/binding_gpu_preregister.json --stage smoke
python -m actmask.experiments.binding_gpu.audit_dataset \
  --preregister docs/binding_gpu_preregister.json --stage smoke

# G3: only if G0/G1/G2 all say GO.
python -m actmask.experiments.binding_gpu.generate \
  --preregister docs/binding_gpu_preregister.json --stage state-pilot
python -m actmask.experiments.binding_gpu.audit_dataset \
  --preregister docs/binding_gpu_preregister.json --stage state-pilot

# G4: baselines and a frozen small bound/unbound verifier comparison.
python -m actmask.experiments.binding_gpu.run_state_methods \
  --preregister docs/binding_gpu_preregister.json
python scripts/evaluate_binding_benchmark.py \
  --plan audit/results/binding_gpu/evaluation_plan.json \
  --output audit/results/binding_gpu/evaluation.json
```

G1–G4 modules are intentionally not claimed as implemented in the current CPU
checkpoint. Implementing them is the next coding milestone after a real remote
host passes G0. The readiness audit fails if planned commands are represented
as completed evidence.

## Terminal interpretation

- Dataset gate failure: benchmark construction `NO-GO`; no learned training.
- State learned-method failure: keep benchmark/negative result; no visual stage.
- All state gates pass: authorize review and freezing of a visual-stage plan,
  not automatic visual training.
- External robosuite replication is required before a platform-general claim.
- DROID/Bridge/OXE can support real logged-transition checks, but not replace
  snapshot-branch counterfactual truth.
