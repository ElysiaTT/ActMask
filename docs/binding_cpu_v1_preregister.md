# BindingCheck-CPU-v1 preregistration

Frozen on 2026-09-03 before the first result from
`scripts/cpu_binding_pre_admission.py`.

## Status and scope

This is a CPU transport and direction-admission screen derived from the existing
History--Action Binding v2.1 symbolic construction. That earlier construction
and its reported development/official outcomes were already visible, so this
run is not presented as a fresh confirmatory test. Its purpose is to establish
that the proposed direction can run reproducibly in the current Windows CPU
environment without importing PyTorch, ManiSkill, or PhysX.

No learned model is trained in this stage. Passing authorizes a new,
independently frozen CPU learned-model experiment. It does not authorize GPU,
simulator, vision, or real-robot claims.

## Fixed question

Can matched twin histories retain candidate-ranking signal that requires the
correct action-effect correspondence after current-state, candidate-slot,
action-marginal, result-marginal, endpoint/arrow, and simple history-lookup
shortcuts are controlled?

## Fixed tasks

Every twin contains two branches with the same public rotation, current state,
goal, probe actions, candidate actions, and result multiset. The hidden response
phase differs by 45 degrees, which permutes which action produced which result
and crosses candidate preferences.

Task A uses

`d(u) = 0.0578 ||u|| cos(2(angle(u) - phi))`.

Task B uses

`d(u) = 0.0578 (alpha ||u|| - beta_factor beta ||u||^2)
cos(2(angle(u) - phi))`,

with `alpha in [1.05, 1.25]`, `beta in [0.18, 0.32]`, and hidden phase jitter
in `[0, 45]` degrees. Public probe results are quantized to `0.001`.

Probe directions are the eight 45-degree compass directions. Task A probe
magnitudes are 0.40 and 0.80; Task B magnitudes are 0.20, 0.50, and 0.80.
There are sixteen candidates per twin. Candidate magnitude is 0.35 or 0.65;
candidate directions do not coincide with the probe directions. Candidate
slots follow a cyclic Latin-square schedule and candidate IDs are not legal
baseline inputs except for the explicitly reported slot baseline.

## Fixed seeds and size

- Seeds: 8301, 8302, 8303, 8304, 8305.
- 256 twins per task and seed.
- Every seed, task, branch, candidate, and failed gate is reported.
- No outcome filtering, resampling, best-seed selection, or threshold tuning.

## Fixed baselines

1. class prior;
2. current only;
3. candidate slot/index;
4. candidate action only;
5. result marginal only;
6. endpoint/arrow-action compatibility;
7. nearest historical action lookup;
8. two-nearest-neighbor action-result lookup;
9. linear system identification;
10. quadratic/Fourier system identification;
11. latent oracle rollout, reported only as an identifiability ceiling.

The quadratic/Fourier fit is the preregistered binding-aware witness. It fits
the response surface using paired probe actions and results and evaluates novel
candidate angles/magnitudes.

## Fixed interventions

- Pair-preserving permutation moves complete action-result pairs together.
- Binding-breaking permutation moves results while leaving actions fixed.
- Twin-history swap exchanges histories across matched branches.
- Candidate-slot reversal jointly permutes candidate actions and labels.
- A 37-degree coordinate rotation jointly rotates probes, candidates, public
  rotation, and latent phase.

## Metrics

The primary metric is candidate-conditioned twin preference accuracy. True and
predicted utilities are centered within each branch; for every candidate
template, the metric tests whether the predicted preference between twin
branches has the correct sign. Ties receive 0.5.

Secondary diagnostics include per-template accuracy, top-1 crossing, slot
balance, and action/result/current multiset integrity.

## Frozen decision gates

The direction is `CPU_DIRECTION_ADMITTED` only if both tasks pass every gate in
the mean over five seeds and no seed exhibits a structural-integrity failure:

1. action marginals, result multisets, current states, candidate sets, and
   cyclic slot counts are exact within their declared numeric representation;
2. candidate Top-1 differs between matched branches in at least 0.90 of twins;
3. current/action/result/slot controls are at most 0.60;
4. endpoint/arrow is at most 0.70;
5. nearest and two-nearest lookup are both strictly below 0.90;
6. Fourier witness is at least 0.90;
7. binding-breaking reduces Fourier accuracy by at least 0.20;
8. pair-preserving permutation changes Fourier accuracy by at most 0.02;
9. candidate-slot and coordinate transforms do not move the Fourier witness to
   the other side of the 0.90 admission threshold;
10. the oracle and Fourier witness exceed the strongest marginal control.

Any failed gate yields `CPU_DIRECTION_REJECTED`. A pass authorizes only the
next CPU learned-model preregistration; no GPU command may be run by this goal.

## Reproduction

```powershell
python scripts\cpu_binding_pre_admission.py `
  --output audit\results\binding_cpu_v1_pre_admission.json
```
