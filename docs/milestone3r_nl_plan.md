# Milestone 3R-NL Plan: Matched-Final Nonlinear-Dynamics OOD

## Status and immutability

This is a separately versioned extension authorized after frozen Milestone 3R.
It does not edit or reinterpret any 3Q/3R artifact, label, split, threshold,
checkpoint, or OOD result. All extension outputs live below
`outputs/actmask/milestone3r_nl_v1/`. No visual, Vulkan, system-package,
asset-download, pretrained-model, or simulator-runtime change is in scope.

## Frozen configuration

The authoritative machine-readable pre-registration is
`outputs/actmask/milestone3r_nl_v1/preregistered_config.json`. It fixes six
observable history frames, a `1e-6` matching tolerance, a common 20-step
action horizon, grouped 60/20/20 world-modulo split, fixed seeds, C5/C10/C20,
and all gates. Its SHA-256 is recorded below before data generation.

## Mechanisms and task families

The extension evaluates two new hidden-but-history-identifiable mechanisms in
three procedural GPU-PhysX task families: damped moving capture, hysteretic
moving container, and damped rotating slot.

1. `history_identifiable_damping_drive` has an episode-specific damping and
   drive frequency. Earlier observable frames follow the corresponding damped
   drive; a common synchronization segment makes the final two observations,
   positions, velocities, and timestamps bit-identical. The hidden parameters
   alter only post-decision GPU execution.
2. `hysteretic_mode_memory` has a hidden active mode selected by an earlier
   observable direction crossing relative to its threshold. Both members end
   in the same synchronization segment but receive distinct post-decision
   acceleration/delay responses.

For each mechanism, the common candidate action tracks the pre-registered
positive continuation. Every accepted pair must have one execution success and
one failure under that same action; no branch-specific candidate, horizon,
threshold, or filter is permitted. Success is monotone simulator interception
within common radius 0.055 on steps 6--20, not an analytic label.

## Fairness, matching, and fields

Every accepted pair must match current observable state, final two history
frames, finite-difference input, final-frame timestamps, candidate action,
action horizon, TCP state, and static feature exactly to `1e-6`; only earlier
ordered history differs. The fair input whitelist is exactly the existing
state-observation schema: history, timestamps, visibility/confidence, action,
nominal timing, and TCP state. Damping, frequency, mode, threshold, branch,
family/mechanism IDs, object IDs, future state/acceleration, and labels remain
metadata/oracle-only. Pair audits reject rather than relax violations.

## Probe, full scale, and OOD

The first GPU probe uses at least 16 accepted groups per mechanism across all
three families and at least two candidates per base world. It must make LastTwo,
static, and unordered controls no better than 0.55 pair order while a longer
history method exceeds 0.70. Only then may the full version generate 500 base
worlds/family, C5/C10/C20, <=100,000 executions, and grouped splits.

The probe established that `damped_rotating_slot` with the damping/drive
mechanism has no strict label-flipping pair after its two documented common
physical corrections.  It is therefore excluded before full generation (not
selected by a model).  The full fixed cell set is the remaining five
family--mechanism combinations.  At 500 worlds and C20, this is exactly
5 * 500 * 20 * 2 = 100,000 GPU-PhysX branch executions; each family is still
represented and both mechanisms retain at least two task families.

OOD holdouts are damping range, drive frequency, hysteresis threshold,
execution delay, and each nonlinear mechanism. They never influence model or
hyperparameter choice. The full suite includes the eight listed fair analytic
baselines plus diagnostic oracle, and GRU/TCN/ordered/masked/unordered/static/
action learned controls. Initial selection is three-seed validation-only;
promising fixed recipes receive five test seeds.

## Acceptance and correction policy

The final nonlinear gate is satisfied only when LastTwo loses >=0.20 pair
order on both mechanisms; a learned ordered method beats the strongest fair
analytic method by >=0.05 with paired CI excluding zero; the effect supports
two OOD axes and three task families; full history gains >=0.10 over final-two
only; reversal drop >=0.15; controls are weaker; audits/tests pass; and C20
p95 is <=200 ms. There may be at most two pre-model, documented physical
corrections per mechanism. Invalid pairs are rejected, never loosened.

## Pre-generation hash

`79d04d3204af01386d1a178a874d537a345bd2a8ca94545e750fcb4d50209b9f`

This hash was recorded before any extension data generation or learned-model
evaluation. Any configuration change requires a new version and a new plan;
the present v1 artifacts remain frozen.
