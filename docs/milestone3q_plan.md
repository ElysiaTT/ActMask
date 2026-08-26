# Milestone 3Q Plan: Signed-Dynamics Necessity

## Scope

3Q remains state-only and uses the existing CUDA-enabled `actmask` environment,
ManiSkill/SAPIEN GPU PhysX, and procedural collision geometry only. No Vulkan,
Isaac Sim, RoboTwin, external asset, pretrained model, visual backbone, or
system package action is authorized. Previous 3P artifacts are immutable.

## Phase-0 finding (frozen)

The 3P matched pilot is solvable without temporal direction. On 3P
original-train to matched-test, scalar magnitude statistics, a shallow tree,
small MLP, DeepSets, coordinate-sorted histories, velocity-magnitude-only, a
timestamp-free temporal MLP, and an unordered-multiset MLP each reached
aggregate Top-1 1.0000 across all three task families. The audit is
`outputs/actmask/milestone3q_signed_dynamics/order_invariant_audit.json`.

Thus 3Q must not reuse 3P's shifted-history construction as signed evidence.
This conclusion is a prerequisite, not a model-selection result.

## Pre-registered signed-order data rules

Every signed-order counterfactual pair will use a closed four-frame loop
`[A, B, C, A]` versus `[A, C, B, A]`. It has identical current state `A`, exact
unordered frame multiset, coordinate-sorted statistics, centroid/covariance,
path length, per-frame displacement magnitudes, motion energy, first-to-last
distance, candidate action, and static feature vector. The final inbound motion
sign is opposite. A hidden future velocity/dynamics branch consistent with that
observable signed history is applied only after the common decision state; GPU
PhysX execution produces the success label.

Four signed mechanisms are generated per task: loop/reversal, opposite signed
velocity, acceleration-sign, and signed phase/delay. The rotating task also
uses clockwise/counter-clockwise angular velocity. A fourth procedural task,
`CrossingObstacleTiming`, evaluates whether a fixed TCP trajectory reaches a
window before a moving obstacle/gate closes.

The generator will reject a pair unless current observation, action, static
vector, unordered features, and required speed magnitudes match within `1e-6`,
the signed sequence differs, and GPU execution flips success. Pairs stay in the
same grouped split. These definitions are frozen before signed-regime results.

## Fixed-phase rotating capture window (Phase-1 authorization)

`FixedPhaseRotatingCaptureWindow` is a separate 3Q-only task and does not alter
the frozen 3P `RotatingTargetInteraction` task.  It uses a kinematic,
collision-only cubic capture window with half-extents `(0.04, 0.04, 0.04)` m.
The window orbits the payload-centred point at radius `0.13` m and height
`0.05` m, from fixed initial phase `0.0`.  The TCP is successful only after it
is inside a fixed `0.06` m relative-position radius for two consecutive control
steps; per-episode success is the monotonic OR of those GPU-PhysX execution
steps.  There is no branch-specific geometry, phase, threshold, action, or
horizon, and no relative-speed exception.

The four preregistered common magnitude schedules are constant signed angular
speed `0.22` rad/control-step, signed acceleration `0.028*n^2` rad, a common
phase-delay schedule `0.32 + 0.30*max(n-4, 0)` rad, and the exact
history-reversal counterpart.  Sign is oracle-only after the common decision
state.  The fair input contains only the four-frame observable history, common
TCP action plan, timestamps, and current state; phase, future sign, mechanism,
and success remain diagnostics/labels.

The first four-world probe used a common 20-step execution horizon.  Its
trajectory trace showed the negative constant-speed and delayed branches first
approach the positive TCP path at steps 14--15 after a near-full orbit, causing
an unintended second intersection and `(success, success)`.  Before any model
evaluation, the first authorized bounded correction freezes the common
rotating-task horizon at **13 control steps**.  Positive branches capture at
steps 6--7; negative branches have at most one qualifying contact by step 13,
so cannot satisfy the unchanged two-step persistence rule.  This is a common
contact-horizon correction, not filtering or a threshold change.  No second
bounded correction has been used.

## Target scale and split

Each of four task families will contain 1,000 base worlds total: 250 worlds in
each of original, static-balanced, matched, and signed-order regimes. Each world
has 20 candidates, enabling C5/C10/C20 evaluation and yielding 80,000 candidate
executions total, below the 100,000 first-run cap. Base worlds use a grouped
60/20/20 split. Mechanism, parameter-family, and base-world holdouts are kept
separate for OOD diagnostics.

## Required evaluation

Static, order-invariant, and signed temporal models receive only observable
history, timestamps, candidate action, and TCP state. We will report the seven
ranking metrics for each regime; correct/reversed/permuted/unordered/duplicated/
zeroed/sign-removed/sign-magnitude/timestamp-shuffled/mismatched histories;
ID/OOD; C5/C10/C20 latency; anti-shortcut training; three-seed diagnostic and
five-seed final comparison. Final gates are exactly those in the 3Q task brief.
