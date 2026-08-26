# Milestone 3Q Handoff

## Current phase

Phase 0 and the signed-dynamics Phase-1/full-generation gate are complete.
The Option-1 container replacement and fixed-phase rotating capture window
passed local probes, followed by accepted full four-task data generation and
five-seed observable baselines.

## Work completed

The 3P order-invariant audit proves that its matched task is solved by unordered
motion statistics: all listed methods reach Top-1 1.0000. 3Q therefore cannot
claim signed-direction necessity from 3P. The pre-registered 3Q closed-loop
pair generator is implemented, as are positive/negative hidden modes for the
three existing GPU-PhysX tasks (without changing frozen 3P 0/1 behavior) and
the fourth `CrossingObstacleTiming` task. All four tasks pass an 8-way CUDA
smoke; the exact closed-loop unordered-statistics test passes. The new rotating
task freezes phase `0.0`, radius `0.13` m, capture radius `0.06` m, two-step
persistence, and a common 13-step action/contact horizon after its trace-based
first bounded correction.

The complete generator produced 80,000 candidate executions (20,000 per task)
with 2,500 accepted signed groups per task and a maximum audit error of
`1.49e-08`. The 3- and 5-seed observable evaluations give 0.5 for static,
action, unordered, and sorted controls; 1.0 for signed finite difference and
ordered temporal MLP; and 0.0 after exact reversed history. Mechanism OOD
phase/delay gives the same 0.5 versus 1.0 separation.

## Blocker

The original rotating task remains rejected and untouched. Its replacement,
`FixedPhaseRotatingCaptureWindow`, now supplies 40 accepted signed groups: all
are GPU-PhysX `(success, failure)` pairs with zero matching-audit error. Static,
action, template, angular-magnitude, unordered-history, and sorted-history
controls are at 0.5 pair order accuracy; signed finite-difference and ordered
temporal MLP are at 1.0 and fall to 0.0 under exact reversed history.

## Evidence

`outputs/actmask/milestone3q_signed_dynamics/order_invariant_audit.json` and
`order_invariant_audit_records.json` contain Phase-0 results. The retained
`probe/moving_container_placement_metadata.jsonl` retains the rejected old
container evidence. `probe_v2/` retains the new window-task probe and rotating
task partial metadata; the generator raises before accepting a partial task.
The accepted fixed-phase probe, anti-shortcut report, four-family CUDA smoke,
and full-regression log are under `rotating_probe_v2/`.
Full data, 3-/5-seed reports, final regression, and `final_gate.json` are under
`full_four_task/`.

## Attempts already made

The audit's NumPy `.square()` typo was corrected and rerun. The Option-1 window
task uses frozen capture thresholds and removed only an unintended shared TCP
startup delay. Rotating task used two bounded timing fixes as above. No matching
tolerance, split group, simulator success predicate, or frozen 3P result was
weakened. The fixed-phase task required one permitted common horizon correction
(20 to 13 control steps) before anti-shortcut evaluation; no second correction
was used.

The full regression suite was rerun before a rotating-task decision: **138
passed** in 57.17 seconds. The log is
`outputs/actmask/milestone3q_signed_dynamics/pre_rotation_decision_regression.log`.

## Exact resume instruction

`The signed-dynamics gate is passed. Preserve all generated artifacts and the
frozen 13-step rotating horizon. Any future extension should add ranking/C5/C10/
C20 reporting as a separately versioned evaluation layer, without changing
these data, labels, or the accepted observable-only conclusion.`
