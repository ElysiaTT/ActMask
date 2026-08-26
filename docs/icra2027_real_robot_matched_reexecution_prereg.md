# ICRA 2027 matched real-robot re-execution preregistration

Status: **frozen protocol; no robot result has been collected or claimed**.
The machine-readable companion is
`docs/icra2027_real_robot_matched_reexecution_prereg.json`. Its SHA-256 must be
stored in the collected data before the verifier will admit a result. Frozen
SHA-256: `73e77b62704e847b3ff0eb8159d68db65cbd43cc329d34e1f841657267994320`.

## Why this is the next experiment

The existing public UR5 analysis contains real trajectories but not real
counterfactual outcomes: only the logged continuation was executed from each
state. The smallest experiment that repairs that limitation while testing more
than one geometry is a three-family matched physical cell, not another verifier
model. For each pair, the robot starts
from the same measured decision state and executes the same committed candidate
trajectory after two histories with different early order. The target fixture
then realizes the response bound to that history, and the outcome is recomputed
from physical tracking rather than assigned by the fixture controller.

Passing this cell would support only the following claim: across three
procedural physical families, early ordered history distinguishes the physical
outcomes of family-fixed candidate actions after measured state matching. It
would not establish
open-world dynamics, contact-rich manipulation, policy improvement, safety, or
sim-to-real generalization.

## Physical cell

Use a robot arm with timestamped joint/TCP feedback and an independently
tracked actuated target, such as a marked two-axis carriage plus turntable or
an equivalent programmable fixture. The three frozen families are
`linear_x_intercept`, `linear_y_intercept`, and `circular_intercept`. They
change the target-motion geometry while retaining the same history length,
matching contract, outcome radius, and evaluation window. The primary setup
is a touch-free intercept so that success can be measured by relative
3-D distance without requiring contact. A human operator owns the robot,
fixture, protective stop, workspace clearance, and emergency stop.

The fixture generates a six-frame pre-decision history. Branch A's first four
target positions must match the reverse of branch B's first four positions;
the last two positions are a common stationary synchronization suffix. At the
decision time, measured target position and velocity, TCP position, robot
joint position, and joint velocity must match within the frozen tolerances.
The target must be stationary to within 5 mm/s at that instant. A controller mode
installed before the common suffix produces the branch-specific future target
motion only after the decision. The robot receives one fixed, precommitted
Cartesian TCP trajectory in both branches. The controller may internally use
joint space, but the recorded candidate is the same timestamped `tcp_xyz_m`
trajectory within a family so that all geometric controls are reproducible.
Different families may use different precommitted candidates; no candidate may
change between A/B pairs within its family.

This is deliberately procedural. The physical intervention tests whether the
paper's admission logic survives measurement noise and re-execution; it does
not turn the controlled TimeArrow family into a natural manipulation task.

## Collection order and sample size

Collect at least 12 complete A/B pairs per family, 36 total. Within each pair,
run both branches from a fresh reset. Alternate or block-randomize which branch
is executed first inside every family; each family's final imbalance between
A-first and B-first pairs may not exceed two, and each trial records its
execution order. Interleave family blocks rather than completing only the
easiest family first. Freeze calibration, tracking,
candidate trajectory, evaluation window, success radius, and controller code
before the first counted pair.

Do not remove a pair after observing its outcome. Tracking loss, a safety
abort, matching failure, or a changed action must remain in the data and makes
the frozen admission gate fail until a new, separately versioned collection is
preregistered. Dry runs are permitted only before the preregistration hash is
attached to the counted dataset.

## Logged evidence

The data file follows `actmask-real-robot-matched-data-v1` and contains a
top-level `metadata` object plus all `trials`. Every pair belongs to one frozen
family and has exactly one A and one B trial. Each trial stores:

- family and pair identifiers, six target-history positions, and relative
  timestamps;
- measured decision target position and velocity, TCP position, joint
  positions, and joint velocities;
- the commanded candidate timestamps and trajectory, identical in every run;
- post-decision target and TCP tracking, tracking-valid flags, and timestamps;
- the recorded success bit, outcome source, execution order, and safety-abort
  flag.

The verifier recomputes success as whether measured target--TCP distance ever
falls at or below 15 mm during 0.45--0.75 s. Planned target motion or a
controller-side label is inadmissible as the outcome source. It also checks
that the candidate starts at the measured decision TCP and that measured TCP
feedback remains within 5 mm of the timestamp-interpolated committed candidate
inside the evaluation window. A sent command without matching robot feedback
is not counted as an execution.

## Frozen shortcut ladder

The same collected traces are evaluated with fixed, non-learned scores; no
model selection or fitting is permitted. All scores use only the declared
observable history and committed TCP candidate inside the frozen evaluation
window:

1. `action_only`: a constant score because the candidate is byte-identical;
2. `current_only`: closest candidate distance to the measured current target;
3. `unordered_history`: closest candidate distance to the orderless mean of
   all six target observations;
4. `final_two_velocity`: constant-velocity prediction from only frames 5--6;
5. `multi_frame_linear_velocity`: least-squares velocity over all six ordered
   frames, anchored at the measured current target;
6. `early_endpoint_velocity`: velocity from ordered frames 1 and 4, also
   anchored at the current target.

Pair-order gives 1 when the score of the physically successful A execution is
higher, 0 when reversed, and 0.5 on a tie. In aggregate, each shortcut control
must stay at or below 0.625 and must not beat chance by a one-sided exact sign
test (`p >= 0.05`, excluding ties); it must also stay at or below 2/3 within
every family. At least one ordered fair estimator must reach 0.80 in aggregate
with `p <= 0.01` and at least 0.75 in every family. These gates prevent one
easy family from hiding shortcut leakage or an ordered-history failure in
another.

The ladder cannot authorize a learned method. If an ordered fair estimator
reaches 0.80, the physical cell is explicitly marked analytically saturated
and learned-method headroom is rejected. If none does, headroom remains
unresolved rather than being automatically granted.

## Frozen gates

All 36 or more complete pairs, with at least 12 from every family, must pass
the early-swap, nontrivial early-order separation, history-timing,
common-suffix, decision-target pose/velocity,
decision-TCP, joint-position, joint-velocity, candidate-hash, candidate
execution fidelity, tracking,
metadata, and safety checks. Each family's first-branch imbalance must be at
most two.
At least 80% of pairs must be correctly discordant (A succeeds and B fails),
at most one pair may be discordant in the wrong direction, the one-sided exact
sign-test value must be at most 0.01, and the 95% Wilson lower bound on the
correct-discordance fraction must exceed 0.5. Within every family, at least
75% must be correctly discordant and at most one may be wrong-discordant. The
four shortcut controls and two ordered fair estimators must also pass the
aggregate and family-wise ladder gates above.

Run the frozen verifier with:

```bash
python scripts/verify_icra2027_real_robot_matched_reexecution.py \
  --data /path/to/collected_real_robot_data.json
```

The script emits every gate and exits nonzero if the result cannot be admitted.

## Manuscript decision rule

Do not add an empty method or preliminary robot number. If and only if every
gate passes, replace the current public-UR5 logged-continuation paragraph and
its admission-table row with this matched physical cell; report the number of
complete pairs, correct/wrong/tied counts in aggregate and by family, exact
sign-test value, Wilson bound, maximum matching errors, strongest family-wise
controls, hardware, and preregistration hash. If any gate fails,
retain the current honest UR5 rejection and report the failed preregistered
attempt only outside the eight-page paper unless it changes the central
conclusion.
