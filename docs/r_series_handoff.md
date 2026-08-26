# R-Series handoff

## Completed state

The active R-Series decision is `E. BASELINES_SATURATE`; see
`outputs/actmask/r_series_real_robot_verification/final_decision.json`.
No `RealRelDynVerifier` checkpoint should be expected: its R4 gate correctly
rejected training after standard baselines saturated on both splits.

## Preserved attempts

- `processed_subset/`: DROID R1 adapter pilot (46 language-valid episodes).
- `task_formulation/`: invalid DROID R2 attempt, preserved only for audit;
  never use it for a baseline comparison.
- `asu_attempts/asu_ur5_object_relation_v1/`: strict repeated-task ASU split
  and all 33 R3 run artifacts.
- `asu_attempts/asu_ur5_object_relation_heldout_goal_object_v1/`: required
  unseen-object repair split and all 33 R3 run artifacts; this is the active
  R5 attempt.

Each attempt has its own frozen config, task data hash, preprocessing,
checkpoint where learned, raw scores, metrics, and raw-score recomputation
audit.  The root report files point at the active held-out-object attempt;
the IID evidence is not overwritten.

## Safe next research move

Do not tune this dataset to force a method win.  If the goal changes to a new
method track, choose a fresh public real-robot source with one of: matched
alternative action executions, a simulator/digital twin, force/contact
outcomes, or substantially richer object-state uncertainty.  Create a new
versioned attempt rather than modifying either ASU attempt.
