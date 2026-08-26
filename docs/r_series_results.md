# R-Series results

## Final decision

**E. BASELINES_SATURATE.** The proposed `RealRelDynVerifier` was not trained.
This is a gate-respecting result, not a failed run: standard baselines solve
the logged task beyond the pre-registered 0.88 saturation ceiling even after a
single, pre-registered harder split.

## R0/R1

- Surveyed DROID, ASU TableTop, RH20T, RoboMIND/RoboMIND 2.0 and BridgeData
  V2; source details are in `dataset_survey.json` and
  `docs/r_series_dataset_survey.md`.
- DROID `droid_100` was the initial 2.19-GB primary.  Its adapter validated
  every TFRecord CRC and yielded 46 language-valid trajectories; 54 genuinely
  blank-language logs were excluded without fabricating instructions.  Its
  first R2 was retained as invalid because all usable full instructions were
  unique and the first sampler's semantic fallback was detected before R3.
- The automatic backup, Open X-Embodiment ASU TableTop, completed successfully:
  110 real UR5 episodes, 18 official action/object tasks, RGB, 7-D
  proprioception/actions, and dataset-provided EE/six-object 6-D poses.
  Its adapter audit passed for all 110 episodes.

## Strict ASU R2

- Selected 84 repeated-task episodes across 11 official `action_inst ×
  goal_object` tasks; 1,151 candidate sets / 5,755 rows.
- IID-style split: 40/22/22 independent train/validation/test episodes, with
  at least two trajectories for every task in every split.
- Harder split: train objects bread/bottle/cube, validation pepsi, test
  unseen coke/milk; 43/13/28 episodes.  Same-task negatives still remain
  exact within each partition.
- Every candidate set has the real continuation plus strict non-overlapping
  same-episode, same official-task other-episode, different-task, and reversed
  chunks.  No fallback pools were used; all source semantics and split matches
  pass.  Future object-minus-EE pose change is target-only.

## R3 metrics (pair-order accuracy)

| Split | Trivial controls | Best standard | Future-relation predictor |
| --- | --- | --- | --- |
| IID-style | action 0.575; language/task/source 0.500 | current-frame 0.974 validation / 0.971 test | 0.837 validation / 0.841 test |
| Unseen-object | action 0.534; language/task/source 0.500 | proprio-action GRU 0.972 validation / 0.950 test | 0.785 validation / 0.781 test |

All 66 run-level raw-score artifacts across the two attempts were independently
recomputed into their saved ranking metrics; both consistency audits pass.
The held-out-object result removes the simplest action/language/task/source
shortcuts yet remains saturated for an ordinary proprio-action GRU.

## Claim boundary and recommendation

This is **logged action-conditioned consistency from real UR5 trajectories**,
not true counterfactual physical-success prediction.  Preserve the ASU
adapter/benchmark/audit track.  A future method track should seek a larger
real source with matched alternate-action outcomes, contact/object-state
ambiguity, or replayable re-execution; adding a relational model to this
saturated task would not be credible method evidence.
