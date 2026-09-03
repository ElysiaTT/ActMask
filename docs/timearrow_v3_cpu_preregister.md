# TimeArrow-v3 CPU preregistration

Frozen before the first v3 run on 2026-09-02.

## Question

After removing the known TimeArrow-v2 endpoint, candidate-ID, strict-flip, and
retained-only shortcuts, can ordered interaction history improve same-state
candidate-action ranking over frozen analytic and history-ablated controls?

This is a CPU synthetic feasibility test. It cannot support claims about vision,
real robots, safety, or open-world dynamics.

## Construction

- Each world contains a one-dimensional controlled dynamical system with latent
  physical parameters and one mechanism family.
- A fixed, signed probe-action sequence produces an eight-frame position history.
- Seven candidate actions are each executed from the exact same saved decision
  state; all seven outcomes are retained.
- The target is observable and generated near one randomly selected candidate
  outcome so that the best candidate varies by world.
- Candidate IDs are independently permuted within every world and are excluded
  from every legal model input.
- Train, validation, ID test, parameter-OOD test, and held-mechanism test contain
  disjoint world IDs.
- Training, validation, and ID/parameter-OOD use linear-damped, saturated, and
  dead-zone response families. The held-mechanism split uses a stateful
  hysteretic response family absent from training.

There is no strict-flip selection and no filtering based on whether paired
candidate outcomes differ.

## Frozen candidate actions and metrics

Candidate amplitudes are `[-1.50, -1.00, -0.50, 0.00, 0.50, 1.00, 1.50]`.
The primary metric is within-world candidate pair-order accuracy computed from
continuous negative target error. Secondary metrics are Top-1 candidate accuracy
and normalized regret. Ties receive 0.5 in pair order.

## Frozen controls

Analytic controls:

1. unit-gain rollout;
2. last-velocity rollout;
3. full-history constant-acceleration rollout;
4. least-squares linear system identification using the known probe sequence.

The declared zero-training rule-search audit additionally checks goal/action
sign alignment, endpoint-displacement/action alignment, last-velocity/action
alignment, response-range-scaled rollout, and probe-response-correlation gain.
These rules are fixed before the first v3 result is produced.

The strongest analytic control is selected by validation pair-order once and is
then frozen for ID and both OOD splits.

Learned controls use identical fixed scikit-learn extremely randomized tree
regression hyperparameters:

1. action and target only;
2. current/final observation, action, and target;
3. unordered history statistics, action, and target;
4. ordered history, action, and target.

No test split may select a method, threshold, architecture, or feature set.

### Pre-result runtime amendment

The first 200-world smoke command was interrupted before it emitted a result:
twenty MLP fits exceeded the intended quick-CPU budget and several reached their
iteration limit. A histogram-gradient-boosting replacement was also interrupted
before result emission after exceeding 90 seconds. Before any v3 metric was
observed, the frozen learner was changed to `ExtraTreesRegressor` with 80 trees,
maximum depth 12, minimum leaf size 3, all features available per split, and all
CPU cores enabled. The features, splits, analytic controls, seeds, gates, and all
data-generation rules are unchanged. These amendments are computational, not
test-selected.

## Decision gates

The v3 direction is **GO for a larger simulator experiment** only if all of the
following hold over five fixed seeds:

1. every split retains exactly seven candidates per world and has no world-ID
   overlap;
2. candidate-ID parity and action-only Top-1 remain within 0.10 of chance;
3. ordered-history mean ID pair-order exceeds the frozen strongest analytic
   control by at least 0.05;
4. ordered-history mean parameter-OOD pair-order exceeds the same frozen control
   by at least 0.03;
5. ordered-history mean ID pair-order exceeds unordered-history MLP by at least
   0.05;
6. normalized ID regret is lower than the frozen analytic control;
7. no post-hoc legal zero-training rule discovered during the audit reaches or
   exceeds ordered-history ID pair-order.

Gate 7 is fail-closed: the current pilot ships a declared rule-search audit, but
absence of a discovered rule is not proof that no rule exists. A GO decision is
therefore only authorization for the next experiment, never a learned-dynamics
claim.

Held-mechanism results are exploratory and cannot independently produce GO.

## Fixed seeds and reporting

Seeds are `7301, 7302, 7303, 7304, 7305`. Every seed and every failed gate is
reported. No best-seed filtering is permitted.
