# ActMask Milestone 2C plan — Synthetic Study Closure and Simulator Readiness

## Frozen starting point

Milestone 2B is preserved at `outputs/actmask/milestone2b/`.  Its resolved
configuration, split digest, checkpoints and reported summary are immutable
reference artifacts.  `configs/actmask/milestone2b_frozen_cpu.yaml` records a
seed-1401 regression AP target (`0.901751 ± 0.010`) and source digests.  No 2B
threshold, data split, model setting or output is overwritten in 2C.

## CPU-only scope and information boundary

This milestone uses only PyTorch/NumPy on CPU.  It does not install or invoke a
simulator, CUDA, VLA, pretrained vision, scene-flow network or GPU dependency.
The observable whitelist remains `points_history`, `visibility_history`,
`timestamps`, `estimated_velocity`, `velocity_confidence`, `action_command`,
`nominal_action_delay`, and `observation_delay`.  Exact future trajectories,
exact kinematics/execution state and exact contacts remain label/oracle-only;
the hidden oracle is excluded from every fair comparison and selection.

## Evaluation protocol frozen before test execution

Training stays compact and uses five deterministic seeds.  Training/validation
generation uses a different master seed namespace from evaluation.  The final
evaluation has at least 500 ID groups and 250 groups for each OOD axis.  Every
identity field (`base_scene_id`, geometry/scenario seed, pair,
trajectory-family, group and candidate-set ID) is checked disjoint across all
partitions.  Thresholds, baseline selection, calibration and final method are
chosen from ID validation only.

Paired group-macro bootstrap (2,000 resamples) and sign-flip permutation
testing (4,096 permutations) report a 95% CI, p-value, sample count and
win/tie/loss count for each Temporal-versus-fair-baseline comparison.  A
positive mean alone is never treated as support: the paired 95% interval must
exclude zero.

## Model, correspondence and baselines

The 2B index-correspondence TemporalActMask is retained only as a diagnostic.
New `SetHistoryTemporalActMask` and `LocalNeighborhoodTemporalActMask` reorder
past frame sets around the final observed anchors using soft or nearest local
spatial association, then recompute motion from those observable matches.  The
selected final model must lose <0.02 AP under independent per-frame permutation
and retain >=85% of clean AP under resampling/birth-death.

Fair observable-only baselines retain current, estimated-velocity,
time-aligned and constant-velocity filters, and add constant acceleration,
alpha-beta, acceleration-state, multi-hypothesis and geometry-with-learned-
residual variants.  The strongest fair method is selected on validation and
frozen before all comparisons.

## Loss and ranking closure

Nine controlled loss configurations test mask-only, contact, utility, each
counterfactual/stability term, their all-auxiliary form and the full 2B
objective.  A loss may remain only if it has a reproducible validation benefit
on its intended metric; selection then prefers fewer retained terms.

Separate physically relabeled 5/10/20-action worlds remove candidate ID/order
cues and add timing, direction, narrow-window, near-miss, distractor and
plausible-success competition.  The selected learned/hybrid utility must show
paired statistical support for +0.03 top-1 success or >=15% regret reduction
against frozen fair geometry on a 10/20-action test.

## Final simulator-readiness gates

The fixed gates are those in the milestone specification: paired supported
mask advantage on ID (or predefined hard-subset justification) and at least
three OOD conditions; history/action evidence >=0.05 AP drops; velocity/action
counterfactual matching >=0.80; irrelevant stability >=0.85; correspondence
robustness; statistically supported hard-ranking utility; demonstrated useful
auxiliary losses only; oracle exclusion; 10-candidate shared CPU feasibility at
>=5Hz; five seeds, manifests and passing tests.  The final report will state
exactly one readiness decision without relaxing any gate.
