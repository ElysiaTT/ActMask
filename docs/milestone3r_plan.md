# Milestone 3R Plan: Robust Signed-Dynamics Reasoning

## Scope

3R is a state-observation extension of frozen Milestone 3Q. It uses the
existing `actmask` CUDA/ManiSkill environment, procedural GPU-PhysX tasks, and
observable-only transformations. It will not modify 3Q files, labels, splits,
horizons, thresholds, artifacts, system packages, or simulator assets.

## Preregistered decision protocol

The clean 3Q result is a freeze check, not a model-selection result. Robust
methods will be selected on grouped validation worlds only. Corruption IDs,
seeds, severities, visibility, timestamps, correspondence variants, and all
analytic/filter hyperparameters are observable or externally specified; success
labels, hidden dynamics, simulator identities, and future trajectories never
enter fair inputs.

The final Stage-3R decision follows the supplied A--E gate unchanged. In
particular, no visual/RGB-D work is authorized unless a learned ordered model
beats the strongest fair analytic estimator by the required held-out robust
margin and confidence interval.

## Fixed validation-selection rule for learned temporal models

The first learned-model sweep uses only world IDs with `world_id % 5 == 3` for
model selection and reserves `world_id % 5 == 4` as the untouched test set.
Training uses IDs with remainder 0, 1, or 2.  One global fair learned recipe
(architecture plus clean/mixed/curriculum training condition) is selected by
its mean signed-pair ordering accuracy across all four task families, three
initial seeds, and the following validation robustness suite: `gaussian_severe`,
`heteroscedastic_medium`, `timestamp_jitter`, `random_dropout`,
`last_frame_dropout`, `burst_occlusion`, `observation_latency`, and
`partial_coordinates`.  This suite was selected for distinct sensing failure
modes, not according to held-out test performance. Ties are broken by the
mean validation score for `last_frame_dropout`, then lexicographically by
`condition` and `method`.

Only the selected recipe receives the two additional preregistered seeds
(`59`, `71`) to form the final five-seed evaluation. Its test scores will be
reported once, accompanied by grouped paired bootstraps against the strongest
fair analytic baseline. No per-corruption test-set model choice is allowed.
