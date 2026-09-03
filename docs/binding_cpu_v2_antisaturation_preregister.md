# BindingCheck-CPU-v2 anti-saturation preregistration

Frozen on 2026-09-03 before implementation and before observing any v2 result.

## Status and purpose

learned-v1 is explicitly downgraded to a plumbing/identifiability pilot because
its unbound representation is identical across twin branches and its bound
representation contains the generator's exact Fourier basis. v2 asks a harder
question: can a generic CPU model operating on raw action-result tokens improve
candidate choice under partial probing and truly structure-held-out response
laws?

No v1 threshold, architecture, seed, or outcome is reused to tune v2. No GPU,
PyTorch, simulator, or visual backbone is allowed.

## Frozen episode construction

Each twin has two branches sharing the same public target, history actions,
history length, candidates, coverage distribution, mechanism family, and scalar
parameter ranges. The branches differ only in a hidden response orientation,
with balanced random branch swapping. Unlike v1, result marginals are not forced
to match exactly; learned result-only and unbound controls must reveal any
remaining marginal signal rather than receiving identical branch tensors by
construction.

Actions are continuous 2-D commands. Histories contain between 8 and 28 probes.
Seventy percent of probes come from a randomly oriented concentrated component
and thirty percent from a uniform component, producing nonuniform coverage.
Candidates are independently sampled continuous actions and do not coincide
with history actions. Each branch has sixteen candidates and a public scalar
target. Utility is negative absolute target error.

Training and ID use two response families:

1. smooth oriented saturation: a `tanh` response of oriented linear and
   cross-action terms;
2. rational drag: an oriented response divided by a radius-dependent drag term.

Parameter OOD uses the same families with non-overlapping gain, slope/drag, and
coupling ranges. Noise OOD uses shorter histories and heteroscedastic result
noise. The held-structure split uses a dead-zone, hard saturation, and a
non-additive perpendicular coupling. That piecewise mechanism does not appear
in training and is not a linear combination of the training laws.

## Frozen sizes and seeds

- Experiment seeds: 10501, 10502, 10503, 10504, 10505.
- Train: 512 twins, balanced across the two training families.
- ID: 256 twins, training parameter ranges, history length 14--28.
- Parameter OOD: 256 twins, disjoint parameter ranges, history length 14--28.
- Noise OOD: 256 twins, history length 10--22 and heteroscedastic noise scale
  0.040.
- Held structure: 256 twins, dead-zone/saturation/non-additive response and
  history length 10--22.
- Sixteen candidates per branch.

All split RNG streams are deterministically separated. No validation-based
hyperparameter selection, seed replacement, resampling, or outcome filtering
is permitted.

## Frozen representations and learned models

Histories are padded to length 28. For each candidate, actions are expressed in
candidate-relative Cartesian coordinates and canonically sorted by relative
angle and radius. Every token contains only relative action coordinates, action
radius, observed scalar result, and a mask. Public candidate coordinates,
target, and history fraction are appended. There are no Fourier, polynomial,
latent-parameter, mechanism-ID, or oracle features.

All learned controls use `ExtraTreesRegressor` with 64 trees, maximum depth 18,
minimum leaf size 2, `max_features=0.7`, bootstrap disabled, and the experiment
seed as random state:

1. paired raw tokens;
2. unbound raw tokens, with observed results sorted independently of actions;
3. action-only, with the result channel zeroed;
4. result-only, with action channels zeroed and results independently sorted;
5. public-context-only, with all history token channels zeroed.

All five models use the same padded dimensionality, training examples, estimator
class, and hyperparameters. They predict candidate response; the utility
transform is fixed.

## Frozen non-learned baselines

- nearest historical action;
- inverse-distance 5-nearest neighbors;
- per-history quadratic ridge over generic Cartesian monomials
  `[1,x,y,x^2,xy,y^2]`;
- per-history fixed RBF kernel ridge, length scale 0.45 and regularization 0.01;
- history-result mean.

These are deliberately generic and do not match all training or held response
families. No oracle mechanism baseline participates in admission decisions.

## Frozen interventions

- pair-preserving permutation randomly moves complete action-result tokens;
- binding-breaking permutation moves results independently within a history;
- twin-history swap exchanges histories across branches;
- candidate-slot reversal jointly moves candidates and truth.

The same trained paired model is used for all interventions. Permutation seeds
are fixed functions of experiment seed and split; no favorable shuffle is
selected.

## Primary and diagnostic metrics

Primary metrics are per-twin candidate normalized regret and Top-1 accuracy.
For every model, normalized regret is the utility loss of its selected candidate
divided by that branch's candidate utility range, then averaged over branches
and twins. Candidate-conditioned twin preference accuracy (TPA) is diagnostic
only.

Every split reports per-seed values and per-twin arrays, paired improvement over
each baseline, and hard/medium/easy terciles defined before prediction by the
oracle best-versus-second-best utility gap. A deterministic 1,000-replicate
hierarchical bootstrap resamples seeds and twins to form a 95% confidence
interval for regret improvement over the strongest non-oracle baseline.

## Frozen admission gates

The direction is `CPU_V2_NONTRIVIAL_SIGNAL` only when every split satisfies all
of the following in the frozen five-seed aggregate:

1. all shape, finite-value, shared-action/candidate, branch-balance, length, and
   candidate-noncollision integrity checks pass;
2. paired raw normalized regret is at least 10% lower than the strongest
   non-oracle baseline named above, including all learned controls;
3. paired raw has lower regret than that baseline in at least four of five
   seeds;
4. the hierarchical-bootstrap lower 95% bound for baseline-minus-paired regret
   is strictly positive;
5. paired raw Top-1 exceeds the strongest baseline Top-1 by at least 0.03;
6. binding breaking increases normalized regret by at least 0.05 and reduces
   diagnostic TPA by at least 0.05;
7. pair-preserving permutation changes normalized regret and TPA by at most
   0.005;
8. twin-history swap increases normalized regret by at least 0.05;
9. candidate-slot reversal changes normalized regret by at most 0.005;
10. anti-saturation: paired TPA is below 0.95 and paired normalized regret is
    at least 0.01. Crossing either easy-task ceiling rejects the split even if
    every accuracy comparison otherwise passes.

The strongest baseline is selected only for conservative reporting: paired raw
must beat every frozen non-oracle baseline, so selection cannot improve the
paired model. Any failed split yields `CPU_V2_NO_GO`. Thresholds may not be
changed after the result. A NO-GO is retained and used to decide whether the
method direction should be abandoned or the benchmark contribution narrowed.
Neither result authorizes GPU use.

## Reproduction

```powershell
python scripts\cpu_binding_v2_antisaturation.py `
  --output audit\results\binding_cpu_v2_antisaturation.json
```
