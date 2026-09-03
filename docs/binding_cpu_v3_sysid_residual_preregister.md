# BindingCheck-CPU-v3 strong-baseline-first preregistration

Frozen on 2026-09-03 before implementation and before observing any v3
performance result.

## Question and terminal rule

CPU-v2 showed that a raw paired ExtraTrees verifier uses action-effect binding
but loses on every seed and split to per-history Poly2/RBF system
identification. v3 tests one final method hypothesis:

> Can a small learned gate/residual/uncertainty head, trained only from generic
> system-ID diagnostics, lower candidate regret beyond every fixed non-oracle
> predictor under heterogeneous and unseen response laws?

This is a terminal method test. If any required split fails, the result is
`CPU_V3_NO_GO`; no v3 architecture, threshold, seed, mechanism, or held split
may be modified to chase a pass. The new-verifier method claim stops and the
project narrows to the Counterfactual Action-Effect Binding Audit benchmark.
Neither outcome authorizes GPU use.

## CPU and information boundary

The experiment uses NumPy and scikit-learn on CPU. PyTorch, CUDA, GPU
simulation, vision encoders, latent mechanism labels, hidden orientations,
generator parameters, and generator-matched Fourier bases are forbidden model
inputs. Candidate truth is used only as a training target on the frozen
training partition and for evaluation metrics.

## Frozen data generator

An episode is a twin with two branches. Branches share public target, history
actions/times, history length, candidates, family, parameter ranges, and action
coverage. Hidden response orientations differ by a random 50--130 degrees and
branch assignment is balanced. Actions and candidates are continuous 2-D
commands; candidates cannot be within 0.03 Euclidean distance of a history
action.

History coverage is nonuniform: 70% of probes are concentrated around a random
direction and 30% are uniform. Radius is sampled on `[0.15, 1.0]`. Observation
noise is heteroscedastic, scaled by action radius. Each branch has sixteen
candidates and utility is negative absolute error from a public target sampled
on `[-0.5, 0.5]`.

Five families coexist in train, ID, parameter-OOD, sparse-history, and
noise-OOD, balanced before shuffling:

1. oriented smooth `tanh` saturation with a cross-action term;
2. oriented rational radius-dependent drag;
3. a localized nonlinear Gaussian bump added to a smooth response;
4. dead-zone plus hard clipping/saturation and non-additive perpendicular
   coupling;
5. mild nonstationarity: a rational response with linearly drifting bias and
   gain over public normalized history time; candidate time is 1.0.

The structure-held-out family is absent from all training episodes. It combines
an asymmetric square-root threshold response with a radial ring and a
three-lobed angular modulation. It is not a linear combination or parameter
extension of the five training families.

## Frozen seeds, sizes, and splits

- Official seeds: 12001, 12002, 12003, 12004, 12005.
- Train: 640 twins, exactly 128 per training family before shuffling.
- Meta-fit/calibration division: the first 480 generated train twins fit all
  learned heads and raw controls; the final 160 calibrate uncertainty only.
- ID: 256 twins, training ranges, history length 16--28, base noise 0.01.
- Parameter OOD: 256 twins, disjoint gain/slope/drag/coupling/drift ranges,
  history length 16--28, base noise 0.015.
- Sparse history: 256 twins, training ranges, history length 6--12, base noise
  0.01.
- Noise OOD: 256 twins, training ranges, history length 10--20, base noise
  0.05.
- Held structure: 256 twins, held family only, history length 10--22, base
  noise 0.015.

Split RNG streams are fixed offsets from the official seed. No validation
search, seed replacement, result filtering, resampling, or test-time fitting is
allowed.

## Frozen per-history predictors

Every history produces candidate responses from five generic predictors:

1. nearest observed action;
2. inverse-distance five-nearest neighbors;
3. candidate-centered local-linear ridge with Gaussian distance weights,
   bandwidth 0.45 and ridge 0.01;
4. global Cartesian Poly2 ridge over `[1,x,y,x^2,xy,y^2]`, ridge 0.01;
5. fixed RBF kernel ridge, length scale 0.45 and ridge 0.01.

For every predictor, leave-one-probe-out mean absolute error is computed without
refitting hyperparameters. Linear/RBF analytic LOO identities may be used;
nearest/kNN/local-linear use deterministic held-one-probe calculations.

The uniform mean of the five predictors is a fixed baseline. A candidate-wise
oracle selector is reported only as an unattainable ceiling and never enters
admission.

## Frozen hybrid features and training

Candidate-level public features are:

- the five predictor responses;
- the five per-history LOO errors;
- nearest, third-nearest, and fifth-nearest candidate coverage distances;
- local five-neighbor result mean and variance;
- predictor standard deviation and range;
- action-radius extrapolation ratio;
- history length fraction;
- heteroscedastic noise estimate from nearest history-action pairs;
- early-minus-late result drift statistic;
- candidate coordinates/radius, target, and public candidate time.

Five error heads independently predict each component's absolute candidate
response error using `ExtraTreesRegressor(n_estimators=64, max_depth=12,
min_samples_leaf=8, max_features=1.0, bootstrap=False)`. Predicted errors are
clipped nonnegative and converted to weights by softmax of `-error/0.05`.

The weighted response is passed to one residual head using the public features,
component weights, weighted response, weighted predicted error, and component
disagreement. The residual head is
`ExtraTreesRegressor(n_estimators=96, max_depth=14, min_samples_leaf=5,
max_features=0.8, bootstrap=False)`.

Raw uncertainty before calibration is weighted predicted error plus predictor
disagreement plus the standard deviation across residual-head trees. The 80th
percentile of `absolute_error/raw_uncertainty` on the frozen 160-twin calibration
partition is the single multiplicative calibration factor. It is never refit
per test split.

## Frozen learned controls

The CPU-v2 candidate-relative padded raw representation is independently
retrained on the 480 meta-fit twins with 64-tree ExtraTrees, maximum depth 18,
minimum leaf 2, and `max_features=0.7`. Equal-capacity variants are:

- raw paired tokens;
- raw unbound results independently sorted from actions;
- action-only;
- result-only;
- public-context-only.

Tokens also contain public normalized time. These are comparison baselines, not
inputs to the hybrid.

## Frozen interventions

The already-trained hybrid is evaluated after:

- pair-preserving permutation of complete action-result-time tokens;
- binding-breaking result permutation with actions/times fixed;
- twin-history swap;
- candidate-slot reversal;
- the independently generated sparse-history split.

Permutation seeds are fixed functions of official seed and split. No shuffle is
selected after evaluation.

## Metrics and uncertainty

Primary metrics are per-twin candidate normalized regret and Top-1 accuracy.
TPA is diagnostic. Results also include mean regret, per-twin arrays, hard/
medium/easy oracle-gap terciles, five-seed mean/standard deviation, and all
per-seed values.

Calibration reporting includes empirical coverage and mean width of the frozen
80% response interval plus Spearman correlation between uncertainty and absolute
response error.

A deterministic 1,000-replicate hierarchical bootstrap resamples official
seeds and twins to form a 95% interval for strongest-baseline-minus-hybrid
normalized regret.

## Frozen admission gates

Every one of ID, parameter OOD, sparse history, noise OOD, and held structure
must pass all gates:

1. all finite/shape/shared-action/shared-time/branch-balance/history-range/
   candidate-noncollision integrity checks pass;
2. hybrid normalized regret is at least 10% lower than the strongest admission
   baseline, where admission baselines include all five fixed predictors,
   uniform ensemble, all five learned raw controls, and fixed public controls;
3. hybrid regret beats that strongest baseline in at least four of five seeds;
4. the hierarchical-bootstrap lower 95% bound of baseline-minus-hybrid regret
   is strictly positive;
5. hybrid Top-1 exceeds the strongest baseline Top-1 by at least 0.03;
6. binding breaking increases hybrid normalized regret by at least 0.05 and
   reduces diagnostic TPA by at least 0.05;
7. pair-preserving permutation changes regret and TPA by at most 0.005;
8. twin-history swap increases regret by at least 0.05;
9. candidate-slot reversal changes regret by at most 0.005;
10. anti-saturation: hybrid TPA is below 0.95 and normalized regret is at least
    0.01;
11. the frozen 80% uncertainty interval has empirical response coverage in
    `[0.70, 0.90]` and uncertainty/error Spearman correlation at least 0.20.

Any failure produces `CPU_V3_NO_GO`. Passing all five splits produces
`CPU_V3_GO`, which authorizes only a separately frozen external CPU evaluation
plan, not GPU use.

## Required artifacts

```powershell
python scripts\cpu_binding_v3_sysid_residual.py `
  --output audit\results\binding_cpu_v3_sysid_residual.json
```

The run must retain the preregistration, standalone CPU implementation, raw
per-seed/split/twin JSON, bootstrap/difficulty/calibration results, final audit
report, and terminal method decision.
