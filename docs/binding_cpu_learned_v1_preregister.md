# BindingCheck learned CPU v1 preregistration

Frozen on 2026-09-03 before implementing or observing any learned-v1 result.

## Purpose and claim boundary

This experiment asks whether a small CPU-trainable verifier can use matched
action-effect histories robustly across independent seeds and controlled OOD
splits. It is a follow-up to the admitted symbolic pre-screen, not a GPU proxy
score and not a claim of superiority over analytic system identification.

The intended contribution is a benchmark/audit protocol. A transparent
spectral system-identification witness remains a mandatory ceiling. If the
learned model merely matches that witness, the result supports transportability
of the audit but does not establish method novelty.

## Frozen response family

Probe directions form a 16-direction grid at 22.5-degree intervals, with
radii 0.20, 0.50, and 0.80. Twin mechanisms differ by exactly one grid step.
For every radius, the second branch's result sequence is therefore a cyclic
permutation of the first branch's sequence. Branch assignment is balanced and
randomized. Both branches have exactly the same probe-action multiset,
probe-result multiset, public rotation, current state, target, candidates, and
candidate slots.

The physical response is assembled from

`g (alpha r - beta r^2) cos(k(theta - phi))`,

where `k` is 2 or 4 during training. Training examples contain one active
harmonic at a time. The held-mechanism split contains an additive mixture of
the two harmonics; this exact composition is absent from training. Its basis
components are not absent, so the split tests compositional mechanism transfer
rather than impossible extrapolation to an unidentified frequency.

Candidate directions lie halfway between probe directions and use radii 0.35
and 0.65. Candidate slots use a cyclic Latin schedule. True candidate utility
is negative absolute distance to a fixed displacement goal and is evaluated
from the noise-free mechanism.

## Frozen splits

Five independent experiment seeds are used: 9401--9405. No best-seed selection
is allowed. For every seed:

- train: 512 twins for harmonic 2 plus 512 twins for harmonic 4;
- ID: 256 twins sampled from the same single-harmonic family;
- parameter OOD: 256 twins with coefficient ranges outside training;
- noise OOD: 256 twins with shared, multiset-preserving Gaussian history noise
  of standard deviation 0.002 before 0.001 quantization;
- held mechanism: 256 twins using the unseen additive harmonic-2/harmonic-4
  composition.

Training coefficient ranges are `alpha in [0.90, 1.20]` and
`beta in [0.10, 0.30]`. Parameter OOD ranges are `alpha in [1.30, 1.60]` and
`beta in [0.35, 0.55]`. Split RNG streams are deterministically separated from
training streams.

## Frozen models

All models are CPU-only and use NumPy/scikit-learn.

1. **Bound spectral Ridge**: fit per-history coefficients over radial powers
   1--2 and harmonics 2 and 4; multiply those coefficients elementwise by the
   candidate basis; standardize and fit `Ridge(alpha=1e-3)` to candidate
   displacement.
2. **Unbound spectral Ridge**: identical features, dimensions, scaler, learner,
   alpha, and sample count, but results are independently sorted within every
   probe radius before coefficient fitting. The result multiset is preserved
   while action-result correspondence is erased.
3. **Analytic spectral witness**: the direct dot product of fitted coefficients
   and candidate basis, with no learned parameters.
4. Frozen marginal controls: class/current, action-only, result-only, candidate
   slot, and arrow-action summaries.

No validation-set hyperparameter selection, feature selection, threshold tuning,
or test-time fitting is permitted. The utility transform is fixed and is not
learned.

## Frozen interventions

The bound model is also evaluated after:

- a pair-preserving cyclic permutation of actions and results together;
- binding erasure by independently sorting results within each radius;
- swapping the two histories in a matched twin;
- reversing candidate slots jointly with their labels;
- rotating histories, candidates, and latent phases together by 37 degrees.

## Metrics

The primary metric is candidate-conditioned twin preference accuracy (TPA)
after centering utilities within each branch. Predicted ties receive 0.5 and
true ties are excluded. Secondary metrics are Top-1 accuracy, mean regret,
normalized regret, informative fraction, Top-1 crossing, and structural
integrity. Metrics are stored for every seed and split.

## Frozen decision gates

The result is `LEARNED_CPU_AUDIT_TRANSPORTS` only if all of the following hold
in the mean over five seeds and no seed has an integrity failure:

1. exact action/result/candidate multiset matching and balanced branch/slot
   construction on every split;
2. Top-1 crossing is at least 0.75 on every split;
3. every marginal control is at most 0.60 TPA;
4. bound Ridge TPA is at least 0.90 on ID, parameter OOD, and held mechanism,
   and at least 0.85 on noise OOD;
5. the minimum single-seed bound TPA is at least 0.85 on ID, parameter OOD,
   and held mechanism, and at least 0.80 on noise OOD;
6. bound minus unbound TPA is at least 0.25 on every split;
7. binding erasure reduces bound TPA by at least 0.25 on every split;
8. pair-preserving permutation changes TPA by at most 0.01;
9. twin-history swap produces at most 0.10 TPA;
10. candidate-slot reversal and coordinate rotation do not move the bound
    model across the applicable admission threshold;
11. normalized regret of the bound model is at most 0.20 on every split;
12. the analytic witness is at least 0.90 TPA on every split, and bound Ridge
    is no more than 0.10 below it.

Any failure yields `LEARNED_CPU_AUDIT_NOT_TRANSPORTED`. Neither outcome
authorizes GPU use. Passing only supports proceeding to an independently frozen
evaluation of external verifier architectures; failing requires retaining the
result and revising or narrowing the direction.

## Reproduction

```powershell
python scripts\cpu_binding_learned_v1.py `
  --output audit\results\binding_cpu_learned_v1.json
```
