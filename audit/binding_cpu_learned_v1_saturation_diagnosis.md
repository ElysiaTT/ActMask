# BindingCheck learned CPU v1 saturation diagnosis

Date: 2026-09-03
Post-result status: **downgraded to plumbing/identifiability pilot**

## Why the 0.5-versus-1.0 table is suspicious

The bimodal result is largely forced by the construction rather than discovered
by a generic learned verifier.

### 1. The unbound score is mathematically pinned to 0.5

The two branches have identical action and result multisets. The unbound
preprocessor independently sorts results within every probe radius. After that
operation, branch 0 and branch 1 have exactly identical result arrays. Their
action arrays and candidates were already identical. Consequently their fitted
coefficients and candidate features are also identical. Candidate-conditioned
twin preference assigns 0.5 to every predicted tie.

A direct replay of the frozen ID seed 9401 gives:

```text
erased_result_branch_maxdiff  0.0
erased_feature_branch_maxdiff 0.0
```

Thus `unbound TPA = 0.5000` is an algebraic consequence, not an empirical model
failure with meaningful uncertainty.

### 2. The bound model receives the generator's exact sufficient basis

The generator is composed only of radial powers 1--2 and angular harmonics 2
and 4. `_action_basis` supplies exactly those eight terms. Each history is first
least-squares fitted in that correct basis, then the fitted coefficients are
multiplied by the candidate's same basis. Ridge only needs to learn the dot
product. It is therefore a calibrated analytic system identifier with a learned
last layer, not a raw-history verifier.

On the same replay, analytic displacement RMSE is `0.0001978`, while the mean
within-episode utility span is `0.06068`: the error scale is roughly 300 times
smaller than the ranking range. Near-one TPA is expected.

### 3. The held mechanism is not structurally held out

Training contains harmonic 2 and harmonic 4 separately. The held response is
hard-coded as `0.65 * h2 + 0.55 * h4`, and both components lie in the supplied
basis. This tests linear compositional closure inside the known hypothesis
class. It is not evidence for transfer to an unknown response structure.

### 4. TPA hides a weaker exact-choice result

Bound Top-1 accuracy is only 0.6762 (ID), 0.6492 (parameter OOD), 0.5270
(noise OOD), and 0.8043 (held). Near-one TPA and near-zero regret coexist
because the dense candidate grid contains many nearly equivalent candidates.
TPA is useful as an identifiability diagnostic, but it cannot be the sole or
primary method-performance metric.

## Corrected interpretation

learned-v1 verifies four pieces of plumbing:

1. the frozen metric reacts to the intended sign change;
2. the matched construction removes declared marginal shortcuts;
3. the exact spectral witness can recover the planted mechanism;
4. action-result binding erasure removes that planted information.

It does **not** establish that a generic learned verifier discovers binding,
outperforms strong non-oracle alternatives, or transfers to a genuinely unseen
mechanism. All earlier files and raw results remain preserved, but any paper or
planning document must label v1 as a symbolic/identifiability pilot.

## Required correction

BindingCheck-CPU-v2 must remove the correct Fourier basis from learned inputs,
use raw action-result tokens, evaluate structurally different held mechanisms,
introduce partial and nonuniform probing, and use candidate normalized regret,
Top-1, paired seed improvement, and uncertainty as primary evidence. A challenge
that again yields TPA above 0.95 or near-zero regret will be rejected as too easy
rather than celebrated.
