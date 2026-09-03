# BindingCheck-CPU-v2 anti-saturation report

Date: 2026-09-03
Frozen decision: **CPU_V2_NO_GO**
Device boundary: CPU only; no PyTorch, CUDA, simulator, or GPU process used.

## Outcome

The user's saturation concern was correct. Once the generator-matched Fourier
basis and algebraically identical unbound features were removed, the apparent
near-perfect learned result disappeared. The v2 challenge itself is
non-saturated, but the generic raw-history ExtraTrees verifier is not
competitive with simple per-history system identification.

The preregistration hash stored in the raw result is
`1608f618c328fe6f654f6b4ccee7d4dd5f35f0dddbf3ec35e6fac94f4f1832a0`.
The one frozen five-seed run took 43.54 seconds.

## Primary results

| Split | Paired raw regret | Strongest baseline | Baseline regret | Paired Top-1 | Baseline Top-1 | Paired TPA |
|---|---:|---|---:|---:|---:|---:|
| ID | 0.1219 | poly2 ridge | 0.0175 | 0.2434 | 0.6703 | 0.8704 |
| Parameter OOD | 0.2116 | poly2 ridge | 0.0647 | 0.2055 | 0.4887 | 0.8062 |
| Noise OOD | 0.1343 | poly2 ridge | 0.0423 | 0.2219 | 0.5102 | 0.8613 |
| Held structure | 0.2049 | RBF ridge | 0.1214 | 0.1633 | 0.3238 | 0.7560 |

All four splits pass the anti-saturation checks: paired TPA is below 0.95 and
paired normalized regret is above 0.01. The outcome is no longer restricted to
0.5 or nearly 1.0.

However, paired raw relative regret “improvement” is negative on every split:
`-595%`, `-227%`, `-218%`, and `-69%`. It loses to the strongest baseline in
all 20 seed/split comparisons. The hierarchical-bootstrap 95% intervals for
baseline-minus-paired regret are also strictly negative:

- ID: `[-0.1133, -0.0966]`;
- parameter OOD: `[-0.1622, -0.1306]`;
- noise OOD: `[-0.1042, -0.0810]`;
- held structure: `[-0.0957, -0.0715]`.

This is decisive evidence against claiming the current raw ExtraTrees verifier
as an improvement.

## Binding is used, but that is insufficient

The paired model does use action-result correspondence:

| Split | Original regret | Binding-broken regret | Original TPA | Binding-broken TPA | Twin-swap regret |
|---|---:|---:|---:|---:|---:|
| ID | 0.1219 | 0.3804 | 0.8704 | 0.5827 | 0.5478 |
| Parameter OOD | 0.2116 | 0.4323 | 0.8062 | 0.5785 | 0.5703 |
| Noise OOD | 0.1343 | 0.3761 | 0.8613 | 0.5883 | 0.5245 |
| Held structure | 0.2049 | 0.3752 | 0.7560 | 0.5625 | 0.4462 |

Pair-preserving permutation changes neither features nor aggregate metrics, and
candidate-slot reversal changes regret by less than `0.00013`. Thus binding
information is causally relevant to the raw model. But a model can rely on the
right information and still implement a worse estimator than a simple
quadratic or RBF fit. Binding sensitivity alone is not evidence of useful
candidate ranking.

## Why raw ExtraTrees loses

The candidate-relative raw representation is generic, but its canonically
sorted slots do not have stable semantics under variable history length and
nonuniform action coverage. A tree must rediscover local interpolation and
episode-level system identification from many unstable scalar splits. In
contrast, poly2 and RBF ridge fit each history directly and exploit the scarce
probes efficiently. The held dead-zone/saturation law hurts the global
quadratic fit, but fixed RBF interpolation remains substantially better than the
raw learned model.

This is a representation/sample-efficiency failure, not evidence that binding
is irrelevant. The unbound raw model is consistently better than no-history
controls yet much worse than paired raw; the decisive comparison is that both
are worse than strong system-identification baselines.

## Direction decision

Stop the “generic memory encoder should outperform” method claim. Preserve
action-effect binding as an audit/benchmark question, and make strong
per-history system identification the starting point rather than an optional
baseline.

The next method hypothesis, if pursued, should be narrower:

> A verifier should adaptively combine a transparent local/global system-ID
> prediction with learned residual and uncertainty estimates; history tokens
> are useful only when they improve candidate regret beyond fixed poly/RBF
> estimators under sparse, nonstationary, or multimodal response histories.

The next CPU preregistration must compare against the best v2 poly/RBF results,
use fresh seeds and heterogeneous mechanism mixtures, train any gating/residual
only on training histories, and require positive paired confidence intervals.
If that hybrid also fails, the paper should remain a counterfactual audit
benchmark without a new verifier-method claim.

## Evidence and reproduction

```powershell
python scripts\cpu_binding_v2_antisaturation.py `
  --output audit\results\binding_cpu_v2_antisaturation.json
```

- Preregistration: `docs/binding_cpu_v2_antisaturation_preregister.md`
- Implementation: `scripts/cpu_binding_v2_antisaturation.py`
- Raw per-seed/per-twin result: `audit/results/binding_cpu_v2_antisaturation.json`
- Saturation diagnosis: `audit/binding_cpu_learned_v1_saturation_diagnosis.md`
