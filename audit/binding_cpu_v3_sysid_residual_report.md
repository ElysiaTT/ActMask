# BindingCheck-CPU-v3 strong-baseline-first final report

Date: 2026-09-03
Frozen decision: **CPU_V3_NO_GO**
Method decision: **terminate the new-verifier method route**
Device boundary: CPU only; no PyTorch, CUDA, GPU simulator, or vision backbone.

## Executive conclusion

The hybrid system-ID gate/residual is a real but limited ID improvement, not a
robust method result. It passes every frozen gate on ID, lowering normalized
regret by 19.7% relative to local linear in all five seeds. That gain does not
transport: parameter OOD improves only 4.4%, sparse history improves 10.4% but
its confidence interval crosses zero, noise OOD degrades by 8.4% in all five
seeds, and held structure improves only 6.5% with severe undercoverage.

The preregistered terminal rule therefore requires `CPU_V3_NO_GO`. No model,
threshold, seed, or held mechanism will be changed to chase a pass. ActMask
should retain action-effect binding as a counterfactual audit benchmark and
drop the claim that the current learned verifier improves candidate selection.

## Frozen run and integrity

- Official seeds: 12001--12005.
- Train: 640 twins per seed, exactly 128 from each of five training families.
- Evaluation: 256 twins per split and seed.
- Meta-fit/calibration: 480/160 train twins.
- Preregistration SHA-256:
  `61e98c2e4ccfc7f7fce6318bfa1ac03f0d75f2195783aaca11b018d9fcf86011`.
- Total wall time: 479.54 seconds on CPU.
- All train and evaluation integrity checks passed.

The raw JSON retains every seed/split/model, per-twin normalized regret,
difficulty bins, intervention metrics, calibration metrics, component weights,
bootstrap intervals, gate values, and failures.

## Primary results

The strongest admission baseline is local linear on all five splits.

| Split | Hybrid regret | Local-linear regret | Relative change | Hybrid Top-1 | Local Top-1 | 95% CI, baseline minus hybrid | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| ID | 0.0339 | 0.0422 | +19.7% | 0.5797 | 0.5441 | [0.0041, 0.0125] | PASS |
| Parameter OOD | 0.0868 | 0.0908 | +4.4% | 0.4273 | 0.3914 | [-0.0017, 0.0091] | FAIL |
| Sparse history | 0.0873 | 0.0974 | +10.4% | 0.4066 | 0.3934 | [-0.0011, 0.0211] | FAIL |
| Noise OOD | 0.0740 | 0.0683 | -8.4% | 0.4090 | 0.4266 | [-0.0105, -0.0004] | FAIL |
| Held structure | 0.1824 | 0.1951 | +6.5% | 0.1914 | 0.1723 | [-0.0059, 0.0283] | FAIL |

Seed-level baseline-minus-hybrid regret differences were:

- ID: `+0.0080, +0.0088, +0.0112, +0.0028, +0.0109` (5/5 wins);
- parameter OOD: `+0.0087, +0.0023, +0.0043, -0.0023, +0.0069` (4/5);
- sparse history: `+0.0100, +0.0087, +0.0280, -0.0086, +0.0128` (4/5);
- noise OOD: `-0.0047, -0.0057, -0.0033, -0.0098, -0.0051` (0/5);
- held structure: `+0.0193, +0.0192, +0.0134, -0.0204, +0.0319` (4/5).

The candidate-wise oracle selector ceiling remains substantially better
(`0.0186`, `0.0397`, `0.0460`, `0.0337`, and `0.0859` regret), so there is
theoretical selector headroom. The learned diagnostics do not identify that
selector reliably under distribution shift.

## Binding and anti-saturation audit

All causal-use controls pass. Binding breaking increases hybrid normalized
regret by 0.4260, 0.3819, 0.3699, 0.3747, and 0.2446 across the five splits;
diagnostic TPA drops by 0.4335, 0.3833, 0.3659, 0.3840, and 0.2312.
Twin-history swap also causes large regret increases. Pair-preserving
permutation and candidate-slot reversal leave regret unchanged to displayed
precision.

Hybrid TPA is 0.9371, 0.8862, 0.8819, 0.8905, and 0.7382. Regret is nonzero on
every split. Thus v3 does not reproduce the v1 0.5-versus-near-1 saturation.
The benchmark detects binding use; the method nevertheless fails its utility
comparison.

## Uncertainty result

| Split | Nominal-80% coverage | Uncertainty/error Spearman | Mean response MAE |
|---|---:|---:|---:|
| ID | 0.8029 | 0.4746 | 0.0418 |
| Parameter OOD | 0.6550 | 0.5179 | 0.1120 |
| Sparse history | 0.7292 | 0.5360 | 0.0813 |
| Noise OOD | 0.7304 | 0.3805 | 0.0723 |
| Held structure | 0.4601 | 0.4450 | 0.1928 |

Uncertainty ranking remains moderately informative, but the single training
calibration factor does not transport. Parameter OOD and especially held
structure are severely overconfident. This is an independent preregistered
failure, not merely a side effect of the regret gate.

## What the hybrid learned

Mean component weights consistently favor local linear (0.354--0.467), followed
by Poly2/RBF; nearest and kNN receive less weight. This agrees with the frozen
baseline ranking. The residual improves all oracle-gap difficulty terciles on
ID and usually on sparse/held data, but it over-corrects noisy histories and
does not generalize its calibration to the held response.

The main lesson is not that adaptive system identification is impossible. It
is that the current training-only LOO/coverage/disagreement signals are
insufficient to guarantee robust selector or residual behavior outside their
training distribution. The fixed local-linear estimator is the safer method.

## Terminal paper-direction decision

The following method claims must be removed or not introduced:

- a generic history/memory encoder improves candidate selection;
- the spectral learned-v1 result demonstrates learned OOD generalization;
- the v3 hybrid robustly improves system identification;
- uncertainty calibrated on train transports to unknown mechanisms.

The defensible ActMask contribution is instead:

> **Counterfactual Action-Effect Binding Audit**: a two-track benchmark that
> separates causal-use integrity from decision usefulness.

Track A should use exact matched twins and pair-preserving versus
binding-breaking interventions to determine whether a verifier uses
action-effect correspondence. Track B should use non-saturated sparse/noisy/
held response tasks and require comparison against nearest, local linear,
Poly2, and RBF candidate regret. v1 belongs only to Track-A plumbing evidence;
v2 and v3 are retained negative method baselines for Track B.

No further synthetic model search is justified under this Goal. The next
scientific activity, if separately authorized, should be external validation of
the audit on existing verifier outputs or precomputed features, with the fixed
local-linear/RBF baselines carried forward. This result does not authorize GPU.

## Reproduction and artifacts

```powershell
python scripts\cpu_binding_v3_sysid_residual.py `
  --output audit\results\binding_cpu_v3_sysid_residual.json
```

- Preregistration: `docs/binding_cpu_v3_sysid_residual_preregister.md`
- Implementation: `scripts/cpu_binding_v3_sysid_residual.py`
- Raw result: `audit/results/binding_cpu_v3_sysid_residual.json`
- v2 terminal predecessor: `audit/binding_cpu_v2_antisaturation_report.md`
