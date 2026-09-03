# Proposed next Codex Goal: CPU-v3 strong-baseline-first verifier

> **Executed 2026-09-03:** frozen result `CPU_V3_NO_GO`. Only ID passed;
> parameter, sparse, noise, and held-structure gates failed. Per the terminal
> rule, the new-verifier method route is stopped. See
> `audit/binding_cpu_v3_sysid_residual_report.md`.

## Goal

Test whether an adaptive system-identification residual/uncertainty verifier can
improve candidate-action normalized regret beyond the strongest fixed poly2 and
RBF baselines from BindingCheck-CPU-v2. Continue CPU-only; do not start GPU.

## Motivation fixed by v2 evidence

The raw paired ExtraTrees verifier uses action-result binding but loses to
poly2/RBF on every seed and split. Therefore the next model must start from
those strong predictions rather than relearn system identification from padded
tokens. This is an independent new hypothesis, not a reinterpretation or
retuning of v2.

## Required design

1. Use fresh seeds and freeze all data, models, thresholds, and bootstrap rules
   before observing v3 results.
2. Build heterogeneous train/test mixtures in which smooth, local nonlinear,
   threshold, saturation, and mild nonstationary mechanisms coexist; retain a
   genuinely structure-held-out family.
3. Compute only generic per-history predictors: nearest/kNN, poly2, fixed RBF,
   and local-linear fits. No generator-matched latent or Fourier features.
4. Train a small CPU gating/residual model only from training histories, using
   baseline predictions, leave-one-probe-out errors, candidate coverage,
   disagreement, and local result variance. It may select/blend/correct the
   predictors but may not see mechanism labels or latent parameters.
5. Compare against every fixed component, their uniform ensemble, raw paired
   v2, unbound controls, and an oracle selector reported outside admission.
6. Primary evidence is per-twin normalized regret, Top-1, calibration/coverage,
   five-seed paired differences, and a hierarchical-bootstrap confidence
   interval. TPA remains diagnostic.
7. Retain pair-preserving, binding-breaking, twin-swap, sparse-history, noise,
   and structure-held-out interventions.

## GO/NO-GO principle

GO only if the hybrid lowers regret by at least 10% relative to every fixed
non-oracle baseline on ID and all challenge splits, wins at least four of five
seeds per split, has a strictly positive bootstrap lower bound, and is neither
saturated nor insensitive to binding. Otherwise retain the full failure and
drop the new-method claim; continue only with the benchmark/audit contribution.

Passing would authorize a new external CPU evaluation plan, not GPU use.
