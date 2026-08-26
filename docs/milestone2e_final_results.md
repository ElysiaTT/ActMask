# Milestone 2E-Final Results (v2)

## Decision

**STOP LEARNED CANDIDATE-VERIFIER LINE.** Do not start Stage B GPU simulator
integration. The learned hybrid is significantly weaker than the
validation-selected fair analytic comparator, only one dynamic OOD axis is
favourable, and the temporal diagnostic exposes a static-geometry shortcut.

## Protocol and integrity

- Primary verifier: `SoftCoordinateInvariantCorrespondence + ScalarInvariantFeatureBackbone`.
- Five frozen seeds: 1601, 2603, 3607, 4611, 5613; corrected metric schema:
  `milestone2e-ranking-v2-standard-ndcg-utility-gain`.
- A2/A3/A5, A6 and the original test run remained CPU-only. A4 began CPU-only,
  was explicitly stopped before producing an artifact, then completed with the
  separately versioned GPU qualification path. Its five-seed utility logits
  passed CPU--CUDA `allclose(atol=1e-5, rtol=1e-4)` and C20 rankings were
  identical on the checked frozen batch.
- All output is new under `outputs/actmask/milestone2e_final_v2/`; legacy
  incremental ranking/NDCG material was not read as corrected-metric input.

## A3: invariance and corruption checks

At C20, SO(3), axis permutation and sign flip had exactly zero mean Top-1 and
normalized-regret deltas. Group-shared point permutation was also exactly
invariant. Per-frame resampling had Top-1 delta +0.0016 (95% CI
[-0.0068, 0.0096]) and occlusion -0.0004 ([-0.0080, 0.0072]).

## A5: ranking and OOD

At C20 the hybrid is useful in absolute terms: Top-1 0.9784, normalized regret
0.02392 and corrected NDCG 0.98356. It nevertheless loses to the frozen
validation-selected fair `CurrentPositionProximity` comparator: hybrid minus
fair Top-1 = -0.0156, 95% CI [-0.0244, -0.0072]; regret = +0.01832, 95% CI
[0.00907, 0.02760]. Thus the fair-advantage gate fails.

Of the five preregistered dynamic axes, only acceleration is favourable on both
decision metrics (Top-1 +0.0152, 95% CI [0.0080, 0.0240]; regret -0.02451,
[-0.03277, -0.01715]). Curvature is significantly adverse (Top-1 -0.0536;
regret +0.0536), velocity magnitude is adverse on regret (+0.00685,
[0.00152, 0.01316]), and observation/action delay intervals cross zero for the
decision metrics. The required two favourable dynamic axes are absent.

## A4: temporal and shortcut diagnostic (GPU-qualified)

The completed GPU report evaluated all four validation-promising arms, all 13
fixed views and all five seeds. For the selected primary learned-only score,
correct history achieves Top-1 0.9572 and regret 0.04716. Reversing history
lowers Top-1 by 0.7856 and raises regret by 0.84732; independently permuting
frame histories lowers Top-1 by 0.2288 and raises regret by 0.2560.

It does not pass the stronger shortcut-control gate: static geometry only
achieves Top-1 0.9944 (better than correct-history 0.9572), while action only
is 0.0168 and candidate-template only is 0.0128. The primary therefore fails
the fixed requirement to exceed each control by 0.05. This is decisive evidence
of a static geometric shortcut despite genuine history sensitivity.

## A6: completion extension

Five-seed micro mask metrics: AP 0.23352, IoU 0.16715, PR-AUC 0.23350,
precision 0.18768 and recall 0.61419. Correspondence accuracy is 0.44702
overall (static 0.68821, dynamic 0.28762), with median error 0 and p95 error
0.07757. Point-permutation ranking deltas are exactly zero at C5/C10/C20/C50.

All C20 seed p95 latencies meet the 200 ms gate (mean 15.30 ms; worst observed
21.22 ms); all seeds meet both 5 Hz and 10 Hz checks.

## A7: tests

The post-GPU full regression suite passed: **123 passed, 14 warnings in
34.73 s**. The warnings are existing matplotlib/PyParsing deprecations only.

## Final gate matrix

| Gate | Result | Status |
| --- | --- | --- |
| Fair C20 advantage | Comparator significantly stronger | Fail |
| Two favourable dynamic OOD axes | Acceleration only | Fail |
| Temporal causality | Reversal/per-frame corruption degrades score | Pass |
| No static/action/template shortcut | Static-only exceeds complete-history | Fail |
| Invariance | Rigid-transform deltas are zero | Pass |
| Useful absolute C20 score | Top-1 0.9784; regret 0.02392 | Pass |
| C20 latency, cache audit, tests | p95 < 200 ms; 123 tests pass | Pass |

The failure gates rule out both `GO TO GPU SIMULATOR INTEGRATION` and an
analytic pivot: although the analytic comparator is stronger, the learned
candidate verifier has not cleared the dynamic-OOD and shortcut controls. No
Stage B action is authorized.
