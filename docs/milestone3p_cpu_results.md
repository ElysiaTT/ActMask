# Milestone 3P CPU Results

## Decision

**GO TO STAGE B LOCAL GPU/RESOURCE AUDIT.** All ten CPU-stage gates passed.
This authorizes local environment and resource inspection only; it does not
authorize a simulator install, download, or benchmark run.

## Prior evidence

The frozen 2E temporal diagnostic had learned Top-1 0.9572 with correct
history but 0.9944 with static geometry only. The source audit confirms that
future trajectories are hidden-state/target data, not observable inputs.

## Static shortcut audit

On the original candidate distribution, `CurrentPositionProximity` and the
static two-layer MLP both reached Top-1 1.0000. The strongest static
correlations were action duration (0.711 bits of mutual information), nominal
action delay (0.420 bits), and path-distance proxy (0.180 bits). This is direct
evidence that the original distribution contains static/action shortcuts.

A held-out permutation audit confirms that this is not only a neural-model
artifact: permuting action duration decreases static-MLP Top-1 by 0.1667 and
increases normalized regret by 0.1667; path-distance proxy is next (Top-1
decrease 0.0694). The new benchmark metrics are explicitly tagged as 3P
artifacts rather than reusing old 2E result values.

The exact matched counterfactual test removes those shortcuts. Current-position
Top-1 fell to 0.2708 (normalized regret 0.7292); the static two-layer MLP was
0.1597. Exact static match differences were zero and static Bayes accuracy on
label-flip pairs was exactly chance, 0.5.

## Dynamic counterfactual evidence

The observable time-aligned baseline achieved Top-1 0.8542 on the matched
test, versus 0.2708 for the preregistered static comparator: paired delta
0.5833, 95% bootstrap CI [0.4167, 0.7292], across 48 candidate sets. Its
counterfactual pair accuracy was 0.9167, while the static comparator was 0.5.
The five-seed confirmation gives 0.9167 ± 0.0147 for the temporal MLP versus
0.1750 ± 0.0238 for the static MLP.

History reversal, mismatched histories, removed histories, shuffled motion,
and zeroed motion caused Top-1 drops of 0.6250, 0.7292, 0.5833, 0.5625, and
0.5833 respectively. The deliberately shortcut-heavy negative control was
solved statically at 0.9167. The positive matched-dynamic control passed.

The frozen previous dynamic verifier was also evaluated under all seven history
views on the matched benchmark. It does not transfer as a temporal solver:
its mean Top-1 changes under the corruptions are between -0.0250 and 0.0000.
This negative result is expected for an out-of-distribution frozen verifier and
does not alter the positive evidence from the explicit time-aligned baseline.

## Anti-shortcut ablation

All five lightweight interventions retain high matched performance
(Top-1 0.8958–0.9236; pair accuracy 1.0000) and lose 0.6250–0.6389 Top-1 when
matched history is removed. None improves on ordinary matched-pair sampling;
the best matched score remains ordinary sampling (0.9236). On the shortcut-
heavy original distribution, every model trained only on the matched regime is
low (0.3532–0.4889), and original history removal has zero effect. Thus the
interventions do not yet lower a measured original shortcut reliance beyond
the distribution shift itself, and no preferred anti-shortcut method is claimed.

## Integrity and verification

All Stage-A computations used CPU only. The initial suite passed 130 tests,
the completion extension passed 133 tests, and final verification passed
**134 tests in 36.03 seconds**. The 14 warnings are existing
matplotlib/PyParsing deprecations only.

Detailed immutable artifacts are in
`outputs/actmask/milestone3p_cpu_shortcut_audit/`, including the method reports,
correlations, tables, figures, corruption tests, ablation results, final gate
matrix, `permutation_feature_importance.json`,
`frozen_dynamic_verifier_corruptions.json`, and
`anti_shortcut_training_extension.json`.
