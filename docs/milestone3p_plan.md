# Milestone 3P Plan: Static-Geometry Shortcut Audit

## Purpose

Determine whether the prior dynamic candidate-verifier result is explained by
temporal reasoning or by static geometry/action cues. Stage A is strictly CPU
only: no CUDA execution, simulator, RGB-D asset, pretrained weight, external
data, package installation, or download is permitted.

## Stage-A protocol

1. Audit the frozen 2E evidence, label construction, grouped splits, and the
   strict observable/hidden-state separation.
2. Derive an observable-only static table: current point geometry, start/end
   and segment distances, clearance, path length, timing, centroid-relative
   path, scene summary, action, and an explicit template diagnostic.
3. Evaluate static methods (current position, endpoint, analytic, logistic,
   shallow tree, two-layer MLP, template-only, scene-only, action-only) and
   temporal methods (time-aligned analytic, two-layer MLP, frozen prior
   verifier) with three seeds; confirm the two learned compact methods with
   five seeds.
4. Keep three distributions separate: original, static-balanced, and exact
   matched counterfactual worlds. Matched worlds hold current geometry, action,
   duration, and static feature vector exactly constant while changing only
   observable history/hidden dynamics and the label.
5. Require a negative static-shortcut control, a positive matched-dynamic
   control, corruption tests, paired bootstrap inference, no-leakage checks,
   and a full regression suite before any GPU work.

## Stage-B authorization rule

Stage B may start only if all ten Stage-A gates pass: strong original static
performance; static collapse on matched worlds; temporal advantage of at least
0.10 with a paired confidence interval; at least two temporal corruptions
hurt; exact matching/no leakage; both controls; at least three dynamic
families; strict observable separation; and all tests passing. If authorized,
the first action is only a local GPU/resource audit. No simulator is selected,
installed, or downloaded without a human decision.
