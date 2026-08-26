# Milestone 3R-NL Results

## Scope and provenance

This separately versioned state-only GPU-PhysX extension leaves frozen 3Q and
3R artifacts unchanged.  Its pre-generation configuration hash is
`79d04d3204af01386d1a178a874d537a345bd2a8ca94545e750fcb4d50209b9f`.
All primary artifacts are below `outputs/actmask/milestone3r_nl_v1/`.

The pre-model corrections are recorded in `physical_corrections.json`: one
common collision-geometry correction for both mechanisms, then one common
drive adjustment for damping/drive and one common acceleration adjustment for
hysteresis.  Thus each mechanism consumed its two allowed corrections.  The
two fairness fixes are also recorded: matching the unordered early-frame
multiset, and resetting decision state/elapsed time for each candidate.

## Probe

The final probe contains 160 accepted strict pairs (320 examples): 32 pairs in
each accepted family--mechanism cell, with damping/drive accepted in capture
and container, hysteresis accepted in all three families, and no accepted
damping/drive rotating-slot pair.  The latter cell was excluded before full
generation after its correction budget was exhausted.  The audit found zero
matching error at tolerance `1e-6` and minimum earlier-history difference
`0.1139873`.

Probe validation passed: LastTwo = 0.50, static = 0.50, unordered = 0.50, and
both GRU and TCN reached 1.00 pair order across the fixed three seeds.

## Full GPU-PhysX data

`full_c20/` was generated on the RTX 4090 D from the five pre-fixed viable
family--mechanism cells.  It used exactly 100,000 branch executions
(500 worlds per retained cell, C20, two branches):

| Quantity | Result |
| --- | ---: |
| attempted / accepted strict pairs | 50,000 / 50,000 |
| examples / label balance | 100,000 / 0.500 |
| maximum matching error | 0 |
| minimum earlier-history difference | 0.1139873 |
| generation time | 29.80 s |
| C5 / C10 derived examples | 25,000 / 50,000 |

C5 and C10 were materialized as aligned prefixes without additional simulator
execution.  The strict audit covers current state, both final frames,
finite-difference input, action/horizon, timestamps, TCP and static features;
only earlier ordered history differs.

## Fair evaluation

The full evaluator accepts only the preregistered observable input keys.  On
the grouped test split, LastTwo pair order is 0.500.  The strongest fair
analytic baseline is RobustLinearVelocity at 0.771.  Validation selected the
GRU; five fixed confirmation seeds yield pair order 1.000.  Its grouped
learned-minus-analytic pair-order estimate is +0.058, 95% CI
[+0.0495, +0.0666] over 10,000 strict pairs.  GRU/TCN solve the pair problem,
whereas unordered-history, static-only and action-only controls remain 0.500.

Full history minus final-two-only pair order is +0.500.  However, exact
history reversal remains 1.000, so the reversal causal drop is **0.000**;
this fails the preregistered >=0.15 causal gate.  Removing or shuffling
timestamps also leaves performance at 1.000.  C20 selected-GRU inference p95
is 0.516 ms, well below 200 ms.

The C5/C10/C20 prefix scores are all 1.000 for the selected model.  They are
not evidence of diverse action-effect ranking: every candidate pulse in a
world retained the same branch outcome.  Ranking is therefore over strict
counterfactual pair members, and stable score ties make LastTwo top-1 appear
artificially high despite its chance pair order.  Pair-order is the reliable
primary statistic here.

## Final decision: E — HUMAN DECISION REQUIRED

The benchmark has a valid matched-final nonlinear-history gap and a learned
advantage over the strongest fair analytic baseline, but it **does not pass
the nonlinear OOD gate**.  Two material blockers remain:

1. Independent nonlinear parameter OOD sets were not generated within the
   fixed 100,000-execution first full run, so no OOD-pass claim is valid.
2. The selected temporal model is insensitive to exact reversal, failing the
   preregistered causal requirement; and candidate actions do not yet create
   outcome diversity within a world.

Resolving either requires a new explicitly versioned pre-registration and
additional GPU-PhysX budget.  No visual/Vulkan/RGB-D work is authorized by
this result.

## Tests and legacy visual status

The focused nonlinear state-only regression set passed: 8 tests in 3.30 s
(`test_milestone3r_nl_*` plus 3R metrics).  `legacy_visual_test_status`:
**not run**; no failure or timeout was produced because the unrelated legacy
renderer is out of scope for this state-only extension and was not modified.
