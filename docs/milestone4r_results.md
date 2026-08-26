# Milestone 4R results

The protocol is frozen with SHA-256
`8b07ab21338da9bf9d39c8684b9bfb10e348ebe9a0bb4b4d27af17db1af47422`.
All four train, validation, and two held-out test camera poses passed the
renderer preflight: 128×128 depth is finite, has nonzero workspace support,
and each sensor exposes 3×3 camera intrinsics plus 3×4 extrinsics. This was
the pre-generation qualification; final probe evidence is reported below.

The v4 all-condition smoke completed successfully. Its three-family matching
sub-audit shows 110 C10 pairs per family with identical current frame, final
two frames, temporal masks, candidate actions, split, and exact unordered
RGB/depth/visibility frame multisets, while chronology differs. All seven
cameras passed calibration; target-only occlusion leaves the similar distractor
visible; fair bundles contain no segmentation/identity; and world appearance
is randomized independently of labels. This smoke is validation only, not an
evaluation result.

The complete `robust_probe_v2` has finished: 3 task families × 128 worlds,
256 paired histories and 2,560 candidate records per family, for 7,680 final
candidate executions. It occupies 16,452,729 bytes (about 16 MiB), far below
the 20 GiB cap; final-probe wall time was below 18 minutes.

All pair matching is exact: in every task family, all 1,280 candidate pairs
match current RGB-D-visibility frame, final two frames, unordered RGB/depth/
visibility frame multiset, temporal masks, candidate action, TCP state and
split; chronology differs in all pairs. All seven calibration cameras, all
eleven observation conditions, asynchronous masks, target-only occlusion,
visible distractors, background randomization, hidden-field exclusion and
grouped C10 alignment pass. Background/label correlations are near zero
(-0.000, -0.000, and 0.025 for the three families).

The required candidate-diversity audit fails in
`SignedMovingWindowPlacement`: all 256 histories are C10-aligned but its
mixed-success fraction is 0.0. Its candidate actions therefore cannot be
meaningfully ranked. The frozen Phase-9 decision is consequently **E.
BENCHMARK INVALID**; no fair baseline, state oracle, segmentation oracle, or
Milestone 4M model is run. This is a scientific stop condition, not a negative
model result.

The predecessor v1 output remains non-final and unevaluated. Including its
known and possible in-flight work, total candidate execution is bounded by
12,800, still below the 20,000 global ceiling.
