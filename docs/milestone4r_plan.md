# Milestone 4R plan

4R is a separately versioned robust visual benchmark following frozen 4V
decision B. It uses four train, one validation, and two held-out test cameras;
six-frame RGB-D/point-cloud histories; C10 candidate sets; and missing-frame,
occlusion, timestamp, latency, and distractor-association conditions. Fair
models receive calibration but not camera or simulator identity. The protocol
is frozen before data generation and is limited to 7,680 paired GPU-PhysX
candidate executions, 20 GiB, and eight GPU-hours.

The frozen configuration hash is
`8b07ab21338da9bf9d39c8684b9bfb10e348ebe9a0bb4b4d27af17db1af47422`.

## Generator-validation status

The validated 11-world-per-family condition smoke is at
`outputs/actmask/milestone4r_robust_visual/generation_condition_smoke_v4/`.
Its matching and observation-integrity audit passes exact RGB/depth/visibility
unordered matching, current and final-frame matching, chronology difference,
shared masks/actions, all
seven camera calibrations, asynchronous modality masks, hidden-field
exclusion, target-only occlusion with a visible distractor, and independent
world-level background randomization. The target-only occluder uses renderer
segmentation only while constructing generator-private diagnostics; no
segmentation is written to fair bundles or supplied to fair models. The later
candidate-diversity audit is reported separately as the final stop condition.

The predecessor `robust_probe_v1/` is intentionally non-final: it was stopped
after one completed task family when a protocol audit found omitted appearance
randomization. No model was evaluated on it. The appearance correction was
made before the complete final probe and recorded in the config manifest; it
does not alter any label, gate, camera range, dropout rule, or budget.

## Final bounded probe and stop condition

`robust_probe_v2/` completed the frozen 3×128-world, paired-C10 probe with
7,680 final GPU-PhysX candidate executions. Its complete matching, camera,
mask, appearance, association, metadata, and grouped-split audit is recorded
in `robust_probe_v2/robust_audit.json`.

The same preregistered audit found a stop condition: all 256 histories in
`SignedMovingWindowPlacement` have a C10-aligned candidate set but none has
both a successful and unsuccessful candidate (`mixed_success_fraction = 0`).
It is therefore not a candidate-ranking benchmark for that family. This is an
integrity/candidate-diversity failure, so the fixed Phase-9 decision is E and
standard-model/oracle evaluation is intentionally skipped rather than treated
as evidence.
