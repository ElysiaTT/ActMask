# Milestone 2E Results

**Technical branch:** `STOP CANDIDATE-VERIFIER LINE`.

**Project decision:** `UNCHANGED: NO-GO FOR GPU SIMULATOR INTEGRATION`.

## Frozen protocol

- CPU only; grouped disjoint train/validation/test candidate worlds; five deterministic seeds.
- C5/C10/C20/C50, validation-only epoch selection and Platt calibration; score is the frozen 0.5 learned / 0.5 MultiHypothesis hybrid.
- GT identity is diagnostic-only and excluded from fair comparisons; transformed tests never select a checkpoint.

## Scoring audit

- Passed: `True`; candidate labels/IDs are aligned and equal-score ties never use candidate ID.
- Selected fair arm from validation only: `CoordinateInvariantMutualNN + ActionFrameRelativeBackbone`.

## Correspondence formulas

- Old diagnostic-only rule: `|dx| + 0.28 * ||d_yz||`.
- New primary rule: with anchor `a_t`, observable motion `v`, and earlier frame time `s`, choose mutual nearest source `j` minimizing `||a_t - v(t-s) - x_{s,j}||_2`, accepting only confidence and normalized-distance passing matches; otherwise source index is `-1`.
- Soft arm uses the same full-3D predicted cost in a confidence-gated soft assignment; no axis is privileged.

## Main result

- Estimated-vs-GT diagnostic C20 Top-1 gap: `-0.3333333333333333`.
- Negative control detects both an axis permutation and rotation: `True` / `True`.

Detailed factorial, matching, ranking-invariance, OOD, latency, per-seed and paired-bootstrap results are stored in the JSON/CSV files beside this summary.
