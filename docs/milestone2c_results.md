# Milestone 2C results (CPU-only)

## Decision

`NO-GO FOR SIMULATOR INTEGRATION`

The required CPU-only study completed, but the selected correspondence-robust model did not outperform the validation-selected fair geometric comparator on ID or OOD mask AP. GPU simulator integration is therefore not authorized by this milestone.

## Verified findings

- Frozen 2B seed-1401 checkpoint regression reproduced AP `0.9017510431`.
- `LocalNeighborhoodTemporalActMask` was invariant to independently permuted frame order: validation AP drop `0.0000`; the index-based model dropped `0.6127`.
- On independent 500-group correspondence evaluation, clean and per-frame-permuted AP were both `0.6230`; resampling retention was `0.8566` (gate `>=0.85`).
- The nine-loss validation ablation selected `mask_only`. None of the six auxiliary terms showed a reproducible improvement on its intended validation metric, so none was retained.
- On 500 held-out 20-candidate worlds, the validation-selected hybrid utility improved top-1 success by `+0.342` and reduced regret by `64.6%` versus MultiHypothesisTrajectoryProximity; paired CIs excluded zero.
- ID AP was lower than the fair comparator by `-0.2508`, 95% paired CI `[-0.2587, -0.2433]`; all nine OOD AP deltas were also significantly negative.
- Action counterfactual matching was `0.7756`, below the pre-registered `0.80` gate. Velocity matching (`0.8960`) and irrelevant stability (`0.9466`) passed.
- Shared 10-candidate CPU p95 latency was `5.27 ms`, exceeding the 5 Hz requirement.

## Artifacts

The complete machine-readable evidence is in `outputs/actmask/milestone2c/summary.json`; tables, figures, checkpoints, and deterministic manifests are colocated in that directory.
