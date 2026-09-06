# Bounded repository improvement task

Date: 2026-09-06. Continuation authorized after CPU G1/G2 freeze `e672a1c`.

## Working prompt

Improve the existing ManiSkill CPU transport verifier, not experiment scores.
Find cases where corrupted, incomplete or mismatched artifacts could receive
a passing report. Add fail-closed validation and adversarial regression tests.
Keep physics, sample sizes, utility and scientific admission thresholds fixed.
Verify deterministic regeneration and actual CPU replay. Explicitly review
and update the source freeze, push GitHub, and require Linux CPU CI to pass.
Do not access SSH hosts, install CUDA, train models or start GPU compute.

## Acceptance

- Every artifact is inventoried; public export contains exactly one allowlisted
  input file. Missing, additional and escaped paths fail.
- Raw trajectories, snapshot properties and action/outcome dimensions are finite
  and valid before replay. Invalid snapshots do not mutate the environment.
- Twin mass, inertia, anchor and clock match; COM is the only hidden difference.
- Attempt IDs, seeds, acceptance and retention agree with raw outcomes and
  dataset rows. Rejected attempts still have raw data and validated snapshots;
  only accepted trajectories receive full physics replay.
- Match complete height/tilt vectors, not independent coordinate marginals.
- The intervention suite's original dataset digest equals the replay dataset.
- Rehashing a corrupted artifact does not bypass semantic validation. Hashes
  are integrity checks, not signatures or an untrusted-code security boundary.
- Reports cannot overwrite evidence or be written into the input dataset.
- Same seeds and same runtime regenerate the same typed dataset digest.
- Preserve old archived evidence; publish new evidence for this revision.

## Stop boundary

Stop after tests, CPU smoke, reviewed freeze and GitHub CI pass. This improves
engineering confidence only. G3 scientific dataset admission, G4 model studies,
GPU backend validation and any claims of learned gains remain separate work.
