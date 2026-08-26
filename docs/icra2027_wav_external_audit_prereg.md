# ICRA 2027 external WAV MiniGrid audit: frozen protocol

Frozen on 2026-08-09 before running any trained single-frame control.

## Question and scope

This audit asks one narrow question: on the public WAV MiniGrid inverse-dynamics
state-complexity benchmark, does the reported physical-outcome score require
the observed transition, or can a classifier that receives only one state
obtain comparable performance? It does **not** claim to reproduce WAV's robot
experiments, improve WAV, or audit the Action Following Score experiment.

The benchmark input is a two-state tuple `(s_t, s_{t+1})` plus carried-object
state, and the target is one of seven actions. The official dynamic-accuracy
metric applies the predicted action to the public MiniGrid physics oracle and
scores the changed channels against the logged next state. Consequently,
action accuracy and official dynamic accuracy will both be reported; the latter
is primary.

## Frozen upstream inputs

- Repository: `world-action-verifier/wav_minigrid`
- Commit: `527159b06149beacfb3b2d77af7d938ca4efa32d`
- Official SparseIDM checkpoint SHA-256:
  `2d0ac5c684c90fc08ed4e280f8944b5d3f8f2be1c1ff8c990eeb708eb2bd67d1`
- Official o6 training data SHA-256:
  `38304df2e59effecd664880fa3e708ef0d3fcf3bd2986a63ca395efaf7c01378`
- Official o6/o8/o10/o12/o14 test SHA-256 values, in order:
  `288534d7575aeb0c95fe2301cc6db87806e7d48f27471b578e9b5b75dcb1b9c0`,
  `1a4974fde645cce28a1e4679ad3442934ccc23dbf2fb4eb81181fc00fcb7683c`,
  `ab4900f995101caf261c90d21435692714444c97dced3a21eee449263891370f`,
  `8602272e50412a7526f7c77b4ad961407ac3f92238271a0d0b86ba5776699ea3`,
  and `7986f1fbb962b7d4cb0c7e6f5bda044856370946a35c4e376d69dca1df74f3a7`.
- Upstream `idm.py` SHA-256:
  `7a7b4135d9528aed69f6d44228ead2ff6fa3d82cd8b78e782093051fd821bb72`
- Upstream physics-oracle source SHA-256:
  `9b0d950a8f8d2c8e7f556fca0747b8113de8bb7ed6c8a52f2eed00f6a8adbcc1`

The already completed feasibility run reproduced the official evaluation path
and is not used to choose any control architecture or threshold.

## Frozen cells

All learned controls use the official o6 training set only. A deterministic,
class-stratified 80/20 development split is made separately for seeds
`9102, 9103, 9104, 9105, 9106`. Test files are never used for checkpoint or
hyperparameter selection.

1. **Train-prior:** always predict the most frequent action in that seed's
   training partition.
2. **Current-only:** duplicate `(s_t, c_t)` into both input slots of a matched
   two-slot CNN and train it from scratch.
3. **Next-only:** duplicate `(s_{t+1}, c_{t+1})` into both slots of the same CNN
   and train it from scratch.
4. **Paired:** train the same CNN on the unmodified
   `(s_t, c_t, s_{t+1}, c_{t+1})` tuple.
5. **SparseIDM:** evaluate the frozen official checkpoint on the unmodified
   pair. Because its mask uses hard Gumbel sampling even in evaluation mode,
   report the same five inference seeds.
6. **Transition rule:** a fixed, non-learned decoder based only on agent pose,
   the agent-front cell, and carried-object changes. Physically equivalent
   pickup/swap outcomes are allowed because the primary metric itself identifies
   actions only through their executed next state. No test labels may be used.

The matched CNN receives one-hot object, color, and state channels for each of
the two slots plus one-hot carried-object/color values. Architecture and
optimizer are identical across current-only, next-only, and paired cells. The
only change is which observable fills the two slots. Training uses Adam at
`1e-3`, batch size 64, at most 200 epochs, validation action-accuracy selection,
and patience 25. No test-time augmentation or per-complexity tuning is allowed.

## Frozen reporting and decision rule

For every cell and o6/o8/o10/o12/o14 test set, report action accuracy and the
official dynamic accuracy. Learned-cell tables report mean and sample standard
deviation over five seeds. Also report macro averages over the five test sets;
these averages do not replace per-complexity values.

The benchmark passes this external transition-necessity check only if:

1. paired mean dynamic accuracy exceeds the stronger of current-only and
   next-only by at least 0.10 on **every** test complexity;
2. the official SparseIDM has at least 0.90 mean dynamic accuracy on every test
   complexity; and
3. the transition rule exceeds the strongest single-frame learned control on
   every test complexity.

If any condition fails, the result must be reported as a failure or ambiguity,
not removed. This audit is admitted to the manuscript only if the script,
frozen source hashes, raw per-seed predictions or sufficient confusion counts,
and an automated verifier can all be included in the anonymous artifact. A
passed audit supports only the claim that this public MiniGrid benchmark
requires transition evidence under these controls; it does not validate the
paper's synthetic generator or establish real-robot verification.
